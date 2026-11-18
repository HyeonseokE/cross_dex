"""What is the residual the shared space leaves out? (RSS2027 problem 2, step 1; dexlatent venv, CPU)

Every demo frame of the joint pi0.5 set (wrist 9 | fingers normalized in model order, zero-padded): shared part
s = D_h(E_h(q*)), residual r = q* - s. Fingertip residual = FK(q*) - FK(s) in the canonical hand frame (mm).
  where   residual size over normalized episode time (10 bins) and by finger closure (quartiles of mean flexion)
  smooth  share of the residual's energy above a 9-frame (0.15 s at 60 fps) moving average, vs the same share of the
          demo fingertip motion itself (teleop jitter reference); lag-1 autocorrelation
  from z  how much of r a function of the shared latent alone explains (R^2 on held-out episodes; ridge and a small MLP):
          high -> the residual is a fixed map of z (a decoder fine-tuned on task data would carry it);
          low  -> it needs the observation / context
  shared  per task, mean fingertip residual curve (thumb, index, middle; 20 time bins) of each hand; cosine similarity
          between hands (common part across hands = something the shared space should have carried)

  python -m cross_dex.ours.diag.residual_char --root <xlvla_joint set>/all --latent_ckpt <vae .pt> [--json out]
"""
import argparse, glob, json

import numpy as np
import pandas as pd
import torch


def r2(pred, y):
    return float(1 - ((pred - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--latent_ckpt", required=True)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    from dexverse_xlvla.eval_latent import load, enc, dec
    tr = load(a.latent_ckpt)
    for m in tr.autoencoders.values():
        m.eval()
    cells = json.load(open(f"{a.root}/meta/xlvla_cells.json"))["cells"]
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "frame_index", "action"])
                    for f in sorted(glob.glob(f"{a.root}/data/*/*.parquet"))]).sort_values(["episode_index", "frame_index"])
    act = np.stack(df["action"].to_numpy()).astype(np.float32); ep = df["episode_index"].to_numpy()
    per = len(np.unique(ep)) // len(cells)
    torch.manual_seed(0)
    out, curves = {}, {}
    for h in dict.fromkeys(c[1] for c in cells):
        M = tr.hand_models[h]; n = M.dof_count(); span = (M._upper - M._lower).numpy()
        cid = np.array([cells[e // per][1] == h for e in ep]); task = np.array([cells[e // per][0] for e in ep])[cid]
        q = torch.tensor(act[cid, 9:9 + n]); e_h = ep[cid]
        with torch.no_grad():
            z = enc(tr, h, q); s = dec(tr, h, z).clamp(-1, 1)
            tq, ts = M.forward(q), M.forward(s)
        r = (q - s).numpy(); rt = ((tq - ts) * 1000).numpy()                 # normalized residual, tip residual mm (B, K, 3)
        rmm = np.linalg.norm(rt, axis=-1).mean(-1)                            # mean over fingertips
        res = {"frames": int(cid.sum()), "res_tip_mm": float(rmm.mean()),
               "res_joint_deg": float(np.rad2deg(np.abs(r) * 0.5 * span).mean())}
        tnorm = np.zeros(len(e_h)); hf_r, hf_q, ac = [], [], []
        for e in np.unique(e_h):
            i = np.where(e_h == e)[0]; tnorm[i] = np.linspace(0, 1, len(i))
            if len(i) < 30:
                continue
            k = np.ones(9) / 9
            for x, acc in [(rt[i].reshape(len(i), -1), hf_r), (tq[i].numpy().reshape(len(i), -1) * 1000, hf_q)]:
                lp = np.stack([np.convolve(c, k, mode="same") for c in x.T], 1)[4:-4]; xx = x[4:-4]
                acc.append(((xx - lp) ** 2).sum() / max(((xx - xx.mean(0)) ** 2).sum(), 1e-12))
            x = rt[i].reshape(len(i), -1); x = x - x.mean(0)
            ac.append((x[1:] * x[:-1]).sum() / max((x * x).sum(), 1e-12))
        res["highfreq_share_residual"] = float(np.mean(hf_r)); res["highfreq_share_demo_motion"] = float(np.mean(hf_q))
        res["lag1_autocorr_residual"] = float(np.mean(ac))
        bins = np.minimum((tnorm * 10).astype(int), 9)
        res["res_tip_mm_by_time"] = [float(rmm[bins == b].mean()) for b in range(10)]
        clos = q.numpy().mean(1); qs = np.quantile(clos, [0.25, 0.5, 0.75]); cb = np.searchsorted(qs, clos)
        res["res_tip_mm_by_closure_quartile"] = [float(rmm[cb == b].mean()) for b in range(4)]
        # from z: held-out episodes (every 5th) -> R^2 of the normalized residual
        te = np.isin(e_h, np.unique(e_h)[::5]); Z = z.numpy(); R = r
        Zt = np.c_[Z[~te], np.ones((~te).sum())]
        W = np.linalg.solve(Zt.T @ Zt + 1e-3 * np.eye(Zt.shape[1]), Zt.T @ R[~te])
        res["r2_from_z_ridge"] = r2(np.c_[Z[te], np.ones(te.sum())] @ W, R[te])
        net = torch.nn.Sequential(torch.nn.Linear(32, 256), torch.nn.ReLU(), torch.nn.Linear(256, 256), torch.nn.ReLU(),
                                  torch.nn.Linear(256, n))
        opt = torch.optim.Adam(net.parameters(), 1e-3); Ztr, Rtr = torch.tensor(Z[~te]), torch.tensor(R[~te])
        for it in range(3000):
            b = torch.randint(0, len(Ztr), (1024,)); opt.zero_grad()
            ((net(Ztr[b]) - Rtr[b]) ** 2).mean().backward(); opt.step()
        with torch.no_grad():
            res["r2_from_z_mlp"] = r2(net(torch.tensor(Z[te])).numpy(), R[te])
            res["r2_from_z_mlp_train"] = r2(net(Ztr).numpy(), R[~te])
        out[h] = res
        for t in np.unique(task):                                             # mean tip residual curve, thumb/index/middle
            m = task == t; b20 = np.minimum((tnorm[m] * 20).astype(int), 19)
            curves[(t, h)] = np.stack([rt[m][b20 == b][:, :3].mean(0) for b in range(20)]).ravel()
        print(h, json.dumps({k: (round(v, 3) if isinstance(v, (int, float)) else [round(x, 2) for x in v]) for k, v in res.items()}), flush=True)
    sim = {}
    for t in dict.fromkeys(c[0] for c in cells):
        hs = [h for (tt, h) in curves if tt == t]
        C = np.stack([curves[(t, h)] for h in hs]); C = C / np.linalg.norm(C, axis=1, keepdims=True)
        S = C @ C.T; sim[t] = {"hands": hs, "mean_offdiag_cos": float((S.sum() - len(hs)) / (len(hs) ** 2 - len(hs))),
                               "matrix": np.round(S, 2).tolist()}
        print(t, "cross-hand residual-curve cosine (mean off-diag)", round(sim[t]["mean_offdiag_cos"], 3), flush=True)
    if a.json:
        json.dump({"per_hand": out, "cross_hand_similarity": sim}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
