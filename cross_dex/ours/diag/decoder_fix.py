"""[1-b] per-hand decoder correction f_h(z), task-agnostic (RSS2027 problem 2; dexlatent venv).

The shared space is kept fixed (E_h, D_h of a trained latent). For each hand h a small MLP learns the residual the decoder
leaves on task-agnostic poses: r = q - D_h(E_h(q)), q from the uniform pair set (h's own uniform draws + every other
hand's draws retargeted onto h, cross_dex.ours.uniform_pairs). Corrected decoder: D'_h(z) = D_h(z) + f_h(z).
No task data is used for training; task demos (teleop pickles' recorded finger actions) are only evaluated.

  train  python -m cross_dex.ours.diag.decoder_fix train --pairs <npz> --latent_ckpt <pt> --out <fix.pt> [--json m.json]
         reports fingertip residual (mm, canonical hand frame) before/after on held-out poses and on task demos
  build  python -m cross_dex.ours.diag.decoder_fix build --fix <fix.pt> --task <T> --hand <h> --episodes 0-4 --out <dir>
         actions.npz for self replay (fingers D'_h(E_h(q*)), wrist copied) in the layout of dexverse_xlvla.replay
"""
import argparse
import glob
import json
import os
import pickle

import numpy as np
import torch

TASKS = "Dexverse-FunctionalHammerStrike-v0,Dexverse-GraspBleach-v0,Dexverse-GraspKettle-v0,Dexverse-GraspPan-v0"


def mlp(n):
    net = torch.nn.Sequential(torch.nn.Linear(32, 512), torch.nn.SiLU(), torch.nn.Linear(512, 512), torch.nn.SiLU(),
                              torch.nn.Linear(512, n))
    torch.nn.init.zeros_(net[-1].weight); torch.nn.init.zeros_(net[-1].bias)       # starts as the plain decoder
    return net


@torch.no_grad()
def encdec(tr, h, q, bs=20000):
    from dexverse_xlvla.eval_latent import dec, enc
    Z, S = [], []
    for i in range(0, len(q), bs):
        z = enc(tr, h, q[i:i + bs]); Z.append(z); S.append(dec(tr, h, z))
    return torch.cat(Z), torch.cat(S)


@torch.no_grad()
def tip_mm(M, a, b, bs=20000):
    return torch.cat([(M.forward(a[i:i + bs].clamp(-1, 1)) - M.forward(b[i:i + bs].clamp(-1, 1))).norm(dim=-1).mean(-1)
                      for i in range(0, len(a), bs)]) * 1000


def own_pkl(task, hand):
    """The hand's teleop pickle; with extra top-up pickles next to it (e.g. GraspBleach leap), the largest (50 ep) one."""
    root = os.environ.get("XLVLA_PKL_ROOT", "/workspace/datasets/dexverse/teleop_pkl_bs1")
    hits = glob.glob(f"{root}/*/{task}/{task}_*_floating_{hand}_right.pkl")
    return max(hits, key=os.path.getsize)


def demo_q(tr, task, hand, eps=None):
    """Recorded finger actions of a hand's own teleop pickle -> normalized joints in model order."""
    from dexverse_xlvla.replay import action_names
    M = tr.hand_models[hand]; ia = M.data_index(action_names(task, hand))
    d = pickle.load(open(own_pkl(task, hand), "rb"))
    ks = range(len(d["episodes"])) if eps is None else eps
    acts = {k: np.asarray(d["episodes"][k]["actions"], dtype=np.float64) for k in ks}
    q = {k: M.angles_to_normalized(torch.tensor(a[:, ia], dtype=torch.float32)).clamp(-1, 1) for k, a in acts.items()}
    return acts, q, ia


def train(a):
    from dexverse_xlvla.eval_latent import load
    tr = load(a.latent_ckpt)
    for m in tr.autoencoders.values():
        m.eval()
    P = np.load(a.pairs)
    hands = a.hands.split(",")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42); g = torch.Generator().manual_seed(42)
    out, nets = {}, {}
    for h in hands:
        M = tr.hand_models[h]; n = M.dof_count()
        q = torch.tensor(np.concatenate([P[f"{s}__{h}"] for s in hands]), dtype=torch.float32)
        assert q.shape[1] == n, (h, q.shape, n)
        z, s = encdec(tr, h, q); r = q - s
        perm = torch.randperm(len(q), generator=g); te, trn = perm[: len(q) // 20], perm[len(q) // 20:]
        net = mlp(n).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        Zt, Rt = z[trn].to(dev), r[trn].to(dev)
        for it in range(a.steps):
            b = torch.randint(0, len(Zt), (4096,), device=dev)
            opt.zero_grad(); loss = ((net(Zt[b]) - Rt[b]) ** 2).mean(); loss.backward(); opt.step(); sched.step()
            if it % 2000 == 0:
                print(h, it, f"{loss.item():.2e}", flush=True)
        net = net.cpu().eval(); nets[h] = net
        with torch.no_grad():
            s_fix = (s[te] + net(z[te])).clamp(-1, 1)
        res = {"pose_frames": len(q), "pose_heldout_tip_mm_before": float(tip_mm(M, q[te], s[te]).mean()),
               "pose_heldout_tip_mm_after": float(tip_mm(M, q[te], s_fix).mean()),
               "uniform_own_tip_mm_before": float(tip_mm(M, q[:60000], s[:60000]).mean())}
        for t in a.tasks.split(","):
            _, dq, _ = demo_q(tr, t, h)
            qq = torch.cat(list(dq.values())); zz, ss = encdec(tr, h, qq)
            with torch.no_grad():
                sf = (ss + net(zz)).clamp(-1, 1)
            res[f"demo_tip_mm_before/{t}"] = float(tip_mm(M, qq, ss).mean())
            res[f"demo_tip_mm_after/{t}"] = float(tip_mm(M, qq, sf).mean())
        out[h] = res
        print(h, json.dumps({k: round(v, 3) for k, v in res.items()}), flush=True)
    torch.save({"latent_ckpt": a.latent_ckpt, "pairs": a.pairs, "hands": hands, "steps": a.steps,
                "state_dicts": {h: nets[h].state_dict() for h in hands},
                "dof": {h: tr.hand_models[h].dof_count() for h in hands}}, a.out)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


def build(a):
    from dexverse_xlvla.eval_latent import load
    from dexverse_xlvla.replay import episode_list
    F = torch.load(a.fix, map_location="cpu", weights_only=False)
    tr = load(F["latent_ckpt"])
    for m in tr.autoencoders.values():
        m.eval()
    M = tr.hand_models[a.hand]; net = mlp(F["dof"][a.hand]); net.load_state_dict(F["state_dicts"][a.hand]); net.eval()
    eps = episode_list(a.episodes)
    acts, dq, ia = demo_q(tr, a.task, a.hand, eps)
    os.makedirs(a.out, exist_ok=True); out = {}
    for k in eps:
        z, s = encdec(tr, a.hand, dq[k])
        with torch.no_grad():
            qb = (s + (net(z) if a.mode == "fixed" else 0)).clamp(-1, 1)
        o = acts[k].copy()
        o[:, ia] = ((qb + 1) * 0.5 * (M._upper - M._lower) + M._lower).numpy()
        out[str(k)] = o.astype(np.float32)
    np.savez(f"{a.out}/actions.npz", **out)
    print(f"{a.out}/actions.npz", len(out), "episodes")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    t = sp.add_parser("train")
    t.add_argument("--pairs", default="/workspace/datasets/dexverse/uniform_pairs/dexpilot_right6_60k_wristfix.npz")
    t.add_argument("--latent_ckpt", default="/workspace/checkpoints/dexverse/xlvla/hf6/latent/checkpoint_epoch_25000.pt")
    t.add_argument("--hands", default="shadow,inspire,sharpa,wuji,allegro,leap")
    t.add_argument("--tasks", default=TASKS)
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--out", required=True)
    t.add_argument("--json", default="")
    b = sp.add_parser("build")
    b.add_argument("--fix", required=True)
    b.add_argument("--task", required=True)
    b.add_argument("--hand", required=True)
    b.add_argument("--episodes", default="0-4")
    b.add_argument("--mode", choices=["fixed", "shared"], default="fixed")
    b.add_argument("--out", required=True)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else build(a)


if __name__ == "__main__":
    main()
