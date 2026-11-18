"""Is the latent inversion of task demos a usable target? (follow-up of detail_analyze, dexlatent venv)

For each hand s, demo finger actions q* (rows of a detail_infer npz) are encoded (z_enc = E_s(q*)) and inverted
(z_inv = argmin_z |D_s(z) - q*|^2, Adam from z_enc). Two questions decide whether a "detail extractor" can put the
task detail inside the shared space (z_inv as the policy target) or must keep it outside (hand residual):
  1 distribution  is z_inv inside the latent region the encoder uses?
                  shift |z_inv - z_enc| vs the spread of z_enc (nearest-neighbour distance between demo latents, mean
                  distance to the hand mean), per-dim z-scores against the encoder latents of all hands (share of rows
                  with any |z| > 3), distance to the nearest encoder latent of any hand
  2 cross-hand    does z_inv still mean the same pose on the other hands? metric 3 of eval_space on these frames:
                  thumb-finger line of hand s (q*) vs of hand t decoded from z (direction deg, length mm), z_enc vs z_inv

  python -m cross_dex.ours.diag.inversion_check --npz <detail_infer out> --latent_ckpt <xlvla vae .pt> [--json <out>]
"""
import argparse, json

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--latent_ckpt", required=True)
    ap.add_argument("--rows", type=int, default=1500)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    from dexverse_xlvla.eval_latent import load, dec, enc, line_err
    D = np.load(a.npz, allow_pickle=True); cells = json.loads(str(D["cells"])); cell = D["cell"]
    tr = load(a.latent_ckpt)
    for m in tr.autoencoders.values():
        m.eval()
    hands = list(dict.fromkeys(h for _, h in cells))
    M = {h: tr.hand_models[h] for h in hands}
    g = torch.Generator().manual_seed(0)
    Q, ZE, ZI = {}, {}, {}
    for h in hands:
        n = M[h].dof_count(); sel = np.array([cells[c][1] == h for c in cell])
        q = torch.tensor(D["joint_gt"][sel][..., 9:9 + n].reshape(-1, n), dtype=torch.float32)
        q = q[torch.randperm(len(q), generator=g)[:a.rows]]
        with torch.no_grad():
            ze = enc(tr, h, q)
        z = ze.clone().requires_grad_(True); opt = torch.optim.Adam([z], lr=0.02)
        for _ in range(a.steps):
            opt.zero_grad(); ((dec(tr, h, z) - q) ** 2).sum().backward(); opt.step()
        Q[h], ZE[h], ZI[h] = q, ze, z.detach()
    allz = torch.cat(list(ZE.values())); mu, sd = allz.mean(0), allz.std(0)
    tips = lambda h, qn: M[h].forward(qn.clamp(-1, 1))
    res = {}
    with torch.no_grad():
        for s in hands:
            ze, zi = ZE[s], ZI[s]
            nn_d = torch.cdist(ze, ze).fill_diagonal_(float("inf")).min(1).values      # neighbour distance between demo latents
            to_any_e = torch.cdist(zi, allz).min(1).values                                # z_inv -> nearest encoder latent (any hand)
            r = {"shift": float((zi - ze).norm(dim=-1).mean()), "nn_dist_enc": float(nn_d.median()),
                 "spread_enc": float((ze - ze.mean(0)).norm(dim=-1).mean()),
                 "inv_to_nearest_enc": float(to_any_e.median()),
                 "rows_any_absz_gt3_enc_pct": float(100 * (((ze - mu) / sd).abs() > 3).any(-1).float().mean()),
                 "rows_any_absz_gt3_inv_pct": float(100 * (((zi - mu) / sd).abs() > 3).any(-1).float().mean()),
                 "self_tip_mm_enc": float((tips(s, dec(tr, s, ze)) - tips(s, Q[s])).norm(dim=-1).mean() * 1e3),
                 "self_tip_mm_inv": float((tips(s, dec(tr, s, zi)) - tips(s, Q[s])).norm(dim=-1).mean() * 1e3)}
            src = tips(s, Q[s])
            for nm, z in [("enc", ze), ("inv", zi)]:
                dd, ll = [], []
                for t in hands:
                    if t == s:
                        continue
                    k = min(M[s].tip_count(), M[t].tip_count())
                    d, l = line_err(src, tips(t, dec(tr, t, z)), [(0, i) for i in range(1, k)])
                    dd.append(d); ll.append(l * 1e3)
                r[f"cross_dir_deg_{nm}"], r[f"cross_len_mm_{nm}"] = float(np.mean(dd)), float(np.mean(ll))
            res[s] = r
    out = {"per_hand": res, "mean": {k: float(np.mean([res[h][k] for h in hands])) for k in res[hands[0]]}}
    print(json.dumps(out, indent=1))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
