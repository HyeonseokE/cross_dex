"""Hand-detail loss diagnosis, step 2 (dexlatent venv): split the latent policy's finger error into
  (1) representation ceiling   dec_h(z*) vs q*            (z* = enc_h(q*), the latent set's action)
  (P) latent policy error      dec_h(z_hat) vs dec_h(z*)
  total                        dec_h(z_hat) vs q*
and compare with the joint policy (q_hat vs q*) trained on the same frames. Wrist (shared 9-d) errors of both policies are
the control: a gap there is policy quality, not the finger space.
  (2) hand averaging: share of the latent residual e = z_hat - z* explained by the hand (per-task hand means, R^2), and the
      same for the joint policy's normalized residual on the dims all hands have
  (3) loss space: per hand, fingertip error per unit latent error (decoder gain kappa); spread over hands = how unequal an
      equal latent loss is across hands
Units: finger joints deg (radians from the URDF limits), fingertips mm (canonical FK), all chunk steps.

  python -m cross_dex.ours.diag.detail_analyze --npz <detail_infer out> --latent_ckpt <xlvla vae .pt> [--json <out>]
"""
import argparse, json

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--latent_ckpt", required=True)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    from dexverse_xlvla.eval_latent import load, dec, enc
    D = np.load(a.npz, allow_pickle=True); cells = json.loads(str(D["cells"])); cell = D["cell"]
    tr = load(a.latent_ckpt)
    for m in tr.autoencoders.values():
        m.eval()
    hands = sorted({h for _, h in cells}, key=[h for _, h in cells].index)
    res, eps_lat, eps_jnt = {}, [], []
    for h in hands:
        M = tr.hand_models[h]; n = M.dof_count(); span = (M._upper - M._lower).numpy()
        sel = np.array([cells[c][1] == h for c in cell])
        zs, zh = D["lat_gt"][sel][..., 9:41], D["lat_pred"][sel][..., 9:41]
        qs, qj = D["joint_gt"][sel][..., 9:9 + n], D["joint_pred"][sel][..., 9:9 + n]
        sh = zs.shape[:2]
        with torch.no_grad():
            f = lambda z: dec(tr, h, torch.tensor(z.reshape(-1, 32), dtype=torch.float32)).clamp(-1, 1)
            q_rep, q_lat = f(zs), f(zh)
        qs_t = torch.tensor(qs.reshape(-1, n)); qj_t = torch.tensor(qj.reshape(-1, n)).clamp(-1, 1)
        deg = lambda x, y: float(np.rad2deg(np.abs((x - y).numpy() * 0.5 * span)).mean())
        tip = lambda x: M._forward_internal(x.float())[0]
        mm = lambda x, y: float((tip(x) - tip(y)).norm(dim=-1).mean() * 1000)
        lat_err = float(np.linalg.norm(zh - zs, axis=-1).mean())
        r = {"frames": int(sel.sum()),
             "rep_deg": deg(q_rep, qs_t), "rep_mm": mm(q_rep, qs_t),
             "latpol_deg": deg(q_lat, q_rep), "latpol_mm": mm(q_lat, q_rep),
             "lat_total_deg": deg(q_lat, qs_t), "lat_total_mm": mm(q_lat, qs_t),
             "joint_deg": deg(qj_t, qs_t), "joint_mm": mm(qj_t, qs_t),
             "latent_err": lat_err, "kappa_mm_per_latent": mm(q_lat, q_rep) / max(lat_err, 1e-9)}
        for nm in ["lat", "joint"]:                                      # wrist control: position (units of the set) and rot6d
            p, g = D[f"{nm}_pred"][sel][..., :9], D[f"{nm}_gt"][sel][..., :9]
            r[f"{nm}_wrist_pos"] = float(np.linalg.norm(p[..., :3] - g[..., :3], axis=-1).mean())
            r[f"{nm}_wrist_rot6d"] = float(np.linalg.norm(p[..., 3:] - g[..., 3:], axis=-1).mean())
        # (1) vs (2): latent inversion = best z for q* found by optimizing z directly (decoder only, no encoder).
        # small inversion error + large round trip -> the encoder drops it (1); large inversion error -> decoder cannot
        # express it (2). Decoder resolution: from z_inv, re-fit q* + d (d: random, RMS 1 deg) and check the decoded change.
        g = torch.Generator().manual_seed(0)
        rows = torch.randperm(qs_t.shape[0], generator=g)[:1500]
        qt, z0 = qs_t[rows].float(), torch.tensor(zs.reshape(-1, 32)[rows.numpy()], dtype=torch.float32)

        def invert(target, z_init, steps=400):
            z = z_init.clone().requires_grad_(True); opt = torch.optim.Adam([z], lr=0.02)
            for _ in range(steps):
                opt.zero_grad(); loss = ((dec(tr, h, z) - target) ** 2).sum(); loss.backward(); opt.step()
            with torch.no_grad():
                return z.detach(), dec(tr, h, z).clamp(-1, 1)
        z_inv, q_inv = invert(qt, z0)
        r["inv_deg"], r["inv_mm"] = deg(q_inv, qt), mm(q_inv, qt)
        d = torch.randn(qt.shape, generator=g)
        d_rad = d * np.deg2rad(1.0) / d.pow(2).mean(-1, keepdim=True).sqrt()
        dn = d_rad / torch.tensor(0.5 * span, dtype=torch.float32)          # same perturbation in normalized units
        ok = ((qt + dn).abs() < 0.98).all(-1)                                  # keep perturbed targets inside the limits
        _, q_inv2 = invert(qt[ok] + dn[ok], z_inv[ok], steps=200)
        with torch.no_grad():
            q_enc = dec(tr, h, enc(tr, h, qt[ok])).clamp(-1, 1); q_enc2 = dec(tr, h, enc(tr, h, qt[ok] + dn[ok])).clamp(-1, 1)
        rel = lambda chg: float(((chg - dn[ok]) * torch.tensor(0.5 * span)).norm(dim=-1).div(d_rad[ok].norm(dim=-1)).mean() * 100)
        r["dec_res_pct"], r["dec_res_mm"] = rel(q_inv2 - q_inv[ok]), mm(q_inv2 - q_inv[ok] + qt[ok], qt[ok] + dn[ok])
        r["roundtrip_res_pct"], r["roundtrip_res_mm"] = rel(q_enc2 - q_enc), mm(q_enc2 - q_enc + qt[ok], qt[ok] + dn[ok])
        res[h] = r
        eps_lat.append((h, sel, (zh - zs)[:, 0]))
        eps_jnt.append((h, sel, (qj - qs)[:, 0, :6] if n >= 6 else None))
    # (2) hand averaging: R^2 of per-(task, hand) means vs per-task means of the step-0 residual
    task_of = np.array([cells[c][0] for c in cell])

    def r2(items):
        E = np.zeros((len(cell), items[0][2].shape[-1])); Hh = np.empty(len(cell), object)
        for h, sel, e in items:
            E[sel] = e; Hh[sel] = h
        tot = hand = 0.0
        for t in np.unique(task_of):
            m = task_of == t; Et = E[m] - E[m].mean(0)
            tot += (Et ** 2).sum()
            for h in np.unique(Hh[m]):
                mh = Hh[m] == h; hand += mh.sum() * ((E[m][mh].mean(0) - E[m].mean(0)) ** 2).sum()
        return hand / tot
    out = {"per_hand": res, "hand_R2_latent_residual": r2(eps_lat),
           "hand_R2_joint_residual_first6": r2([x for x in eps_jnt if x[2] is not None])}
    print(json.dumps(out, indent=1))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
