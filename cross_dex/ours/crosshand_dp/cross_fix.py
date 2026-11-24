"""Cross-hand residual module for an unseen hand (RSS2027, 2026-11-23; dexlatent venv).

Target hand t (default shadow) has NO task data. The other hands' GraspKettle demos q_s are retargeted onto t by DexPilot
(cross_dex.ours.uniform_pairs.retarget, the objective that made the spaces' training pairs): q~_t = DP(q_s -> t).
f_t(z) (same MLP as decoder_fix) is fitted on these pseudo demos in two ways:
  pseudo  z = E_t(q~_t),  r = q~_t - D_t(z)          (t's own encoder on the retargeted poses; hand clustering not involved)
  cross   z = E_s(q_s),   r = q~_t - D_t(E_s(q_s))   (source-hand z, cross residual; depends on how close E_s and E_t
                                                       put the same pose, i.e. on cross-hand clustering of the space)
Both are evaluated on t's REAL GraspKettle demos: fingertip error of D_t(E_t(q_t)) [+ f_t] vs q_t (mm, canonical frame).
The DexPilot retargets do not depend on the space and are cached (--cache).

  cd /workspace/codes/dexverse/DexLatent && PYTHONPATH=/workspace/codes/cross_dex python -m \
     cross_dex.ours.crosshand_dp.cross_fix --latent_ckpt <space model.pt> --cache <npz> --out <prefix> [--json m.json]
"""
import argparse
import json
import os

import numpy as np
import torch

from cross_dex.ours.diag.decoder_fix import demo_q, encdec, mlp, tip_mm

TASK = "Dexverse-GraspKettle-v0"
HANDS = ["shadow", "inspire", "sharpa", "wuji", "allegro", "leap"]
DP = {"huber_delta": 0.03, "project_dist": 0.03, "eta1": 1e-4, "eta2": 3e-2}   # uniform_pairs defaults


def fit(z, r, n, steps, dev):
    net = mlp(n).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    Z, R = z.to(dev), r.to(dev)
    for _ in range(steps):
        b = torch.randint(0, len(Z), (4096,), device=dev)
        opt.zero_grad(); loss = ((net(Z[b]) - R[b]) ** 2).mean(); loss.backward(); opt.step(); sched.step()
    return net.cpu().eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_ckpt", required=True)
    ap.add_argument("--target", default="shadow")
    ap.add_argument("--task", default=TASK)
    ap.add_argument("--cache", required=True, help="npz of DexPilot retargets (created if missing)")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--out", required=True, help="output prefix: <out>_pseudo.pt, <out>_cross.pt")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    from dexverse_xlvla.eval_latent import dec, enc, load
    tr = load(a.latent_ckpt)
    for m in tr.autoencoders.values():
        m.eval()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    t = a.target; srcs = [h for h in HANDS if h != t]; Mt = tr.hand_models[t]; n = Mt.dof_count()
    qs = {s: torch.cat(list(demo_q(tr, a.task, s)[1].values())) for s in srcs}
    if not os.path.exists(a.cache):
        from cross_dex.ours.uniform_pairs import retarget
        C = {}
        for s in srcs:
            outs = []
            for i in range(0, len(qs[s]), 20000):
                q, _, _ = retarget(tr.hand_models[s], Mt, qs[s][i:i + 20000].to(dev), 300, 2, 0.05, 1e-6, DP)
                outs.append(q.detach().cpu().clamp(-1, 1))
            C[s] = torch.cat(outs).numpy(); print("retarget", s, C[s].shape, flush=True)
        np.savez(a.cache, **C)
    C = np.load(a.cache)
    qt_ps = torch.cat([torch.tensor(C[s]) for s in srcs])
    with torch.no_grad():
        z_ps, s_ps = encdec(tr, t, qt_ps)
        z_cr = torch.cat([enc(tr, s, qs[s]) for s in srcs]); s_cr = dec(tr, t, z_cr)
    nets = {"pseudo": fit(z_ps, qt_ps - s_ps, n, a.steps, dev), "cross": fit(z_cr, qt_ps - s_cr, n, a.steps, dev)}
    q_real = torch.cat(list(demo_q(tr, a.task, t)[1].values()))
    z_r, s_r = encdec(tr, t, q_real)
    res = {"pseudo_frames": len(qt_ps), "real_frames": len(q_real), "real_tip_mm_before": float(tip_mm(Mt, q_real, s_r).mean())}
    with torch.no_grad():
        for k, net in nets.items():
            res[f"real_tip_mm_after_{k}"] = float(tip_mm(Mt, q_real, (s_r + net(z_r)).clamp(-1, 1)).mean())
        # how close the two encoders put the same pose: |E_s(q_s) - E_t(DP(q_s))| relative to the spread of t's z
        res["pair_z_dist_over_spread"] = float((z_cr - z_ps).norm(dim=-1).median() / z_r.std(0).norm())
        res["pseudo_fit_tip_mm"] = float(tip_mm(Mt, qt_ps, (s_ps + nets["pseudo"](z_ps)).clamp(-1, 1)).mean())
    for k, net in nets.items():
        torch.save({"latent_ckpt": a.latent_ckpt, "pairs": None, "task": a.task, "hands": [t], "kind": f"cross_hand_{k}",
                    "sources": srcs, "state_dicts": {t: net.state_dict()}, "dof": {t: n}}, f"{a.out}_{k}.pt")
    print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}), flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
