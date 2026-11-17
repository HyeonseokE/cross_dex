"""Ours (2026-11-17, user: pairs must cover the whole joint range): hand-to-hand pairs from XL-VLA's pose generator.

For every source hand s, N poses are drawn uniformly in s's joint limits (XL-VLA Sec. 3.2: normalized joints uniform in
[-1, 1]) and retargeted to every other hand by DexPilot's objective (dex_retargeting DexPilotOptimizer, the optimizer
of the DexVerse retargeter that made the LAD pairs and the evaluation pairs), solved for a whole batch at once on the GPU:

  vectors   finger-finger tip_i - tip_j (i < j, thumb pairs first) + wrist -> tip_i      (generate_link_indices)
  reference source robot's vectors x length ratio (finger-finger: mean ratio r, wrist->tip: per-finger r_i = target /
            source wrist->tip length at the open pose) in place of the scaled human vectors
  pinch     thumb-finger pair < project_dist 3 cm -> reference eta1 1e-4 m along its direction, weight 200; finger-finger
            pair of two projected fingers within 3 cm -> eta2 3 cm, weight 400; other pairs weight 1; wrist->tip
            weight n_vec_ff + k                                                               (get_objective_function)
  cost      sum_v w_v * huber_0.03(|robot_v - ref_v|) / n_vec

Differences from DexPilot (recorded in the run README): source = robot hand instead of a human hand, so no hand YAML
scaling / finger length correction for human bones (replaced by the robot length ratios); no escape hysteresis and no
previous-frame regulariser (independent poses: a weak |q|^2 instead, and several random starts, best cost kept); no
collision projection; Adam on tanh-parametrised joints (in the limits) instead of NLopt SLSQP; fingertip / wrist =
dexverse_xlvla.hands canonical model (wrist = palm root). Fingers the target lacks (Allegro / LEAP little) drop out.

  cd /workspace/codes/dexverse/DexLatent && PYTHONPATH=/workspace/codes/cross_dex \
      python -m cross_dex.ours.uniform_pairs --out <file.npz> --per_hand 200000 [--hands ...]

Output npz: for source s, ``<s>__<t>`` = (N, dof_t) normalized joints of hand t (t == s: the drawn poses),
``<s>__<t>__res_mm`` = per-pose unweighted RMS DexPilot vector residual (mm); meta.json beside it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get("DEXLATENT_ROOT", "/workspace/codes/dexverse/DexLatent"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dexverse_xlvla import hands as H  # noqa: E402


# Wrist point = FK root + this offset (m, canonical axes: x root -> fingertips). The FK root is the wrist for every
# hand except Allegro, whose base_link sits on the finger-root line with the whole palm behind it (palm mesh x from
# -98 to 0 mm at joints 0; 2026-11-18 check, root_check.png) -> moved to the palm's lower end. Without it Allegro's
# wrist -> tip vectors are knuckle -> tip vectors and its thumb vector points backward (thumb residual 106 mm / 28 deg).
WRIST_OFFSET = {"allegro": (-0.095, 0.0, 0.0), "allegro_left": (-0.095, 0.0, 0.0)}
USE_WRIST_OFFSET = True                                          # False = the 2026-11-17 pairs (root as the wrist)


def keyvecs(model, q):
    tips, _, _ = model._forward_internal(q)                       # (B, k, 3), canonical frame
    wrist = model.canon[:3, 3].to(tips)                          # root origin in the canonical frame
    if USE_WRIST_OFFSET and model.hand_name in WRIST_OFFSET:
        wrist = wrist + torch.tensor(WRIST_OFFSET[model.hand_name]).to(tips)
    return tips, tips - wrist


def open_lengths(model):
    _, w2t = keyvecs(model, model.open_qpos().unsqueeze(0))
    return w2t[0].norm(dim=-1)                                   # (k,)


def dexpilot_vectors(tips, w2t, k):
    """DexPilot's vector set (dex_retargeting DexPilotOptimizer.generate_link_indices): finger-finger tip_i - tip_j for
    1 <= i < j <= k (thumb pairs first), then wrist -> tip_i."""
    ii, jj = zip(*[(i, j) for i in range(k - 1) for j in range(i + 1, k)])
    ii, jj = list(ii), list(jj)
    return torch.cat([tips[:, ii] - tips[:, jj], w2t[:, :k]], 1), len(ii)


def dexpilot_reference(src_vec, n_ff, k, r_i, r, project_dist, eta1, eta2):
    """Reference vectors and weights as DexPilotOptimizer.get_objective_function, with the source robot's vectors
    (x per-finger / mean length ratio) in place of the scaled human vectors. A thumb-finger pair closer than
    project_dist is projected to eta1 (contact), a finger-finger pair to eta2 if both fingers are projected with the
    thumb and it is within 3 cm. No escape hysteresis (independent poses, no previous frame)."""
    ff, wr = src_vec[:, :n_ff], src_vec[:, n_ff:]
    dist = ff.norm(dim=-1)
    n1 = k - 1                                                       # thumb pairs (s1)
    proj = torch.zeros_like(dist, dtype=torch.bool)
    proj[:, :n1] = dist[:, :n1] < project_dist
    o2, t2 = zip(*[(j, i) for i in range(k - 2) for j in range(i + 1, k - 1)]) if k > 2 else ((), ())
    if o2:
        proj[:, n1:] = proj[:, list(o2)] & proj[:, list(t2)] & (dist[:, n1:] <= 0.03)
    pd = torch.tensor([eta1] * n1 + [eta2] * (n_ff - n1), device=ff.device)
    ref_ff = torch.where(proj[..., None], ff / (dist[..., None] + 1e-6) * pd[None, :, None], ff * r)
    ref = torch.cat([ref_ff, wr * r_i[None, :, None]], 1)
    w = torch.where(proj, torch.tensor([200.0] * n1 + [400.0] * (n_ff - n1), device=ff.device), torch.ones_like(dist))
    w = torch.cat([w, torch.full((ff.shape[0], k), float(n_ff + k), device=ff.device)], 1)
    return ref, w, proj


def retarget(src, tgt, qs, iters, starts, lr, reg, dp):
    ts, ws = keyvecs(src, qs)
    k = min(ts.shape[1], tgt.tip_count())
    r_i = (open_lengths(tgt)[:k] / open_lengths(src)[:k]).to(qs)
    r = r_i.mean()
    sv, n_ff = dexpilot_vectors(ts, ws, k)
    ref, w, proj = dexpilot_reference(sv, n_ff, k, r_i, r, dp["project_dist"], dp["eta1"], dp["eta2"])
    huber = torch.nn.SmoothL1Loss(beta=dp["huber_delta"], reduction="none")
    nv = ref.shape[1]

    def cost(q):
        tt, wt = keyvecs(tgt, q)
        tv, _ = dexpilot_vectors(tt, wt, k)
        d = (tv - ref).norm(dim=-1)
        return (huber(d, torch.zeros_like(d)) * w).sum(-1) / nv, d

    B, dd = qs.shape[0], tgt.dof_count()
    best_q, best, best_d = None, None, None
    for s in range(starts):
        u0 = torch.zeros(B, dd, device=qs.device) if s == 0 else torch.empty(B, dd, device=qs.device).uniform_(-1.5, 1.5)
        u = u0.requires_grad_(True)
        opt = torch.optim.Adam([u], lr=lr)
        for _ in range(iters):
            opt.zero_grad()
            q = torch.tanh(u)
            c, _ = cost(q)
            (c.sum() + reg * q.pow(2).sum()).backward()
            opt.step()
        with torch.no_grad():
            q = torch.tanh(u)
            c, d = cost(q)
            if best is None:
                best_q, best, best_d = q.clone(), c, d
            else:
                m = c < best
                best_q[m], best[m], best_d[m] = q[m], c[m], d[m]
    rms = best_d.pow(2).mean(-1).sqrt() * 1000                        # mm, unweighted RMS over DexPilot vectors
    return best_q, rms, proj.any(-1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hands", default=",".join(H.HAND_NAMES))
    ap.add_argument("--per_hand", type=int, default=200000, help="uniform poses drawn per source hand")
    ap.add_argument("--chunk", type=int, default=32768)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--starts", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--reg", type=float, default=1e-6)
    ap.add_argument("--seed", type=int, default=42)
    # DexPilotOptimizer defaults (dex_retargeting)
    ap.add_argument("--huber_delta", type=float, default=0.03)
    ap.add_argument("--project_dist", type=float, default=0.03)
    ap.add_argument("--eta1", type=float, default=1e-4)
    ap.add_argument("--eta2", type=float, default=3e-2)
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names = a.hands.split(",")
    models = {h: H.CanonicalHandModel(h) for h in names}
    dp = {k: getattr(a, k) for k in ("huber_delta", "project_dist", "eta1", "eta2")}
    out, stats, t0 = {}, {}, time.time()
    for s in names:
        qs_all = torch.empty(a.per_hand, models[s].dof_count()).uniform_(-1, 1)
        res = {t: [] for t in names if t != s}
        qt = {t: [] for t in names if t != s}
        pj = {t: [] for t in names if t != s}
        for c0 in range(0, a.per_hand, a.chunk):
            qs = qs_all[c0:c0 + a.chunk].to(dev)
            for t in res:
                q, e, p = retarget(models[s], models[t], qs, a.iters, a.starts, a.lr, a.reg, dp)
                qt[t].append(q.cpu()); res[t].append(e.cpu()); pj[t].append(p.cpu())
            print(f"[{s}] {min(c0 + a.chunk, a.per_hand)}/{a.per_hand} {time.time() - t0:.0f}s", flush=True)
        out[f"{s}__{s}"] = qs_all.numpy().astype(np.float32)
        for t in res:
            out[f"{s}__{t}"] = torch.cat(qt[t]).numpy().astype(np.float32)
            r = torch.cat(res[t])
            out[f"{s}__{t}__res_mm"] = r.numpy().astype(np.float32)
            stats[f"{s}->{t}"] = {"median_mm": round(float(r.median()), 2), "p90_mm": round(float(r.quantile(0.9)), 2),
                                  "pinch_projected_pct": round(100 * float(torch.cat(pj[t]).float().mean()), 1)}
        print(f"[{s}] residual " + json.dumps({k: v for k, v in stats.items() if k.startswith(s + '->')}), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.savez(a.out, **out)
    json.dump({"args": vars(a), "hands": names, "residual": stats, "sec": round(time.time() - t0)},
              open(os.path.splitext(a.out)[0] + ".meta.json", "w"), indent=1)
    print(f"[done] {a.out} {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
