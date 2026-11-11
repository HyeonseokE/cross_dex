"""Data (b): kinematically aligned TRANSITION chunks between the same semantic poses on every hand.

Semantic poses per hand (same meaning on every 5-finger hand, deterministic so they pair across hands):
  0 open   : URDF zero pose clipped to the limits (CanonicalHandModel.open_qpos, the open hand on all DexVerse hands)
  k=1..4   : thumb-finger k pinch (index, middle, ring, little), a contact configuration near the open pose:
             minimise |tip_0 - tip_k|^2 + w |q - q_open|^2 from q_open (best contact over a few w) (XL-VLA/DexLatent pinch templates bring the thumb
             and one finger together; the DexVerse latent run found them from random starts, which do not pair).
A transition a -> b: hold a, min-jerk interpolation in normalised joint space over D seconds, hold b, at --fps.
Every H-step window (stride --stride) of that trajectory is a chunk; the same (a, b, D, window) on every hand is one
aligned chunk tuple. Joint space = the latent model's (normalised [-1, 1], URDF DoF order, dexverse_xlvla.hands).

  /workspace/envs/dexverse/dexlatent/bin/python -m cross_dex.dataset.chunk_paired.transition_chunks --out <dir>/transition_H16.npz
"""

from __future__ import annotations

import argparse
import itertools
import json
import os

import numpy as np
import torch

from dexverse_xlvla.hands import CanonicalHandModel

POSE_NAMES = ["open", "pinch_index", "pinch_middle", "pinch_ring", "pinch_little"]


def pinch_pose(m: CanonicalHandModel, k: int, iters: int, lr: float, w: float) -> tuple[torch.Tensor, float]:
    q0 = m.open_qpos()
    q = q0.clone().requires_grad_(True)
    opt = torch.optim.Adam([q], lr=lr)
    for _ in range(iters):
        tips, _ = m.forward_with_wrist_pose(q.unsqueeze(0))
        tips = tips[0]
        loss = ((tips[0] - tips[k]) ** 2).sum() + w * ((q - q0) ** 2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            q.clamp_(-1, 1)
    with torch.no_grad():
        tips, _ = m.forward_with_wrist_pose(q.unsqueeze(0))
        dist = float((tips[0, 0] - tips[0, k]).norm())
    return q.detach(), dist


# fraction of a min-jerk transition spent between 10 % and 90 % of the way (numerically 0.507)
MINJERK_10_90 = 0.507


def min_jerk(s: np.ndarray) -> np.ndarray:
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hands", default="shadow,inspire,sharpa,wuji")
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--durations", default="0.25,0.5,0.75,1.0,1.5", help="transition durations (s)")
    ap.add_argument("--timing", default="", help="motion_timing.py json: durations = human closing time p10/p50/p90 "
                    "(10-90 %% aperture time) / 0.507 (= 10-90 %% fraction of a min-jerk profile); overrides --durations")
    ap.add_argument("--hold", type=float, default=0.27, help="hold before/after (s), >= one chunk")
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--w_open", type=lambda x: [float(v) for v in x.split(",")], default=[1e-3, 1e-4, 1e-5])
    a = ap.parse_args()
    if a.timing:
        q = json.load(open(a.timing))["close"]["all"]["p10_p50_p90"]
        a.durations = ",".join(f"{v / MINJERK_10_90:.2f}" for v in q)
        print("durations from human timing:", a.durations, flush=True)
    hands = a.hands.split(",")
    models = {h: CanonicalHandModel(h) for h in hands}
    poses, report = {}, {}
    for h, m in models.items():
        P = [m.open_qpos()]
        report[h] = {"dof_joints": list(m.dof_joints), "pinch_dist_mm": {}}
        for k in range(1, 5):
            # IK from the open pose; the open-pose weight trades contact vs. a local minimum (shadow ring: 9 mm at
            # 1e-4, 46 mm at 1e-5; index: the reverse) -> keep the closest contact over the weights
            q, d = min((pinch_pose(m, k, a.iters, a.lr, w) for w in a.w_open), key=lambda r: r[1])
            P.append(q)
            report[h]["pinch_dist_mm"][POSE_NAMES[k]] = round(d * 1000, 1)
        poses[h] = torch.stack(P).numpy()
        print(h, report[h]["pinch_dist_mm"], flush=True)
    hold = int(round(a.hold * a.fps))
    chunks = {h: [] for h in hands}
    meta = []
    for (pa, pb), D in itertools.product(itertools.permutations(range(len(POSE_NAMES)), 2),
                                         [float(x) for x in a.durations.split(",")]):
        n = max(int(round(D * a.fps)), 1)
        s = np.concatenate([np.zeros(hold), min_jerk(np.linspace(0, 1, n + 1)), np.ones(hold)])
        for st in range(0, len(s) - a.H + 1, a.stride):
            meta.append((pa, pb, D, st))
            for h in hands:
                traj = poses[h][pa][None] + s[:, None] * (poses[h][pb] - poses[h][pa])[None]
                chunks[h].append(traj[st: st + a.H])
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out = {f"{h}_q": np.asarray(c, np.float32) for h, c in chunks.items()}
    out.update({f"{h}_poses": poses[h].astype(np.float32) for h in hands})
    out["meta"] = np.asarray(meta, np.float32)   # pose_a, pose_b, duration_s, window_start
    np.savez_compressed(a.out, **out)
    report.update({"poses": POSE_NAMES, "n_chunks": len(meta), "H": a.H, "fps": a.fps, "args": vars(a)})
    with open(os.path.splitext(a.out)[0] + ".json", "w") as f:
        json.dump(report, f, indent=1)
    print("chunks", len(meta), {h: out[f"{h}_q"].shape for h in hands})


if __name__ == "__main__":
    main()
