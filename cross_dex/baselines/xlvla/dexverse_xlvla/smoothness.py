"""Step 6: temporal consistency of executed actions (bundle README "예상 차별점" 2).

Reads the per-episode logs written by dexverse_xlvla.policy (latent_dp and joint_dp modes): executed action per step and
whether a new policy chunk started there. Per hand, finger joints and wrist joints separately (rad or m per control step):
  step        mean |a_t - a_{t-1}|                       inside chunks
  jerk        mean |third difference of a|               windows fully inside one chunk
  boundary    mean |a_t - a_{t-1}| at chunk starts
  boundary_ratio = boundary / step                       > 1: chunk boundaries are rougher than motion inside chunks
  jerk2_inside  mean ||a_t - 2a_{t-1} + a_{t-2}||_2      windows fully inside one chunk (second-difference L2, as in
                                                         arXiv 2603.11642 "action jerk")
  jerk2_boundary  the same at windows that cross a chunk start
  jerk2_contrast = jerk2_boundary / jerk2_inside         boundary-interior jerk contrast (> 1: rougher at boundaries)
Parts: fingers / wrist = executed joint actions (both policies); latent_z = the latent DP's own output z (--z_policies),
same metrics in z units -> compare its ratio/contrast metrics with the decoded fingers (policy vs decoder roughness).
Writes a CSV (policy,hand,part,metric,value) and prints a table.

  python -m dexverse_xlvla.smoothness --log latent=<dir_with_hand_subdirs> --log joint=<dir> --out <csv>
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from dexverse_xlvla import hands as H
from dexverse_xlvla.replay import action_names


def metrics(a: np.ndarray, new: np.ndarray) -> dict:
    d1 = np.abs(np.diff(a, axis=0)).mean(1)                        # (T-1,) step size at t = 1..T-1
    start = new[1:]                                                # chunk starts at t
    inside = ~start
    chunk_id = np.cumsum(new)
    j = np.abs(np.diff(a, n=3, axis=0)).mean(1)                    # jerk at windows t..t+3
    same = chunk_id[3:] == chunk_id[:-3]
    j2 = np.linalg.norm(np.diff(a, n=2, axis=0), axis=1)          # (T-2,) window t-2..t, indexed by t = 2..T-1
    crosses = chunk_id[2:] != chunk_id[:-2]                        # a chunk start inside the window
    j2_in = float(j2[~crosses].mean()) if (~crosses).any() else np.nan
    j2_b = float(j2[crosses].mean()) if crosses.any() else np.nan
    return {"jerk2_inside": j2_in, "jerk2_boundary": j2_b, "jerk2_contrast": j2_b / j2_in if j2_in > 1e-9 else np.nan,
            "step": float(d1[inside].mean()) if inside.any() else np.nan,
            "boundary": float(d1[start].mean()) if start.any() else np.nan,
            "jerk": float(j[same].mean()) if same.any() else np.nan}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Dexverse-GraspKettle-v0")
    ap.add_argument("--log", action="append", required=True, help="<policy>=<dir>, dir/<hand>/ep*.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--z_policies", default="latent",
                    help="comma-separated policies whose logged policy output is [canonical wrist 9 | latent z]: the z part is "
                         "also measured (part 'latent_z', z units), to see whether roughness comes from the policy or the decoder")
    args = ap.parse_args()
    rows = []
    for spec in args.log:
        pol, root = spec.split("=", 1)
        for hdir in sorted(glob.glob(f"{root}/*/")):
            hand = os.path.basename(hdir.rstrip("/"))
            if hand not in H.HANDS:
                continue
            m = H.CanonicalHandModel(hand)
            idx = m.data_index(action_names(args.task, hand))
            acc = {}
            for f in sorted(glob.glob(f"{hdir}/ep*.npz")):
                d = np.load(f)
                a, new = d["action"], d["new_chunk"].astype(bool)
                if len(a) < 8:
                    continue
                for part, cols in (("fingers", idx), ("wrist", list(range(6)))):
                    for k, v in metrics(a[:, cols], new).items():
                        acc.setdefault((part, k), []).append(v)
                if pol in args.z_policies.split(",") and "policy_out" in d and d["policy_out"].shape[1] > 9:
                    for k, v in metrics(d["policy_out"][:, 9:], new).items():   # latent z as output by the policy
                        acc.setdefault(("latent_z", k), []).append(v)
            for (part, k), v in acc.items():
                rows.append((pol, hand, part, k, float(np.nanmean(v))))
            for part in ("fingers", "wrist", "latent_z"):
                s, b = acc.get((part, "step")), acc.get((part, "boundary"))
                if s and b:
                    rows.append((pol, hand, part, "boundary_ratio", float(np.nanmean(b) / np.nanmean(s))))
    with open(args.out, "w") as f:
        f.write("policy,hand,part,metric,value\n")
        for r in rows:
            f.write(",".join(map(str, r[:4])) + f",{r[4]:.6g}\n")
    for r in rows:
        print(f"{r[0]:8s} {r[1]:8s} {r[2]:8s} {r[3]:15s} {r[4]:.5f}")
    json.dump({"rows": len(rows)}, open(args.out + ".json", "w"))


if __name__ == "__main__":
    main()
