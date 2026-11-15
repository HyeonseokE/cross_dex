"""Step 1 check: DexVerse hand models for DexLatent.

For every hand: DoF / fingertip count, canonical frame, dataset joint mapping, how much of the teleop data lies inside
the URDF limits (a sim/URDF mismatch shows up as data outside the limits), and whether DexLatent's pinch templates
(IK to the pinch point) converge. Writes a JSON report.

  python -m dexverse_xlvla.check_hands --task Dexverse-GraspKettle-v0 --out <dir>/hands_check.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import torch

import HandLatent.kinematics as K
from dexverse_xlvla.hands import HAND_NAMES, CanonicalHandModel

LEROBOT_ROOT = os.environ.get("XLVLA_TELEOP_ROOT", "/workspace/datasets/dexverse/teleop_lerobot")   # read only


def dataset_dir(task: str, hand: str) -> str:
    hits = sorted(glob.glob(f"{LEROBOT_ROOT}/{task}/{hand}_right/*_right/meta/info.json"))
    if not hits:
        raise FileNotFoundError(f"no LeRobot set for {task} {hand}")
    return os.path.dirname(os.path.dirname(hits[0]))


def load_actions(ds: str, key: str = "action") -> tuple[np.ndarray, list[str]]:
    info = json.load(open(f"{ds}/meta/info.json"))
    names = info["features"][key]["names"]
    names = list(names.values())[0] if isinstance(names, dict) else names
    frames = [pd.read_parquet(f, columns=[key]) for f in sorted(glob.glob(f"{ds}/data/*/*.parquet"))]
    arr = np.stack(pd.concat(frames)[key].to_numpy()).astype(np.float64)
    return arr, names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Dexverse-GraspKettle-v0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    report = {}
    for hand in HAND_NAMES:
        m = CanonicalHandModel(hand)
        r = {"dof": m.dof_count(), "dof_joints": m.joint_name_order(), "tips": m.tip_count(),
             "root_link": m.root_link}
        # canonical frame: open-hand fingertips (m), thumb should have +y, fingers +x
        tips_open = m.forward(m.open_qpos())
        r["open_tips_canonical_m"] = np.round(tips_open.numpy(), 4).tolist()
        # dataset mapping and range check (actions = commanded joint targets)
        ds = dataset_dir(args.task, hand)
        act, names = load_actions(ds)
        idx = m.data_index(names)
        q = act[:, idx]
        lo, hi = m._lower.numpy(), m._upper.numpy()
        tol = 1e-3
        out = (q < lo - tol) | (q > hi + tol)
        r["dataset"] = ds
        r["frames"] = int(len(q))
        r["data_cols_for_dof"] = idx
        r["frac_outside_limits"] = float(out.mean())
        r["joints_outside"] = {m.dof_joints[k]: {"frac": round(float(out[:, k].mean()), 4),
                                                  "data_min": round(float(q[:, k].min()), 4),
                                                  "data_max": round(float(q[:, k].max()), 4),
                                                  "lower": round(float(lo[k]), 4), "upper": round(float(hi[k]), 4)}
                               for k in range(len(lo)) if out[:, k].any()}
        r["data_range_used"] = float(np.mean((q.max(0) - q.min(0)) / np.maximum(hi - lo, 1e-8)))
        # pinch templates: thumb + finger k to the canonical origin (pinch point), DexLatent IK
        res = {}
        neutral = tips_open.clone()
        for k in range(1, m.tip_count()):
            target = neutral.clone()
            target[0] = 0.0
            target[k] = 0.0
            hist = K.solve_inverse_kinematics(m, target.unsqueeze(0), iterations=100, learning_rate=0.05)
            tips = m.forward(hist[-1])[0]
            res[f"0-{k}"] = {"thumb_err_mm": round(float(tips[0].norm()) * 1000, 1),
                             "finger_err_mm": round(float(tips[k].norm()) * 1000, 1),
                             "pinch_dist_mm": round(float((tips[0] - tips[k]).norm()) * 1000, 1)}
        r["pinch_ik"] = res
        report[hand] = r
        print(f"{hand:8s} dof={r['dof']:2d} tips={r['tips']} frames={r['frames']} "
              f"outside={r['frac_outside_limits']:.4f} range_used={r['data_range_used']:.2f} "
              f"pinch_dist_mm={[v['pinch_dist_mm'] for v in res.values()]}")
        for j, v in r["joints_outside"].items():
            print(f"    outside {j}: {v}")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(report, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    torch.set_grad_enabled(True)
    main()
