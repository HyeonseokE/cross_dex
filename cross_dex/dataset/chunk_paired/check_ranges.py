"""Check (a): retargeted DexCanvas finger targets vs the finger actions of real DexVerse teleop demos, per hand/joint.

Both come from the same teleop retargeter, so a joint the human data never moves (or pins at a limit) while teleop
moves it, or values far outside the teleop range, point to a mapping problem (frame / joint order / scale).

  /workspace/envs/dexverse/lerobot/bin/python -m cross_dex.dataset.chunk_paired.check_ranges --rt <retarget out dir> [--out report.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import pyarrow.parquet as pq


def teleop_actions(root: str, hand: str, n_finger: int) -> np.ndarray:
    files = glob.glob(os.path.join(root, "*", f"{hand}_right", "*", "data", "chunk-*", "*.parquet"))
    acts = [np.stack(pq.read_table(f, columns=["action"]).column("action").to_numpy(zero_copy_only=False)) for f in files]
    a = np.concatenate(acts)
    return a[:, -n_finger:]                     # floating layout: 6 wrist joints, then the finger joints


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rt", required=True)
    ap.add_argument("--teleop", default="/workspace/datasets/dexverse/teleop_lerobot")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    meta = json.load(open(os.path.join(a.rt, "meta.json")))
    files = sorted(glob.glob(os.path.join(a.rt, "*.npz")))
    rep = {"n_traj": len(files)}
    for h, hm in meta["hands"].items():
        names = hm["finger_joint_names"]
        q = np.concatenate([np.load(f)[f"{h}_q"] for f in files])
        t = teleop_actions(a.teleop, h, len(names))
        tlo, thi = np.percentile(t, 1, 0), np.percentile(t, 99, 0)
        qlo, qhi = np.percentile(q, 1, 0), np.percentile(q, 99, 0)
        rows = []
        for i, n in enumerate(names):
            span = max(thi[i] - tlo[i], 1e-6)
            rows.append({"joint": n, "teleop_p1_p99": [round(float(tlo[i]), 3), round(float(thi[i]), 3)],
                         "human_p1_p99": [round(float(qlo[i]), 3), round(float(qhi[i]), 3)],
                         "coverage": round(float((min(qhi[i], thi[i]) - max(qlo[i], tlo[i])) / span), 2),
                         "outside_frac": round(float(((q[:, i] < tlo[i] - .05 * span) | (q[:, i] > thi[i] + .05 * span)).mean()), 3)})
        rep[h] = {"frames_human": int(len(q)), "frames_teleop": int(len(t)), "joints": rows}
        print(f"== {h} human {len(q)} frames, teleop {len(t)} frames")
        for r in rows:
            flag = " <-- low coverage" if r["coverage"] < 0.5 else (" <-- outside" if r["outside_frac"] > 0.1 else "")
            print(f"  {r['joint']:28s} teleop {r['teleop_p1_p99']} human {r['human_p1_p99']} cov {r['coverage']} out {r['outside_frac']}{flag}")
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
