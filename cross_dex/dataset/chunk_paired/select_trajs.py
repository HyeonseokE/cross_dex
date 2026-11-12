"""Pick up to N trajectories per Feix grasp type from a keypoint source (input selection of the retargeting run).

DexCanvas labels are mapped to Feix numbers (FEIX_MAPPING.md; '01' and '001-Palmar-Pinch' are the same type, task
label 005 and unmapped labels are dropped); HOGraspNet carries ``feix`` per trajectory. Selection is uniform at random
(seed) within a type; types with fewer trajectories are taken whole.

Output json: {"files": {<shard file>: [trajectory index, ...]}, "per_feix": {feix: n}, "args": ...}; consumed by
``retarget_dexcanvas --select <json> --part i/P`` (P processes split the list).

  PYTHONPATH=/workspace/codes/cross_dex /workspace/envs/dexverse/lerobot/bin/python -m cross_dex.dataset.chunk_paired.select_trajs \
      --data <keypoints dir> --per_feix 400 --out <json>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from cross_dex.dataset.chunk_paired import save_manifest

# DexCanvas label number -> Feix number (FEIX_MAPPING.csv); "Prismatic n-Finger" counts the thumb
DEXCANVAS_FEIX = {1: 9, 2: 8, 3: 7, 4: 6, 7: 20, 8: 22, 9: 1, 10: 18, 11: 28, 12: 19, 13: 31, 14: 4, 15: 17, 16: 23,
                  18: 16, 19: 25}


def feix_of(z, i: int) -> int | None:
    if "feix" in z.files:
        return int(z["feix"][i])
    lab = str(z["action_label"][i])
    return DEXCANVAS_FEIX.get(int(lab.split("-")[0])) if lab[:1].isdigit() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="keypoint npz dir (common format)")
    ap.add_argument("--per_feix", type=int, default=400)
    ap.add_argument("--min_T_s", type=float, default=0.3, help="skip trajectories shorter than this (s)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cand: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(a.data, "*.npz"))):
        with np.load(f) as z:
            off, t = z["offsets"], z["t"]
            for i in range(len(off) - 1):
                g = feix_of(z, i)
                if g is None or t[off[i + 1] - 1] - t[off[i]] < a.min_T_s:
                    continue
                cand[g].append((os.path.basename(f), i))
    rng = np.random.default_rng(a.seed)
    files: dict[str, list[int]] = defaultdict(list)
    per = {}
    for g in sorted(cand):
        c = cand[g]
        pick = rng.choice(len(c), size=min(a.per_feix, len(c)), replace=False)
        for k in sorted(pick):
            files[c[k][0]].append(c[k][1])
        per[g] = {"selected": int(len(pick)), "available": len(c)}
    res = {"data": a.data, "files": {k: sorted(v) for k, v in sorted(files.items())}, "per_feix": per, "args": vars(a)}
    json.dump(res, open(a.out, "w"), indent=1)
    save_manifest(a.out, a, per_feix=per)
    print(json.dumps(per))


if __name__ == "__main__":
    main()
