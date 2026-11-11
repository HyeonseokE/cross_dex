"""Human finger proportions of a keypoint source (pose-invariant), for the per-finger length correction of the retargeting
wrapper (retarget_dexcanvas --finger_lengths).

Per finger: meta = |wrist -> finger root| (root = thumb CMC / long-finger MCP), phal = sum of bone lengths root -> tip.
Both are rigid, so they are measured on every frame; per trajectory the median, over trajectories the median.

  PYTHONPATH=/workspace/codes/cross_dex /workspace/envs/dexverse/lerobot/bin/python -m cross_dex.dataset.chunk_paired.finger_lengths
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from cross_dex.dataset.chunk_paired import DATA_ROOT, save_manifest

# DexCanvas / SNAP order: palm(wrist) 0, index mcp/pip/dip 1-3, middle 4-6, ring 7-9, pinky 10-12, thumb cmc/mcp/ip 13-15,
# tips thumb/index/middle/ring/pinky 16-20
CHAINS = {"thumb": [13, 14, 15, 16], "index": [1, 2, 3, 17], "middle": [4, 5, 6, 18], "ring": [7, 8, 9, 19],
          "pinky": [10, 11, 12, 20]}


def lengths(j: np.ndarray) -> dict[str, dict[str, float]]:
    out = {}
    for f, c in CHAINS.items():
        meta = np.linalg.norm(j[:, c[0]] - j[:, 0], axis=-1)
        phal = sum(np.linalg.norm(j[:, c[i + 1]] - j[:, c[i]], axis=-1) for i in range(len(c) - 1))
        out[f] = {"meta": float(np.median(meta)), "phal": float(np.median(phal))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(DATA_ROOT, "dexcanvas", "keypoints"))
    ap.add_argument("--shards", default="0:894:30", help="start:stop:step")
    ap.add_argument("--out", default=os.path.join(DATA_ROOT, "dexcanvas", "finger_lengths.json"))
    a = ap.parse_args()
    s0, s1, st = (int(x) for x in a.shards.split(":"))
    per = {f: {"meta": [], "phal": []} for f in CHAINS}
    for fn in sorted(glob.glob(os.path.join(a.data, "train-*.npz")))[s0:s1:st]:
        with np.load(fn) as z:
            off, J = z["offsets"], z["joints"]
            for i in range(len(off) - 1):
                for f, v in lengths(J[off[i]: off[i + 1]].astype(np.float64)).items():
                    per[f]["meta"].append(v["meta"])
                    per[f]["phal"].append(v["phal"])
    res = {f: {k: round(float(np.median(v[k])), 4) for k in v} | {"n": len(v["meta"])} for f, v in per.items()}
    for f in res:
        res[f]["total"] = round(res[f]["meta"] + res[f]["phal"], 4)
    json.dump({"fingers": res, "args": vars(a)}, open(a.out, "w"), indent=1)
    save_manifest(a.out, a)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
