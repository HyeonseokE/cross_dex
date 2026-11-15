"""Keyframe store of the full Dexonomy set for on-the-fly chunk generation in training (train.py --kf).

Input: retarget_dexcanvas --keyframe_shards outputs (per grasp 3 keyframes pregrasp / grasp / squeeze per hand, with the
finger-finger penetration measured after the collision correction, <hand>_pen_mm). Filter (user rule 2026-11-12): drop a
grasp if any hand penetrates deeper than --drop_pen_mm (8) at any keyframe; then keep only grasps with every keyframe of
every hand within --pen_mm (2) (episode-level filter: Dexonomy is abundant). Trajectories are made in training: each hand's
joints interpolated by min-jerk between its keyframes (hold 0.1 s, pregrasp->grasp over a closing time drawn from the
human closing-time distribution of the type, grasp->squeeze 0.2 s, hold 0.3 s; the same timing for all hands of a grasp).

Output <out>/kf.npz: q_<hand> [N, 3, n] float32, feix [N] int16, split [N] (0 train / 1 val, by grasp id hash);
<out>/meta.json: hands, finger names, closing-time quantiles per type (from timing.json), counts per type.

  PYTHONPATH=/workspace/codes/cross_dex python -m cross_dex.dataset.chunk_paired.build_keyframes --rt <keyframe retarget dir> --out <dir>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import zlib
from collections import Counter

import numpy as np

from cross_dex.dataset.chunk_paired import DATA_ROOT, save_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rt", default=os.path.join(DATA_ROOT, "dexonomy", "retarget", "keyframes_all"),
                    help="comma list of keyframe retarget dirs with different hands (same shards)")
    ap.add_argument("--fingers", default=os.path.join(DATA_ROOT, "chunks", "feix400v2_H16", "meta.json"))
    ap.add_argument("--timing", default=os.path.join(DATA_ROOT, "dexcanvas", "timing.json"))
    ap.add_argument("--out", default=os.path.join(DATA_ROOT, "chunks", "dexonomy_kf_all"))
    ap.add_argument("--hands", default="shadow,inspire,sharpa,wuji")
    ap.add_argument("--drop_pen_mm", type=float, default=8.0)
    ap.add_argument("--pen_mm", type=float, default=2.0)
    ap.add_argument("--val_frac", type=float, default=0.02)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    hands = a.hands.split(",")
    Q = {h: [] for h in hands}
    F, S = [], []
    cnt = Counter()
    rts = a.rt.split(",")                 # several runs of the same shards with different hands (joined by shard name)
    for f in sorted(glob.glob(os.path.join(rts[0], "*.npz"))):
        if f.endswith(".tmp.npz"):
            continue
        others = [os.path.join(r, os.path.basename(f)) for r in rts[1:]]
        if not all(os.path.exists(o) for o in others):
            print("skip (missing in a hand run):", os.path.basename(f), flush=True)
            continue
        zz = [np.load(f)] + [np.load(o) for o in others]
        z = {}
        for zi in zz:
            z.update({kk: zi[kk] for kk in zi.files})
        assert all((zi["trajectory_id"] == zz[0]["trajectory_id"]).all() for zi in zz[1:]), f"id mismatch in {f}"
        n = len(z["trajectory_id"])
        pen = np.stack([z[f"{h}_pen_mm"].reshape(n, 3) for h in hands])           # [hands, N, 3]
        pen = np.nan_to_num(pen, nan=0.0)
        worst = pen.max(axis=(0, 2))
        deep = worst > a.drop_pen_mm
        keep = (~deep) & (worst <= a.pen_mm)
        fx = z["feix"]
        for g in np.unique(fx):
            m = fx == g
            cnt[(int(g), "all")] += int(m.sum())
            cnt[(int(g), "deep")] += int((m & deep).sum())
            cnt[(int(g), "kept")] += int((m & keep).sum())
        for h in hands:
            Q[h].append(z[f"{h}_q"].reshape(n, 3, -1)[keep])
        F.append(fx[keep].astype(np.int16))
        S.append(np.array([zlib.crc32(t.encode()) % 10000 < a.val_frac * 10000 for t in z["trajectory_id"][keep]], np.int8))
        print(os.path.basename(f), n, "kept", int(keep.sum()), "deep", int(deep.sum()), flush=True)
    out = {f"q_{h}": np.concatenate(Q[h]).astype(np.float32) for h in hands}
    out["feix"], out["split"] = np.concatenate(F), np.concatenate(S)
    np.savez(os.path.join(a.out, "kf.npz"), **out)
    timing = json.load(open(a.timing))["close"]
    types = sorted({g for g, _ in cnt})
    fj = {}
    for fpath in a.fingers.split(","):
        fj.update(json.load(open(fpath))["finger_joint_names"] if "finger_joint_names" in json.load(open(fpath))
                  else {h: v["finger_joint_names"] for h, v in json.load(open(fpath))["hands"].items()})
    meta = {"hands": hands, "finger_joint_names": {h: fj[h] for h in hands},
            "close_q": {str(g): timing.get(str(g), timing["all"])["p10_p50_p90"] for g in types},
            "per_feix": {str(g): {k: cnt[(g, k)] for k in ("all", "deep", "kept")} for g in types},
            "n": int(len(out["feix"])), "n_val": int(out["split"].sum()), "args": vars(a),
            "timing_rule": "hold 0.1 s, pregrasp->grasp min-jerk over close/0.507 s (close ~ quantiles), grasp->squeeze 0.2 s, hold 0.3 s"}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    save_manifest(a.out, a, n=meta["n"])
    print(json.dumps({"n": meta["n"], "per_feix": meta["per_feix"]}))


if __name__ == "__main__":
    main()
