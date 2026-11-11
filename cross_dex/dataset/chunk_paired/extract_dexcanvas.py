"""Extract the right-hand keypoint trajectories of DexCanvas shards to npz (no pyarrow in the DexVerse main env).

Per shard -> <out>/train-XXXXX.npz, flat arrays (no pickles: the lerobot and main envs have different numpy majors):
trajectory_id [N], action_label [N], offsets [N+1], t [sum T], joints [sum T,21,3] (world m, DexCanvas/SNAP order),
wrist_rotvec [sum T,3] (MANO global/root rotation in world, axis-angle rad), mano_shape [N,10]. Trajectory i = rows offsets[i]:offsets[i+1].

  /workspace/envs/dexverse/lerobot/bin/python -m cross_dex.dataset.chunk_paired.extract_dexcanvas --out <dir> --shards 0:894
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
from cross_dex.dataset.chunk_paired import DATA_ROOT, save_manifest  # noqa: E402
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/workspace/datasets/dexverse/human_motion/dexcanvas/data")
    ap.add_argument("--out", default=os.path.join(DATA_ROOT, "dexcanvas", "keypoints"))
    ap.add_argument("--shards", default="0:")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.data, "train-*.parquet")))
    s, e = (a.shards.split(":") + [""])[:2] if ":" in a.shards else (a.shards, str(int(a.shards) + 1))
    for f in files[int(s or 0): int(e) if e else len(files)]:
        dst = os.path.join(a.out, os.path.basename(f).replace(".parquet", ".npz"))
        if os.path.exists(dst):
            with np.load(dst) as z:
                if "wrist_rotvec" in z.files:   # older files lack the wrist rotation -> re-extract
                    continue
        rows = pq.read_table(f, columns=["trajectory_id", "action_label", "timestamps_s", "hands"]).to_pylist()
        tid, lab, ts, js, rv, shp = [], [], [], [], [], []
        for r in rows:
            h = [x for x in r["hands"] if x["side"] == "right"]
            if not h:
                continue
            h = h[0]
            t = np.asarray(r["timestamps_s"], np.float64)
            tid.append(r["trajectory_id"])
            lab.append(r["action_label"] or "")
            ts.append(t)
            js.append(np.asarray(h["joint_positions_world_m"], np.float32).reshape(len(t), 21, 3))
            rv.append(np.asarray(h["global_rotation_world_rotvec_rad"], np.float32).reshape(len(t), 3))
            shp.append(np.asarray(h["mano_shape"], np.float32))
        off = np.concatenate([[0], np.cumsum([len(t) for t in ts])]).astype(np.int64)
        np.savez(dst + ".tmp.npz", trajectory_id=np.array(tid), action_label=np.array(lab), offsets=off,
                 t=np.concatenate(ts), joints=np.concatenate(js), wrist_rotvec=np.concatenate(rv), mano_shape=np.stack(shp))
        os.replace(dst + ".tmp.npz", dst)
        print(os.path.basename(dst), len(tid), flush=True)
    save_manifest(a.out, a, n_npz=len(glob.glob(os.path.join(a.out, "train-*.npz"))))


if __name__ == "__main__":
    main()
