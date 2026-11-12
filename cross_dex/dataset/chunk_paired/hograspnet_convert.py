"""HOGraspNet -> common keypoint format (same as DexCanvas keypoints, input of the retargeting wrapper).

Input: hograspnet_index output (per subject S<k>.npz: OpenPose-order joints in m, 10 fps annotated frames).
Per trial, every run of consecutive annotated frames (>= --min_run, 10 fps) becomes one trajectory:
  joints   OpenPose order -> handframe.DC order
  wrist    MANO-root-equivalent rotation from keypoints (handframe.root_rotvec_from_keypoints; HOGraspNet MANO has no
           global rotation) -> same wrist convention as DexCanvas
  t        frame / 10 s (the retargeting wrapper resamples to the policy rate)
  label    feix = Feix GRASP number of the trial (annotated per frame, constant in a trial)
Output per subject -> <out>/S<k>.npz with the DexCanvas keypoint keys (trajectory_id, action_label, offsets, t, joints,
wrist_rotvec, mano_shape) plus feix, subject, object_id, trial_key.

  PYTHONPATH=/workspace/codes/cross_dex /workspace/envs/dexverse/lerobot/bin/python -m cross_dex.dataset.chunk_paired.hograspnet_convert
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from cross_dex.dataset.chunk_paired import DATA_ROOT, save_manifest
from cross_dex.dataset.chunk_paired.handframe import OPENPOSE_TO_DC, root_rotvec_from_keypoints

FEIX_NAMES = ["Large Diameter", "Small Diameter", "Medium Wrap", "Adducted Thumb", "Light Tool", "Prismatic 4 Finger",
              "Prismatic 3 Finger", "Prismatic 2 Finger", "Palmar Pinch", "Power Disk", "Power Sphere", "Precision Disk",
              "Precision Sphere", "Tripod", "Fixed Hook", "Lateral", "Index Finger Extension", "Extension Type",
              "Distal Type", "Writing Tripod", "Tripod Variation", "Parallel Extension", "Adduction Grip", "Tip Pinch",
              "Lateral Tripod", "Sphere 4 Finger", "Quadpod", "Sphere 3 Finger", "Stick", "Palmar", "Ring", "Ventral",
              "Inferior Pincer"]


def convert_subject(path: str, out_dir: str, fps: float, min_run: int) -> dict:
    z = np.load(path)
    off, frame, J = z["offsets"], z["frame"], z["joints"]
    subj = int(os.path.basename(path)[1:3])
    ids, labels, feix, objs, keys, ts, js, rvs, shp = [], [], [], [], [], [], [], [], []
    n_frames_kept = 0
    for i in range(len(off) - 1):
        f = frame[off[i]: off[i + 1]]
        j = J[off[i]: off[i + 1]][:, OPENPOSE_TO_DC].astype(np.float64)
        order = np.argsort(f)
        f, j = f[order], j[order]
        cut = np.where(np.diff(f) != 1)[0] + 1
        for r, (fr, jr) in enumerate(zip(np.split(f, cut), np.split(j, cut))):
            if len(fr) < min_run or not np.isfinite(jr).all():
                continue
            g = int(z["feix"][i])
            ids.append(f"S{subj:02d}/{z['trial_key'][i]}/run{r}")
            labels.append(f"feix{g:02d}-{FEIX_NAMES[g - 1].replace(' ', '-')}")
            feix.append(g)
            objs.append(int(z["object_id"][i]))
            keys.append(str(z["trial_key"][i]))
            ts.append(fr.astype(np.float64) / fps)
            js.append(jr.astype(np.float32))
            rvs.append(root_rotvec_from_keypoints(jr).astype(np.float32))
            shp.append(z["mano_betas"][i])
            n_frames_kept += len(fr)
    if not ids:
        return {"subject": subj, "trajectories": 0, "frames": 0}
    np.savez(os.path.join(out_dir, f"S{subj:02d}.npz"), trajectory_id=np.array(ids), action_label=np.array(labels),
             feix=np.array(feix), subject=np.full(len(ids), subj), object_id=np.array(objs), trial_key=np.array(keys),
             offsets=np.concatenate([[0], np.cumsum([len(t) for t in ts])]).astype(np.int64), t=np.concatenate(ts),
             joints=np.concatenate(js), wrist_rotvec=np.concatenate(rvs), mano_shape=np.stack(shp))
    return {"subject": subj, "trajectories": len(ids), "frames": n_frames_kept}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default=os.path.join(DATA_ROOT, "hograspnet", "annotations"))
    ap.add_argument("--out", default=os.path.join(DATA_ROOT, "hograspnet", "keypoints"))
    ap.add_argument("--fps", type=float, default=10.0, help="annotation frame grid of HOGraspNet")
    ap.add_argument("--min_run", type=int, default=4, help="consecutive 10 fps frames; 4 frames span 0.3 s >= one 16-step chunk at 60 Hz (0.27 s)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = [convert_subject(p, a.out, a.fps, a.min_run) for p in sorted(glob.glob(os.path.join(a.index, "S*.npz")))]
    tot = {"subjects": len(res), "trajectories": sum(r["trajectories"] for r in res), "frames": sum(r["frames"] for r in res)}
    print(json.dumps(tot))
    save_manifest(a.out, a, total=tot)


if __name__ == "__main__":
    main()
