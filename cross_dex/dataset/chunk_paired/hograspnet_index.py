"""HOGraspNet annotations -> per-subject npz (flat arrays + offsets) and per-Feix-type statistics.

Input: <raw>/labeling_data/<date>_S<subj>_obj_<obj>_grasp_<feix>/trial_<k>/annotation/mas/mas_<frame>.json (the official
"annotations" download; one json per annotated frame, world = 'mas' camera frame). Only the 'mas' view is read
(the hand / object annotation is shared by the 4 views).

Per subject -> <out>/S<subj>.npz: trial_key [N], feix [N], object_id [N], offsets [N+1], frame [sum T] (annotated frame
index, 10 fps grid), joints [sum T,21,3] (3D hand joints, source cm -> m, OpenPose order: wrist, thumb 1-4, index 1-4, ..., pinky 1-4),
mano_pose [sum T,45] (finger joints only, no global rotation), mano_trans [sum T,3] (MANO fit units, with hand.mano_scale not applied), mano_betas [N,10],
object_mat [sum T,4,4]. Stats (-> <out>/stats.json): per Feix type the trials, annotated frames and runs of consecutive
frames (a run of >= --min_run frames at 10 fps covers one 16-step chunk at 60 Hz when >= 3 frames).

  PYTHONPATH=/workspace/codes/cross_dex /workspace/envs/dexverse/lerobot/bin/python -m cross_dex.dataset.chunk_paired.hograspnet_index
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from cross_dex.dataset.chunk_paired import DATA_ROOT, save_manifest

SEQ = re.compile(r"(?P<date>\d+)_S(?P<subj>\d+)_obj_(?P<obj>\d+)_grasp_(?P<feix>\d+)$")


def _arr(x, n):
    a = np.asarray(x, np.float64).reshape(-1)
    return a[:n] if a.size >= n else np.full(n, np.nan)


def index_subject(args) -> tuple[int, dict]:
    subj, seqs, out = args
    keys, feix, obj, off, fr, J, P, T, B, O = [], [], [], [0], [], [], [], [], [], []
    for sd in seqs:
        m = SEQ.match(os.path.basename(sd))
        for td in sorted(glob.glob(os.path.join(sd, "trial_*"))):
            files = glob.glob(os.path.join(td, "annotation", "mas", "mas_*.json"))
            items = sorted((int(os.path.basename(f)[4:-5]), f) for f in files)
            if not items:
                continue
            beta = None
            for k, f in items:
                d = json.load(open(f, encoding="utf-8-sig"))   # some files start with a UTF-8 BOM
                me = d["Mesh"][0]
                fr.append(k)
                J.append(_arr(d["annotations"][0]["data"], 63).reshape(21, 3) / 100.0)   # source is cm (camera z ~ 70) -> m
                P.append(_arr(me["mano_pose"], 45))
                T.append(_arr(me["mano_trans"], 3))
                O.append(_arr(me["object_mat"], 16).reshape(4, 4))
                beta = _arr(me["mano_betas"], 10)
            keys.append(f"{os.path.basename(sd)}/{os.path.basename(td)}")
            feix.append(int(m["feix"]))
            obj.append(int(m["obj"]))
            B.append(beta)
            off.append(off[-1] + len(items))
    if not keys:
        return subj, {}
    np.savez(os.path.join(out, f"S{subj:02d}.npz"), trial_key=np.array(keys), feix=np.array(feix), object_id=np.array(obj),
             offsets=np.array(off), frame=np.array(fr), joints=np.array(J, np.float32), mano_pose=np.array(P, np.float32),
             mano_trans=np.array(T, np.float32), mano_betas=np.array(B, np.float32), object_mat=np.array(O, np.float32))
    return subj, {"feix": feix, "frames": [np.asarray(fr[a:b]) for a, b in zip(off[:-1], off[1:])]}


def runs(frames: np.ndarray) -> list[int]:
    """Lengths of runs of consecutive frame indices."""
    if len(frames) == 0:
        return []
    cut = np.where(np.diff(frames) != 1)[0] + 1
    return [len(r) for r in np.split(frames, cut)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/workspace/datasets/dexverse/grasp_taxonomy/hograspnet/labeling_data")
    ap.add_argument("--out", default=os.path.join(DATA_ROOT, "hograspnet", "annotations"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--min_run", type=int, default=3, help="consecutive 10 fps frames that span a 16-step 60 Hz chunk")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    by_subj: dict[int, list[str]] = {}
    for sd in sorted(glob.glob(os.path.join(a.raw, "*_S*_obj_*_grasp_*"))):
        m = SEQ.match(os.path.basename(sd))
        if m:
            by_subj.setdefault(int(m["subj"]), []).append(sd)
    stats: dict[int, dict] = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for subj, r in ex.map(index_subject, [(s, v, a.out) for s, v in sorted(by_subj.items())]):
            for g, frs in zip(r.get("feix", []), r.get("frames", [])):
                s = stats.setdefault(g, {"trials": 0, "frames": 0, "run_lengths": []})
                s["trials"] += 1
                s["frames"] += len(frs)
                s["run_lengths"] += runs(frs)
            print("S", subj, "trials", len(r.get("feix", [])), flush=True)
    summ = {}
    for g in sorted(stats):
        rl = np.asarray(stats[g]["run_lengths"])
        long = rl[rl >= a.min_run]
        summ[g] = {"trials": stats[g]["trials"], "frames": stats[g]["frames"],
                   "frames_per_trial_mean": round(stats[g]["frames"] / max(stats[g]["trials"], 1), 1),
                   "runs_ge_min": int(len(long)), "frames_in_runs_ge_min": int(long.sum()),
                   "longest_run": int(rl.max()) if len(rl) else 0}
    tot = {k: sum(v[k] for v in summ.values()) for k in ("trials", "frames", "runs_ge_min", "frames_in_runs_ge_min")}
    json.dump({"per_feix": summ, "total": tot, "subjects": len(by_subj), "args": vars(a)},
              open(os.path.join(os.path.dirname(a.out), "stats.json"), "w"), indent=1)
    print(json.dumps(tot))
    save_manifest(a.out, a, total=tot)
    for g, s in summ.items():
        print(f"feix {g:2d}", s)


if __name__ == "__main__":
    main()
