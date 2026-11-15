"""Step 5 data: one LeRobot dataset per hand whose observation.state and action are the XL-VLA policy space
[canonical wrist pose (9) | hand latent (32)] = 41 dims, then one multi-hand dataset <out>/all (merge(): file copy +
index shift; lerobot's aggregate_datasets truncated the merged videos).
The common LeRobot training code (teleop_dp_image/train_lerobot.py) then trains on it unchanged.

Per hand, from the teleop LeRobot set (read only): videos and meta are copied, data parquet keeps only the index columns,
observation.state = [wrist(state joints), E_h(state finger joints)], action = [wrist(action joints), E_h(action finger
joints)], with E_h = posterior mean of the hand encoder. Stats (episode and dataset) are recomputed for the two vectors;
the other features (tactile, ee, ...) are dropped so that every hand has the same features (aggregate requirement).

  python -m dexverse_xlvla.make_dataset --ckpt <latent.pt> --wrist <calib.json> --out <root> [--task ...]
  multi-task: --task <t1,t2,...> --wrist <folder of wrist_calib_<task>.json> [--exclude <task>:<hand>,...] [--task_onehot]
    -> <out>/<task>/<hand> per cell, <out>/all = every cell not excluded (task table = union, task_index remapped)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import numpy as np
import pandas as pd
import torch

from dexverse_xlvla import hands as H
from dexverse_xlvla.check_hands import dataset_dir
from dexverse_xlvla.eval_latent import enc, load
from dexverse_xlvla.wrist import load_wrists

KEEP = ("timestamp", "frame_index", "episode_index", "index", "task_index")
QS = (0.01, 0.10, 0.50, 0.90, 0.99)
ROBOT_TYPE = "xlvla_latent_dexverse"


def stats(x: np.ndarray) -> dict:
    s = {"min": x.min(0), "max": x.max(0), "mean": x.mean(0), "std": x.std(0), "count": np.array([len(x)])}
    for q in QS:
        s[f"q{int(q * 100):02d}"] = np.quantile(x, q, axis=0)
    return s


def names(info, key):
    n = info["features"][key]["names"]
    return list(n.values())[0] if isinstance(n, dict) else n


def convert(tr, wrists, hand, src, dst, onehot=None):
    """onehot: task one-hot (list) appended to observation.state (multi-task policies), or None."""
    m = tr.hand_models[hand]
    info = json.load(open(f"{src}/meta/info.json"))
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(f"{src}/meta", f"{dst}/meta")
    shutil.copytree(f"{src}/videos", f"{dst}/videos")
    files = sorted(glob.glob(f"{src}/data/*/*.parquet"))

    def policy_space(q, cols, kind):   # kind: "action" (wrist action terms) or "observation.state" (wrist joints)
        wrist = wrists[hand].vec_from_actions(q[:, :6]) if kind == "action" else wrists[hand].vec_from_joints(q[:, :6])
        fingers = m.angles_to_normalized(torch.tensor(q[:, m.data_index(cols)], dtype=torch.float32)).clamp(-1, 1)
        with torch.no_grad():
            z = enc(tr, hand, fingers).numpy()
        return np.concatenate([wrist, z], 1).astype(np.float32)

    per_ep = {}
    for f in files:
        df = pd.read_parquet(f)
        a = policy_space(np.stack(df["action"].to_numpy()).astype(np.float64), names(info, "action"), "action")
        s = policy_space(np.stack(df["observation.state"].to_numpy()).astype(np.float64), names(info, "observation.state"),
                         "observation.state")
        if onehot is not None:
            s = np.concatenate([s, np.tile(np.asarray(onehot, np.float32), (len(s), 1))], 1)
        out = df[list(KEEP)].copy()
        out["action"], out["observation.state"] = list(a), list(s)
        rel = os.path.relpath(f, src)
        os.makedirs(os.path.dirname(f"{dst}/{rel}"), exist_ok=True)
        out[["observation.state", "action", *KEEP]].to_parquet(f"{dst}/{rel}")
        for ep in np.unique(df["episode_index"]):
            k = df["episode_index"].to_numpy() == ep
            per_ep[int(ep)] = (a[k], s[k])

    dims = 9 + tr.config.latent_dim_hand
    vec_names = [f"wrist_{n}" for n in ("x", "y", "z", "r6d_0", "r6d_1", "r6d_2", "r6d_3", "r6d_4", "r6d_5")] + \
                [f"z_{i}" for i in range(tr.config.latent_dim_hand)]
    feats = {k: v for k, v in info["features"].items() if v["dtype"] == "video" or k in KEEP}
    for k in ("observation.state", "action"):
        feats[k] = {"dtype": "float32", "shape": [dims], "names": vec_names}
    if onehot is not None:
        feats["observation.state"] = {"dtype": "float32", "shape": [dims + len(onehot)],
                                      "names": vec_names + [f"task_{i}" for i in range(len(onehot))]}
    info["features"], info["robot_type"] = feats, ROBOT_TYPE
    json.dump(info, open(f"{dst}/meta/info.json", "w"), indent=4)

    gone = [k for k in json.load(open(f"{src}/meta/info.json"))["features"] if k not in feats]
    for f in sorted(glob.glob(f"{dst}/meta/episodes/*/*.parquet")):
        df = pd.read_parquet(f)
        df = df[[c for c in df.columns if not any(c.startswith(f"stats/{g}/") for g in gone)]].copy()
        rows = [per_ep[int(e)] for e in df["episode_index"]]
        for j, key in ((0, "action"), (1, "observation.state")):
            per_row = [stats(r[j]) for r in rows]
            for st in per_row[0]:
                df[f"stats/{key}/{st}"] = [p[st] for p in per_row]
        df.to_parquet(f)
    allstats = json.load(open(f"{src}/meta/stats.json"))
    allstats = {k: v for k, v in allstats.items() if k in feats}
    a = np.concatenate([v[0] for v in per_ep.values()])
    s = np.concatenate([v[1] for v in per_ep.values()])
    for key, x in (("action", a), ("observation.state", s)):
        allstats[key] = {k: v.tolist() for k, v in stats(x).items()}
    json.dump(allstats, open(f"{dst}/meta/stats.json", "w"), indent=4)
    print(f"[data] {hand}: {len(a)} frames -> {dst}", flush=True)


CHUNK = 1000   # LeRobot v3.0 chunks_size


def merge(roots: list, dst: str) -> None:
    """Concatenate per-hand datasets into one LeRobot v3.0 dataset by copying files and shifting indices.
    Replaces lerobot.datasets.aggregate.aggregate_datasets (0.4.2), whose video concatenation of these per-episode AV1
    files left the merged mp4 far shorter than the merged meta (third_person 3789 frames vs 460 s of timestamps)."""
    from lerobot.datasets.compute_stats import aggregate_stats
    if os.path.exists(dst):
        shutil.rmtree(dst)
    infos = [json.load(open(f"{r}/meta/info.json")) for r in roots]
    tasks = [pd.read_parquet(f"{r}/meta/tasks.parquet") for r in roots]      # index = task string, column task_index
    union = list(dict.fromkeys(t for tb in tasks for t in tb.sort_values("task_index").index))
    remap = [{int(i): union.index(t) for t, i in zip(tb.index, tb["task_index"])} for tb in tasks]
    vkeys = [k for k, v in infos[0]["features"].items() if v["dtype"] == "video"]
    assert all(i["features"] == infos[0]["features"] and i["fps"] == infos[0]["fps"] for i in infos)
    os.makedirs(f"{dst}/meta/episodes/chunk-000", exist_ok=True)
    pd.DataFrame({"task_index": range(len(union))}, index=pd.Index(union, name=tasks[0].index.name)).to_parquet(
        f"{dst}/meta/tasks.parquet")
    rows, ep_off, fr_off, nfile = [], 0, 0, {k: 0 for k in ["data", *vkeys]}
    for r, info, rm in zip(roots, infos, remap):
        eps = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{r}/meta/episodes/*/*.parquet"))])
        eps = eps.sort_values("episode_index").reset_index(drop=True)
        if len(eps) != info["total_episodes"]:   # source meta inconsistent (e.g. Bleach leap: info 44 eps, 50 episode files):
            print(f"[merge] {r}: {len(eps)} episode rows but info.json total_episodes={info['total_episodes']} -> keep the "
                  f"first {info['total_episodes']} (indices beyond would collide with the next dataset)", flush=True)
            eps = eps[eps["episode_index"] < info["total_episodes"]].reset_index(drop=True)
            assert int(eps["dataset_to_index"].max()) == info["total_frames"], "frames of kept episodes != info total_frames"
        fmap = {}                                                      # (key, chunk, file) -> new file index
        for key, pat in [("data", "data/chunk-{c:03d}/file-{f:03d}.parquet")] + \
                        [(k, f"videos/{k}/chunk-{{c:03d}}/file-{{f:03d}}.mp4") for k in vkeys]:
            col = "data" if key == "data" else f"videos/{key}"
            for c, f in sorted(set(zip(eps[f"{col}/chunk_index"], eps[f"{col}/file_index"]))):
                n = divmod(nfile[key], CHUNK)   # (chunk, file): <= CHUNK files per chunk — LeRobot reads files in name order,
                nfile[key] += 1                 # so file-1000 in one chunk would sort before file-101 and shift every row
                fmap[(key, c, f)] = n
                os.makedirs(os.path.dirname(f"{dst}/{pat.format(c=n[0], f=n[1])}"), exist_ok=True)
                if key == "data":
                    df = pd.read_parquet(f"{r}/{pat.format(c=c, f=f)}")
                    df["episode_index"] += ep_off
                    df["index"] += fr_off
                    df["task_index"] = df["task_index"].map(rm)
                    df.to_parquet(f"{dst}/{pat.format(c=n[0], f=n[1])}")
                else:
                    shutil.copy(f"{r}/{pat.format(c=c, f=f)}", f"{dst}/{pat.format(c=n[0], f=n[1])}")
            new = [fmap[(key, c, f)] for c, f in zip(eps[f"{col}/chunk_index"], eps[f"{col}/file_index"])]
            eps[f"{col}/chunk_index"] = [n[0] for n in new]
            eps[f"{col}/file_index"] = [n[1] for n in new]
        if "task_index" in eps:
            eps["task_index"] = eps["task_index"].map(rm)
        for st in ("min", "max", "mean", "q01", "q10", "q50", "q90", "q99"):      # one task per episode: constant
            c = f"stats/task_index/{st}"
            if c in eps:
                eps[c] = [np.asarray([rm[int(round(float(np.ravel(v)[0])))]], dtype=np.asarray(v).dtype) for v in eps[c]]
        eps["episode_index"] += ep_off
        eps["dataset_from_index"] += fr_off
        eps["dataset_to_index"] += fr_off
        rows.append(eps)
        ep_off += info["total_episodes"]
        fr_off += info["total_frames"]
    eps = pd.concat(rows).reset_index(drop=True)
    eps["meta/episodes/chunk_index"], eps["meta/episodes/file_index"] = 0, 0
    eps.to_parquet(f"{dst}/meta/episodes/chunk-000/file-000.parquet")
    st = aggregate_stats([{k: {s: np.asarray(v) for s, v in d.items()} for k, d in json.load(open(f"{r}/meta/stats.json")).items()}
                          for r in roots])
    json.dump({k: {s: np.asarray(v).tolist() for s, v in d.items()} for k, d in st.items()},
              open(f"{dst}/meta/stats.json", "w"), indent=4)
    info = dict(infos[0], total_episodes=ep_off, total_frames=fr_off, total_tasks=len(union), splits={"train": f"0:{ep_off}"})
    json.dump(info, open(f"{dst}/meta/info.json", "w"), indent=4)
    print(f"[merge] {len(roots)} datasets: {ep_off} episodes, {fr_off} frames -> {dst}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="Dexverse-GraspKettle-v0", help="task, or comma-separated tasks (multi-task)")
    ap.add_argument("--ckpt")
    ap.add_argument("--wrist", help="calibration json (one task), or a folder with wrist_calib_<task>.json per task")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hands", default=",".join(H.HAND_NAMES))
    ap.add_argument("--exclude", default="", help="comma-separated <task>:<hand> cells left out of <out>/all")
    ap.add_argument("--task_onehot", action="store_true", help="append the task one-hot (order of --task) to the state")
    ap.add_argument("--merge_only", action="store_true", help="only merge the converted sets into <out>/all")
    args = ap.parse_args()
    hands, tasks = args.hands.split(","), args.task.split(",")
    multi = len(tasks) > 1
    sub = (lambda t, h: f"{args.out}/{t}/{h}") if multi else (lambda t, h: f"{args.out}/{h}")
    excl = {tuple(c.split(":")) for c in args.exclude.split(",") if c}
    bad = excl - {(t, h) for t in tasks for h in hands}
    if bad:
        ap.error(f"--exclude cells not in --task x --hands: {sorted(bad)}")
    cells = [(t, h) for t in tasks for h in hands if (t, h) not in excl]
    if not args.merge_only:
        tr = load(args.ckpt)
        for a in tr.autoencoders.values():
            a.eval()
        for i, t in enumerate(tasks):
            cal = f"{args.wrist}/wrist_calib_{t}.json" if os.path.isdir(args.wrist) else args.wrist
            if multi and not os.path.isdir(args.wrist):
                ap.error("several tasks need --wrist <folder> with one calibration per task")
            wrists = load_wrists(cal, {h: tr.hand_models[h] for h in hands})
            onehot = [float(j == i) for j in range(len(tasks))] if args.task_onehot else None
            for h in hands:
                if (t, h) in excl:
                    continue
                if os.path.exists(f"{sub(t, h)}/meta/info.json") and os.path.exists(f"{sub(t, h)}/.done"):
                    continue                                              # converted before (resume)
                convert(tr, wrists, h, dataset_dir(t, h), sub(t, h), onehot)
                open(f"{sub(t, h)}/.done", "w").close()
    merge([sub(t, h) for t, h in cells], f"{args.out}/all")
    json.dump({"tasks": tasks, "hands": hands, "excluded": sorted(map(list, excl)), "task_onehot": args.task_onehot,
               "cells": [list(c) for c in cells]}, open(f"{args.out}/all/meta/xlvla_cells.json", "w"), indent=1)


if __name__ == "__main__":
    main()
