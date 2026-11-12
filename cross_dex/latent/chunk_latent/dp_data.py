"""Chunk-latent policy space for the common LeRobot Diffusion Policy (ours; mirrors dexverse_xlvla.make_dataset, which is
called for its helpers only — wrist calibration, stats, merge — and not modified).

Per teleop LeRobot set (read only), per frame t of an episode:
  action(t)            = [canonical wrist pose of action(t) (9) | Z(t) = E_h(finger targets action[t : t+H]) (K*d)]
                         (window padded with the episode's last frame), i.e. the latent of the finger chunk that starts at t
  observation.state(t) = [canonical wrist pose of state(t) (9) | mean_k E_h(state fingers(t) repeated H times) (d)
                         | task one-hot]
The DP (horizon 16, 8 executed) is trained on this unchanged; at evaluation (dp_policy.py) the Z of the first predicted
step is decoded into an H-step finger chunk, whose first n_action_steps steps are executed with the predicted wrist poses.
Finger columns are selected by name (= chunk dataset finger_joint_names = teleop action names [6:], checked 2026-11-12).

  /workspace/envs/dexverse/dexlatent/bin/python -m cross_dex.latent.chunk_latent.dp_data --latent <best.pt> \
      --xlvla_ckpt <XL-VLA latent .pt (hand models for the wrist)> --wrist <dir of wrist_calib_<task>.json> --out <root> \
      --task T1,T2,T3,T4 --hands shadow,inspire,sharpa,wuji --task_onehot
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

from cross_dex.latent.chunk_latent.model import HandCodec, load_model

ROBOT_TYPE = "cross_dex_chunk_latent"


def convert(codec: HandCodec, wrist, src: str, dst: str, onehot=None, finger_names=None):
    from dexverse_xlvla.make_dataset import KEEP, names, stats
    info = json.load(open(f"{src}/meta/info.json"))
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(f"{src}/meta", f"{dst}/meta")
    shutil.copytree(f"{src}/videos", f"{dst}/videos")
    files = sorted(glob.glob(f"{src}/data/*/*.parquet"))
    an, sn = names(info, "action"), names(info, "observation.state")
    ai = [an.index(n) for n in finger_names]
    si = [sn.index(n) for n in finger_names]
    H, K, d = codec.H, codec.K, codec.d
    dfs = [pd.read_parquet(f) for f in files]
    per_ep = {}
    for f, df in zip(files, dfs):
        A = np.stack(df["action"].to_numpy()).astype(np.float64)
        S = np.stack(df["observation.state"].to_numpy()).astype(np.float64)
        ep = df["episode_index"].to_numpy()
        a_out = np.zeros((len(df), 9 + K * d), np.float32)
        s_out = np.zeros((len(df), 9 + d), np.float32)
        a_out[:, :9] = wrist.vec_from_actions(A[:, :6])
        s_out[:, :9] = wrist.vec_from_joints(S[:, :6])
        for e in np.unique(ep):
            rows = np.where(ep == e)[0]
            fa = A[rows][:, ai]
            pad = np.concatenate([fa, np.repeat(fa[-1:], H - 1, 0)])
            win = np.stack([pad[t:t + H] for t in range(len(rows))])               # [T, H, n]
            z = torch.cat([codec.encode(win[i:i + 4096]) for i in range(0, len(win), 4096)]).cpu().numpy()
            a_out[rows, 9:] = z.reshape(len(rows), K * d)
            fs = np.repeat(S[rows][:, si][:, None], H, 1)                             # static chunk of the state
            zs = torch.cat([codec.encode(fs[i:i + 4096]) for i in range(0, len(fs), 4096)]).mean(1).cpu().numpy()
            s_out[rows, 9:] = zs
        if onehot is not None:
            s_out = np.concatenate([s_out, np.tile(np.asarray(onehot, np.float32), (len(s_out), 1))], 1)
        out = df[list(KEEP)].copy()
        out["action"], out["observation.state"] = list(a_out), list(s_out)
        rel = os.path.relpath(f, src)
        os.makedirs(os.path.dirname(f"{dst}/{rel}"), exist_ok=True)
        out[["observation.state", "action", *KEEP]].to_parquet(f"{dst}/{rel}")
        for e in np.unique(ep):
            k = ep == e
            per_ep[int(e)] = (a_out[k], s_out[k])
    wn = [f"wrist_{n}" for n in ("x", "y", "z", "r6d_0", "r6d_1", "r6d_2", "r6d_3", "r6d_4", "r6d_5")]
    a_names = wn + [f"z_{k}_{i}" for k in range(K) for i in range(d)]
    s_names = wn + [f"zs_{i}" for i in range(d)] + ([f"task_{i}" for i in range(len(onehot))] if onehot is not None else [])
    feats = {k: v for k, v in info["features"].items() if v["dtype"] == "video" or k in KEEP}
    feats["action"] = {"dtype": "float32", "shape": [len(a_names)], "names": a_names}
    feats["observation.state"] = {"dtype": "float32", "shape": [len(s_names)], "names": s_names}
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
    allstats = {k: v for k, v in json.load(open(f"{src}/meta/stats.json")).items() if k in feats}
    a = np.concatenate([v[0] for v in per_ep.values()])
    s = np.concatenate([v[1] for v in per_ep.values()])
    for key, x in (("action", a), ("observation.state", s)):
        allstats[key] = {k: v.tolist() for k, v in stats(x).items()}
    json.dump(allstats, open(f"{dst}/meta/stats.json", "w"), indent=4)
    print(f"[data] {dst}: {len(a)} frames, action {a.shape[1]}, state {s.shape[1]}", flush=True)
    return a, s


def main():
    from dexverse_xlvla.check_hands import dataset_dir
    from dexverse_xlvla.eval_latent import load
    from dexverse_xlvla.make_dataset import merge
    from dexverse_xlvla.wrist import load_wrists
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent", required=True, help="chunk latent checkpoint (train.py best.pt)")
    ap.add_argument("--xlvla_ckpt", required=True, help="XL-VLA latent checkpoint: hand models used by the wrist calibration")
    ap.add_argument("--wrist", required=True, help="folder of wrist_calib_<task>.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--hands", default="shadow,inspire,sharpa,wuji")
    ap.add_argument("--exclude", default="")
    ap.add_argument("--task_onehot", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(a.latent, dev)
    tr = load(a.xlvla_ckpt)
    tasks, hands = a.task.split(","), a.hands.split(",")
    excl = {tuple(c.split(":")) for c in a.exclude.split(",") if c}
    cells = [(t, h) for t in tasks for h in hands if (t, h) not in excl]
    fnames = ck["finger_joint_names"]   # = teleop action names [6:] (env action finger order)
    for i, t in enumerate(tasks):
        wrists = load_wrists(f"{a.wrist}/wrist_calib_{t}.json", {h: tr.hand_models[h] for h in hands})
        onehot = [float(j == i) for j in range(len(tasks))] if a.task_onehot else None
        for h in hands:
            if (t, h) in excl:
                continue
            dst = f"{a.out}/{t}/{h}"
            if os.path.exists(f"{dst}/.done"):
                continue
            convert(HandCodec(model, ck, h, dev), wrists[h], dataset_dir(t, h), dst, onehot, fnames[h])
            open(f"{dst}/.done", "w").close()
    merge([f"{a.out}/{t}/{h}" for t, h in cells], f"{a.out}/all")
    json.dump({"tasks": tasks, "hands": hands, "excluded": sorted(map(list, excl)), "task_onehot": a.task_onehot,
               "cells": [list(c) for c in cells], "latent": a.latent, "K": model.K, "d": model.d, "H": model.H},
              open(f"{a.out}/all/meta/chunk_latent_cells.json", "w"), indent=1)


if __name__ == "__main__":
    main()
