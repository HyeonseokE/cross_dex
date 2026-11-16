"""Bimanual training sets for the pi0.5 baselines (make_dataset.py / pi05/make_joint_dataset.py for one hand):
one LeRobot v3.0 set per (task, hand) bimanual teleop cell whose observation.state / action are the bimanual policy space
(dexverse_xlvla.bimanual), then <out>/all = every cell (make_dataset.merge: file copy + index shift).

  --space latent : [R wrist 9 | R z 32 | L wrist 9 | L z 32] = 82  (pi05_XL;  openpi_train --action_dim 82 --max_token_len 360)
  --space joint  : [R wrist 9 | R fingers 22 | L wrist 9 | L fingers 22] = 62  (pi05; --action_dim 62 --max_token_len 280)

Videos (third_person, eye_in_hand = right wrist camera, eye_in_hand_left) and meta are copied, the parquet keeps only the
index columns plus the two vectors, stats are recomputed (as make_dataset). The dataset's action / state names must equal
the env action joints read from the simulator (wrist calibration), otherwise the cell is refused.

  python -m dexverse_xlvla.make_dataset_bimanual --space latent --task <t1,...> --hands <h1,...> --wrist <calib folder>
         --ckpt <right latent.pt> --ckpt_left <left latent.pt> --out <root> [--exclude <task>:<hand>,...]
  (--space joint: --ckpt / --ckpt_left not needed)
Cells split over servers: on each server --missing_ok --no_merge (converts the cells whose teleop set is there), copy
the <out>/<task>/<hand> folders to one place, then --merge_only there (merges every cell with a .done marker).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import numpy as np
import pandas as pd

import dexverse_xlvla.make_dataset as MD
from dexverse_xlvla import bimanual as B

LEROBOT_ROOT = os.environ.get("XLVLA_TELEOP_ROOT", "/workspace/datasets/dexverse/teleop_lerobot")   # read only
SKIP = ("_failed_", "_r1_", "_try")
TASKS = ("Dexverse-OpenFlatFolder-v1", "Dexverse-SlideUtilityKnife-v1", "Dexverse-BimanualLiftTray-v0",
         "Dexverse-BimanualLiftCarton-v0")
HANDS = ("shadow", "inspire", "sharpa", "wuji", "allegro", "leap")


def bimanual_dir(task: str, hand: str) -> str:
    hits = [h for h in sorted(glob.glob(f"{LEROBOT_ROOT}/{task}/{hand}_bimanual/*_bimanual/meta/info.json"))
            if not any(s in h for s in SKIP)]
    if len(hits) != 1:
        raise FileNotFoundError(f"{task} {hand}: need one bimanual LeRobot set, found {hits}")
    return os.path.dirname(os.path.dirname(hits[0]))


def available(tasks, hands) -> list:
    out = []
    for t in tasks:
        for h in hands:
            try:
                bimanual_dir(t, h)
                out.append((t, h))
            except FileNotFoundError:
                pass
    return out


def names_ok(info: dict, space: B.BimanualSpace) -> bool:
    """action names == env action joints; state names = env action joints first (+ passive joints, e.g. sharpa / inspire)."""
    st = MD.names(info, "observation.state")
    return MD.names(info, "action") == space.action_joints and st[: space.n_act] == space.action_joints and \
        (space.state_joints is None or st == space.state_joints[: len(st)])


def convert(space: B.BimanualSpace, src: str, dst: str, robot_type: str) -> None:
    info = json.load(open(f"{src}/meta/info.json"))
    if not names_ok(info, space):
        raise ValueError(f"{src}: action / state names != simulator action joints (state = action joints first, then the "
                         f"other joints, as the evaluator's state order)\n{MD.names(info, 'action')}\n{space.action_joints}")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(f"{src}/meta", f"{dst}/meta")
    shutil.copytree(f"{src}/videos", f"{dst}/videos")
    per_ep = {}
    for f in sorted(glob.glob(f"{src}/data/*/*.parquet")):
        df = pd.read_parquet(f)
        a = space.to_policy(np.stack(df["action"].to_numpy()), "action")
        s = space.to_policy(np.stack(df["observation.state"].to_numpy()), "state")
        out = df[list(MD.KEEP)].copy()
        out["action"], out["observation.state"] = list(a), list(s)
        rel = os.path.relpath(f, src)
        os.makedirs(os.path.dirname(f"{dst}/{rel}"), exist_ok=True)
        out[["observation.state", "action", *MD.KEEP]].to_parquet(f"{dst}/{rel}")
        for ep in np.unique(df["episode_index"]):
            k = df["episode_index"].to_numpy() == ep
            per_ep[int(ep)] = (a[k], s[k])
    feats = {k: v for k, v in info["features"].items() if v["dtype"] == "video" or k in MD.KEEP}
    for k in ("observation.state", "action"):
        feats[k] = {"dtype": "float32", "shape": [space.dim], "names": space.names()}
    info["features"], info["robot_type"] = feats, robot_type
    json.dump(info, open(f"{dst}/meta/info.json", "w"), indent=4)
    gone = [k for k in json.load(open(f"{src}/meta/info.json"))["features"] if k not in feats]
    for f in sorted(glob.glob(f"{dst}/meta/episodes/*/*.parquet")):
        df = pd.read_parquet(f)
        df = df[[c for c in df.columns if not any(c.startswith(f"stats/{g}/") for g in gone)]].copy()
        rows = [per_ep[int(e)] for e in df["episode_index"]]
        for j, key in ((0, "action"), (1, "observation.state")):
            per_row = [MD.stats(r[j]) for r in rows]
            for st in per_row[0]:
                df[f"stats/{key}/{st}"] = [p[st] for p in per_row]
        df.to_parquet(f)
    allstats = {k: v for k, v in json.load(open(f"{src}/meta/stats.json")).items() if k in feats}
    a = np.concatenate([v[0] for v in per_ep.values()])
    s = np.concatenate([v[1] for v in per_ep.values()])
    for key, x in (("action", a), ("observation.state", s)):
        allstats[key] = {k: v.tolist() for k, v in MD.stats(x).items()}
    json.dump(allstats, open(f"{dst}/meta/stats.json", "w"), indent=4)
    print(f"[data-bi] {space.hand}: {len(a)} frames, {space.dim} dims -> {dst}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", choices=["latent", "joint"], required=True)
    ap.add_argument("--task", default=",".join(TASKS))
    ap.add_argument("--hands", default=",".join(HANDS))
    ap.add_argument("--wrist", required=True, help="folder with wrist_calib_<task>.json (measure_wrist_bimanual.py)")
    ap.add_argument("--ckpt", help="right 6-hand latent (latent space)")
    ap.add_argument("--ckpt_left", help="left 6-hand latent (latent space)")
    ap.add_argument("--pad", type=int, default=B.PAD, help="joint space: fingers per hand zero-padded to this")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default="", help="comma-separated <task>:<hand> cells left out of <out>/all")
    ap.add_argument("--missing_ok", action="store_true", help="skip cells whose teleop set is not on this server")
    ap.add_argument("--merge_only", action="store_true")
    ap.add_argument("--no_merge", action="store_true", help="convert the cells only")
    a = ap.parse_args()
    tasks, hands = a.task.split(","), a.hands.split(",")
    excl = {tuple(c.split(":")) for c in a.exclude.split(",") if c}
    if a.merge_only:   # cells converted before (possibly on other servers and copied here): every <out>/<t>/<h>/.done
        cells = [(t, h) for t in tasks for h in hands if os.path.exists(f"{a.out}/{t}/{h}/.done") and (t, h) not in excl]
    else:
        cells = [c for c in ([(t, h) for t in tasks for h in hands] if not a.missing_ok else available(tasks, hands))
                 if c not in excl]
    if a.space == "latent" and not (a.ckpt and a.ckpt_left) and not a.merge_only:
        ap.error("--space latent needs --ckpt and --ckpt_left")
    robot_type = f"xlvla_{a.space}_bimanual_dexverse"
    if not a.merge_only:
        spaces = {}
        for t, h in cells:
            d = f"{a.out}/{t}/{h}"
            if os.path.exists(f"{d}/.done"):
                continue
            cal = f"{a.wrist}/wrist_calib_{t}.json"
            if h not in spaces or spaces[h][0] != cal:
                spaces[h] = (cal, B.build(cal, h, a.space, a.ckpt, a.ckpt_left, a.pad))
            print(f"[data-bi] {t} {h}: finger names {[(s, spaces[h][1].side[s]['rule']) for s in B.SIDES]}", flush=True)
            convert(spaces[h][1], bimanual_dir(t, h), d, robot_type)
            open(f"{d}/.done", "w").close()
    if a.no_merge:
        return
    MD.merge([f"{a.out}/{t}/{h}" for t, h in cells], f"{a.out}/all")
    json.dump({"space": a.space, "tasks": tasks, "hands": hands, "excluded": sorted(map(list, excl)),
               "cells": [list(c) for c in cells], "ckpt": a.ckpt, "ckpt_left": a.ckpt_left, "wrist": a.wrist,
               "pad": a.pad if a.space == "joint" else None},
              open(f"{a.out}/all/meta/xlvla_cells.json", "w"), indent=1)


if __name__ == "__main__":
    main()
