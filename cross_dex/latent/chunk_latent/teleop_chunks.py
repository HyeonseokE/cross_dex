"""Finger-target chunks of teleop LeRobot demonstrations (unpaired, one hand each) for the chunk-latent training
(train.py --teleop) and as the common test set of how well a latent covers the task action distribution.

Per dataset {task}/{hand}_{right|bimanual}/*/ (failed / previous attempts skipped): action columns of the hand's finger
joints (names = chunk dataset finger_joint_names; bimanual sets: the right hand's columns, matched exactly or with a
right-hand prefix), windows of H steps with --stride, per episode. Output npz: q_<hand> [N, H, n], task_<hand> [N]
(index into tasks), ep_<hand> [N]; json sidecar with tasks and counts.

  python -m cross_dex.latent.chunk_latent.teleop_chunks --tasks T1,T2 --episodes 10-49 --out <npz>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

ROOT = os.environ.get("TELEOP_LEROBOT_ROOT", "/workspace/datasets/dexverse/teleop_lerobot")
PREFIX = ("", "right_", "rh_", "R_", "r_")


def columns(names: list[str], fingers: list[str]) -> list[int] | None:
    for p in PREFIX:
        if all(p + f in names for f in fingers):
            return [names.index(p + f) for f in fingers]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--hands", default="shadow,inspire,sharpa,wuji")
    ap.add_argument("--fingers", default="/workspace/datasets/cross_dex/chunk_paired_dataset/chunks/feix400v2_H16/meta.json",
                    help="chunk dataset meta.json (finger_joint_names)")
    ap.add_argument("--episodes", default="", help="a-b: keep episode indices in [a, b] (default all)")
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fingers = json.load(open(a.fingers))["finger_joint_names"]
    tasks = a.tasks.split(",")
    e0, e1 = (int(x) for x in a.episodes.split("-")) if a.episodes else (0, 10 ** 9)
    out, info = {}, {"tasks": tasks, "sets": [], "args": vars(a)}
    for h in a.hands.split(","):
        Q, TI, EP = [], [], []
        for ti, t in enumerate(tasks):
            for d in sorted(glob.glob(f"{ROOT}/{t}/{h}_*/*/meta/info.json")):
                sd = d.split("/")[-4]
                if not re.fullmatch(rf"{h}_(right|bimanual)", sd):
                    continue                                   # failed / previous attempts
                ds = os.path.dirname(os.path.dirname(d))
                meta = json.load(open(d))
                n = meta["features"]["action"]["names"]
                n = list(n.values())[0] if isinstance(n, dict) else n
                ix = columns(n, fingers[h])
                if ix is None:
                    info["sets"].append({"set": ds, "skipped": "finger columns not found"})
                    continue
                k = 0
                for f in sorted(glob.glob(f"{ds}/data/*/*.parquet")):
                    df = pd.read_parquet(f, columns=["action", "episode_index"])
                    act = np.stack(df["action"].to_numpy())[:, ix].astype(np.float32)
                    ep = df["episode_index"].to_numpy()
                    for e in np.unique(ep):
                        if not e0 <= e <= e1:
                            continue
                        q = act[ep == e]
                        for s in range(0, len(q) - a.H + 1, a.stride):
                            Q.append(q[s:s + a.H]); TI.append(ti); EP.append(int(e)); k += 1
                info["sets"].append({"set": ds, "hand": h, "chunks": k})
        if Q:
            out[f"q_{h}"], out[f"task_{h}"], out[f"ep_{h}"] = np.stack(Q), np.array(TI, np.int16), np.array(EP, np.int16)
        info[h] = len(Q)
    np.savez(a.out, **out)
    json.dump(info, open(a.out[:-4] + ".json", "w"), indent=1)
    print(json.dumps({h: info[h] for h in a.hands.split(",")}))


if __name__ == "__main__":
    main()
