"""Multi-task multi-hand DP sets for compositional generalization (RSS2027, 2026-11-23).

4 tasks x 6 hands with ONE (task, hand) cell held out (default GraspKettle x shadow): the policy has seen shadow on the
other tasks and GraspKettle on the other hands, never the pair. Same policy vector as make_data.py (canonical wrist 9 |
z of the space, or normalized joints padded to 22 for joint), plus the task one-hot (4) in the state for every arm;
joint additionally has the hand one-hot (6) before it (state = [wrist | fingers | hand 6 | task 4]).
Built with the XL-VLA baseline's converter (dexverse_xlvla.make_dataset convert/merge, called unchanged).

  python -m cross_dex.ours.crosshand_dp.make_data_mt --space xlvla --ckpt <model.pt> --wrist <folder of wrist_calib_<task>.json> \
     --out <root>   (dexlatent env, run from codes/dexverse/DexLatent) -> <root>/<task>/<hand>, <root>/cg (merged, cell excluded)
"""
import argparse
import json
import os

os.environ.setdefault("XLVLA_TELEOP_ROOT", "/workspace/datasets/dexverse/teleop_lerobot_goal")

import dexverse_xlvla.make_dataset as MD  # noqa: E402
from dexverse_xlvla.check_hands import dataset_dir  # noqa: E402
from dexverse_xlvla.wrist import load_wrists  # noqa: E402

from cross_dex.ours.crosshand_dp.make_data import HANDS, PAD, enc_joint  # noqa: E402

TASKS = ["Dexverse-GraspKettle-v0", "Dexverse-GraspBleach-v0", "Dexverse-GraspPan-v0", "Dexverse-FunctionalHammerStrike-v0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wrist", required=True, help="folder with wrist_calib_<task>.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", default="Dexverse-GraspKettle-v0:shadow")
    a = ap.parse_args()
    joint = a.space == "joint"
    tr = MD.load(a.ckpt)
    for m in tr.autoencoders.values():
        m.eval()
    if joint:
        MD.enc = enc_joint
        tr.config.latent_dim_hand = PAD
        MD.ROBOT_TYPE = "crosshand_joint_dexverse"
    excl = tuple(a.exclude.split(":"))
    cells = []
    for ti, t in enumerate(TASKS):
        wrists = load_wrists(f"{a.wrist}/wrist_calib_{t}.json", {h: tr.hand_models[h] for h in HANDS})
        task1h = [float(j == ti) for j in range(len(TASKS))]
        for hi, h in enumerate(HANDS):
            if (t, h) == excl:
                continue
            cells.append((t, h)); dst = f"{a.out}/{t}/{h}"
            if os.path.exists(f"{dst}/.done"):
                continue
            onehot = ([float(j == hi) for j in range(len(HANDS))] if joint else []) + task1h
            MD.convert(tr, wrists, h, dataset_dir(t, h), dst, onehot)
            open(f"{dst}/.done", "w").close()
            print("converted", t, h, flush=True)
    MD.merge([f"{a.out}/{t}/{h}" for t, h in cells], f"{a.out}/cg")
    json.dump({"tasks": TASKS, "hands": list(HANDS), "excluded": [list(excl)], "space": a.space, "ckpt": a.ckpt,
               "wrist": a.wrist, "source_root": os.environ["XLVLA_TELEOP_ROOT"], "task_onehot_order": TASKS,
               "hand_onehot_order": list(HANDS) if joint else None, "pad": PAD if joint else None,
               "cells": [list(c) for c in cells]},
              open(f"{a.out}/cg/meta/crosshand_cells.json", "w"), indent=1)


if __name__ == "__main__":
    main()
