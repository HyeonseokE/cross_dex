"""Multi-hand single-task DP datasets (RSS2027 shared-space policy comparison, 2026-11-20).

One LeRobot set per hand from the teleop sets (read only), then merged sets for the two experiments:
  a6          all six hands                    (experiment A: hands trained together)
  b_noshadow  every hand but shadow            (experiment B: shadow unseen by the policy)

Policy vector (observation.state and action), built with the XL-VLA baseline's converter (dexverse_xlvla.make_dataset,
called unchanged):
  space  [canonical wrist pose 9 | z of the shared space (E_h posterior mean)]        no hand id
  joint  [canonical wrist pose 9 | finger joints normalised to [-1, 1], zero-padded to 22] + hand one-hot (6) in the state
The wrist part is the same in every arm, so only the finger representation differs between the arms.
Evaluation decodes it as dexverse_xlvla.policy (latent_dp): wrist by IK, fingers = D_h(z) (joint: first n_h values).

  python -m cross_dex.ours.crosshand_dp.make_data --space eh --ckpt <model.pt> --wrist <wrist_calib_<task>.json> --out <root>
  (dexlatent env, run from codes/dexverse/DexLatent; source sets: $XLVLA_TELEOP_ROOT, default teleop_lerobot_goal)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

os.environ.setdefault("XLVLA_TELEOP_ROOT", "/workspace/datasets/dexverse/teleop_lerobot_goal")   # sets with goal cues

import dexverse_xlvla.make_dataset as MD  # noqa: E402
from dexverse_xlvla.check_hands import dataset_dir  # noqa: E402
from dexverse_xlvla.wrist import load_wrists  # noqa: E402

TASK = "Dexverse-GraspKettle-v0"
HANDS = ("shadow", "inspire", "sharpa", "wuji", "allegro", "leap")
SETS = {"a6": HANDS, "b_noshadow": tuple(h for h in HANDS if h != "shadow")}
PAD = 22


def enc_joint(tr, hand, fingers):        # replaces the encoder: normalised finger joints, zero-padded to PAD
    f = fingers.numpy()
    out = np.zeros((len(f), PAD), np.float32)
    out[:, : f.shape[1]] = f
    return torch.from_numpy(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True, help="name of the arm (xlvla, ea, eh, e2c, joint)")
    ap.add_argument("--ckpt", required=True, help="space checkpoint (joint: any XL-VLA checkpoint, hand models only)")
    ap.add_argument("--wrist", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    joint = a.space == "joint"
    tr = MD.load(a.ckpt)
    for m in tr.autoencoders.values():
        m.eval()
    if joint:
        MD.enc = enc_joint
        tr.config.latent_dim_hand = PAD          # make_dataset sizes the vector as 9 + latent_dim_hand
        MD.ROBOT_TYPE = "crosshand_joint_dexverse"
    wrists = load_wrists(a.wrist, {h: tr.hand_models[h] for h in HANDS})
    for i, h in enumerate(HANDS):
        dst = f"{a.out}/{h}"
        if os.path.exists(f"{dst}/.done"):
            continue
        onehot = [float(j == i) for j in range(len(HANDS))] if joint else None
        MD.convert(tr, wrists, h, dataset_dir(TASK, h), dst, onehot)
        open(f"{dst}/.done", "w").close()
    for name, hands in SETS.items():
        MD.merge([f"{a.out}/{h}" for h in hands], f"{a.out}/{name}")
        json.dump({"task": TASK, "hands": list(hands), "space": a.space, "ckpt": a.ckpt, "wrist": a.wrist,
                   "source_root": os.environ["XLVLA_TELEOP_ROOT"], "hand_onehot_order": list(HANDS) if joint else None,
                   "pad": PAD if joint else None},
                  open(f"{a.out}/{name}/meta/crosshand_cells.json", "w"), indent=1)


if __name__ == "__main__":
    main()
