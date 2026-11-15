"""Joint-space dataset for the plain pi0.5 baseline (XL-VLA paper Sec. 4: "a strong pi0 baseline trained on the same
multi-hand, multi-task dataset as a single shared policy ... pi0 can nominally accommodate different embodiments by varying
sequence lengths"), built from the same teleop cells, frames, wrist representation and finger values as the latent
dataset (dexverse_xlvla.make_dataset), only without the hand encoder:

  observation.state / action = [canonical wrist (9) | finger joints of the hand, zero-padded to --pad (22)] = 31 dims

Finger values are exactly what the latent run's hand encoder receives (hand_model.angles_to_normalized of the finger
joints, joint range -> [-1, 1]), so evaluation reuses the latent run's decode path with the decoder replaced by
"take the first n_h values". Zero padding = varying action length (openpi then pads 31 -> model action_dim 32).
No hand id is given to the policy (as in the paper's baseline). Normalisation is openpi's global per-dimension stats.

Everything else (copy of videos/meta, per-cell sets, merge into <out>/all, cells json) is make_dataset's code, used
unchanged by replacing its encoder call.

  python -m dexverse_xlvla.pi05.make_joint_dataset --ckpt <latent.pt (hand models only)> --wrist <calib folder>
         --task <t1,...> --out <root> [--pad 22]
"""
import json
import sys

import numpy as np
import torch

import dexverse_xlvla.make_dataset as MD

PAD = 22


def enc_joint(tr, hand, fingers):
    f = fingers.numpy()
    if f.shape[1] > PAD:
        raise SystemExit(f"{hand}: {f.shape[1]} finger joints > --pad {PAD}")
    out = np.zeros((len(f), PAD), np.float32)
    out[:, : f.shape[1]] = f
    return torch.from_numpy(out)


def load_joint(ckpt):
    tr = _load(ckpt)
    tr.config.latent_dim_hand = PAD        # make_dataset sizes the vector as 9 + latent_dim_hand
    return tr


def main():
    global PAD, _load
    if "--pad" in sys.argv:
        i = sys.argv.index("--pad")
        PAD = int(sys.argv[i + 1])
        del sys.argv[i:i + 2]
    _load = MD.load
    MD.load, MD.enc, MD.ROBOT_TYPE = load_joint, enc_joint, "xlvla_joint_dexverse"
    MD.main()
    out = sys.argv[sys.argv.index("--out") + 1]
    p = f"{out}/all/meta/info.json"
    info = json.load(open(p))
    for k in ("observation.state", "action"):
        info["features"][k]["names"] = [n.replace("z_", "finger_") for n in info["features"][k]["names"]]
    json.dump(info, open(p, "w"), indent=4)
    print(f"[joint] {out}/all: state/action = wrist 9 + fingers padded to {PAD}", flush=True)


if __name__ == "__main__":
    main()
