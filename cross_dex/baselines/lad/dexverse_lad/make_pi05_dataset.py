"""LAD-latent dataset for pi0.5 (same policy, data, frames and wrist as the XL-VLA latent run; only the finger space differs):

  observation.state / action = [canonical wrist (9) | LAD embedding of the finger joints (16)] = 25 dims

Built with dexverse_xlvla.make_dataset unchanged (copy of videos/meta, per-cell sets, merge into <out>/all) by replacing
its hand encoder with the frozen LAD encoder (dexverse_lad.pi05_space.LadSpace). The XL-VLA checkpoint is loaded only for
the hand models (joint order, joint ranges). openpi pads 25 -> the model's action_dim 32, so the pi0.5 base action
projections are kept (as in the joint-space run).

  PYTHONPATH=<DexLatent>:<robot_clip> python -m dexverse_lad.make_pi05_dataset --lad <lad_cam.pt> --ckpt <xlvla latent.pt>
         --wrist <calib folder> --task <t1,...> --out <root>
"""
import json
import sys

import dexverse_xlvla.make_dataset as MD

from dexverse_lad.pi05_space import LadSpace

LAD = None
SPACE = None


def load_lad(ckpt):
    global SPACE
    tr = _load(ckpt)
    SPACE = LadSpace(LAD, tr.hand_models)
    tr.config.latent_dim_hand = SPACE.dim        # make_dataset sizes the vector as 9 + latent_dim_hand
    return tr


def enc_lad(tr, hand, fingers):
    return SPACE.encode(hand, fingers)


def main():
    global LAD, _load
    i = sys.argv.index("--lad")
    LAD = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    _load = MD.load
    MD.load, MD.enc, MD.ROBOT_TYPE = load_lad, enc_lad, "lad_latent_dexverse"
    MD.main()
    out = sys.argv[sys.argv.index("--out") + 1]
    info = json.load(open(f"{out}/all/meta/info.json"))
    print(f"[lad] {out}/all: state/action = wrist 9 + LAD z {SPACE.dim} ({info['total_episodes']} episodes)", flush=True)


if __name__ == "__main__":
    main()
