"""Latent replay through the LAD space (check before policy training: demo fingers -> LAD embedding -> decoder -> simulator).
dexverse_xlvla.replay unchanged, with its hand encoder / decoder replaced by the frozen LAD ones (pi05_space.LadSpace).

  PYTHONPATH=<DexLatent>:<robot_clip> python -m dexverse_lad.replay_lad --lad <model.pt> --task <task> --src <hand> --dst <hand>
         --mode latent --ckpt <xlvla latent.pt (hand models)> --wrist <calib.json> --out <dir> [--episodes 0-4]
then: python dexverse_xlvla/run_eval.py --demos <printed demos> --ckpt <out>/policy --output_dir <out>/eval --episodes ... --headless
"""
import sys

import dexverse_xlvla.replay as R

from dexverse_lad.pi05_space import LadSpace

SPACE = None


def main():
    global SPACE
    i = sys.argv.index("--lad")
    lad = sys.argv[i + 1]
    del sys.argv[i:i + 2]
    load = R.load

    def load_lad(ckpt):
        global SPACE
        tr = load(ckpt)
        SPACE = LadSpace(lad, tr.hand_models)
        return tr
    R.load = load_lad
    R.enc = lambda tr, h, q: SPACE.encode(h, q)
    R.dec = lambda tr, h, z: SPACE.decode(h, z)
    R.main()


if __name__ == "__main__":
    main()
