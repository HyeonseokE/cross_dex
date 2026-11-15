"""DexJoCo-X training demos -> H-step chunks of right-hand actuator targets (20 Hz, actuator order), in the
teleop_chunks.py format (q_<hand> [N,H,n], task_<hand>, ep_<hand>) -> usable as --teleop / --teleop_test of train.py.

Source: training_v1/bundle/episodes/<task>/<hand>/scene_XXX/native.npz, action [T, 2(right, left), 6 + n_act]
(world delta wrist 6 + native position-actuator targets; simulator/dexjoco/benchmark6/dataset.py).

  python -m cross_dex.latent.chunk_latent.dexjoco_demo_chunks --episodes 0-9 --out dj_ep0-9.npz
"""
import argparse
import glob
import os

import numpy as np

ROOT = "/workspace/datasets/dexjoco/training_v1/bundle/episodes"
# training hand name -> DexJoCo folder
HANDS = {"allegro": "allegro", "inspire": "inspire", "leap": "leap", "sharpa": "sharpa_wave", "wuji": "wuji",
         "linkerhand": "linkerhand_l20", "xhand": "xhand"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--hands", default=",".join(HANDS))
    ap.add_argument("--episodes", default="0-9", help="scene index range lo-hi (inclusive)")
    ap.add_argument("--H", type=int, default=16)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lo, hi = map(int, a.episodes.split("-"))
    tasks = sorted(os.listdir(a.root)) if a.tasks == "all" else a.tasks.split(",")
    out = {}
    for h in a.hands.split(","):
        Q, TK, EP = [], [], []
        for ti, task in enumerate(tasks):
            for f in sorted(glob.glob(f"{a.root}/{task}/{HANDS[h]}/scene_*/native.npz")):
                ep = int(f.split("scene_")[1][:3])
                if not lo <= ep <= hi:
                    continue
                z = np.load(f)
                m = z["action_mask"][0, 0]           # right row
                n = int(m.sum()) - 6
                q = z["action"][:, 0, 6:6 + n].astype(np.float32)
                for t0 in range(0, len(q) - a.H + 1, a.stride):
                    Q.append(q[t0:t0 + a.H]); TK.append(ti); EP.append(ep)
        out[f"q_{h}"] = np.asarray(Q, np.float32)
        out[f"task_{h}"] = np.asarray(TK, np.int16)
        out[f"ep_{h}"] = np.asarray(EP, np.int16)
        print(h, out[f"q_{h}"].shape, flush=True)
    np.savez(a.out, tasks=np.asarray(tasks), **out)


if __name__ == "__main__":
    main()
