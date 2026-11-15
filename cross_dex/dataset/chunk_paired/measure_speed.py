"""Finger-joint speed of task demos vs retargeted human data, per benchmark.

Speed of one frame = mean over the hand's finger joints of |dq|/dt (rad/s).
Used to pick the time-scale (resampling) of human trajectories when chunks are
pre-built per benchmark (DexVerse 60 Hz, DexJoCo-X 20 Hz).

  python -m cross_dex.dataset.chunk_paired.measure_speed --out speed.json
"""
import argparse, glob, json, os
import numpy as np

D = "/workspace/datasets/cross_dex/chunk_paired_dataset"
DJ = "/workspace/datasets/dexjoco/training_v1/bundle/episodes"
QS = [10, 25, 50, 75, 90, 95]
MOVE = 0.05  # rad/s, below = hand still


def stats(s):
    s = np.asarray(s, np.float64)
    m = s[s > MOVE]
    if len(m) == 0:
        return dict(n=int(len(s)), moving_frac=0.0)
    return dict(n=int(len(s)), moving_frac=round(float(len(m) / len(s)), 3),
                moving_q={str(q): round(float(np.percentile(m, q)), 4) for q in QS},
                all_mean=round(float(s.mean()), 4))


def speeds(q, dt):  # q [T, n]
    return np.abs(np.diff(q, axis=0)).mean(1) / dt


def dexverse_teleop(paths, hands):
    out = {}
    for h in hands:
        s = []
        for p in paths:
            z = np.load(p, allow_pickle=True)
            if f"q_{h}" not in z.files:
                continue
            q = z[f"q_{h}"]  # [N,16,n] chunks at 60 Hz
            s.append((np.abs(np.diff(q, axis=1)).mean(2) * 60.0).ravel())
        if s:
            out[h] = stats(np.concatenate(s))
    return out


def dexjoco(hands):
    out = {}
    for h in hands:
        per_task = {}
        for task_dir in sorted(glob.glob(f"{DJ}/*")):
            task = os.path.basename(task_dir)
            s = []
            for f in sorted(glob.glob(f"{task_dir}/{h}/scene_*/native.npz")):
                z = np.load(f)
                a, m = z["action"], z["action_mask"]
                for side in range(2):
                    if not m[0, side, 0]:
                        continue
                    n = int(m[0, side].sum()) - 6
                    s.append(speeds(a[:, side, 6:6 + n], 0.05))
            if s:
                per_task[task] = stats(np.concatenate(s))
        if per_task:
            out[h] = per_task
    return out


def human(src, run, hands, limit):
    files = sorted(glob.glob(f"{D}/{src}/retarget/{run}/*.npz"))[:limit]
    out = {h: [] for h in hands}
    for f in files:
        z = np.load(f, allow_pickle=True)
        t = z["t"]
        dt = np.diff(t)
        if len(t) < 3 or not np.all(dt > 0):
            continue
        for h in hands:
            if f"{h}_q" in z.files:
                q = z[f"{h}_q"]
                out[h].append(np.abs(np.diff(q, axis=0)).mean(1) / dt)
    return {h: stats(np.concatenate(v)) for h, v in out.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", default="shadow,inspire,sharpa,wuji")
    ap.add_argument("--dj_hands", default="allegro,inspire,leap,sharpa_wave,wuji,linkerhand_l20,xhand")
    ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    hands = a.hands.split(",")
    tel = sorted(glob.glob(f"{D}/teleop/tel_*.npz"))
    r = dict(move_threshold_rad_s=MOVE, unit="rad/s, mean |dq|/dt over finger joints",
             dexverse_teleop=dexverse_teleop(tel, hands),
             dexjoco_demos=dexjoco(a.dj_hands.split(",")),
             human_dexcanvas=human("dexcanvas", "feix400_v2", hands, a.limit),
             human_hograspnet=human("hograspnet", "feix400_v2", hands, a.limit))
    json.dump(r, open(a.out, "w"), indent=1)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
