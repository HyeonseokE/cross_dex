"""LAD paired action data per benchmark and hand side (2026-11-15; user decisions A/B/C = reuse / all sources / mirror).

A  pairs = the existing retargeting output (cross_dex retarget_dexcanvas.py, DexVerse teleop retargeter): every hand's
   ``<hand>_q`` of one frame comes from the same human hand pose (paper Eq. 1).
B  human poses = DexCanvas + HOGraspNet + Dexonomy (feix400 selections), each as two runs joined by trajectory id
   (``<src>/retarget/feix400_v2`` = shadow, inspire, sharpa, wuji; ``feix400_v2_x`` = allegro, leap, wuji_dj, linkerhand,
   xhand). Trajectories where a benchmark hand penetrates itself deeper than --drop_pen_mm anywhere are dropped (the rule
   used for the chunk sets, 8 mm).
C  left hand = the right-hand pairs converted joint-wise to the left model's convention (LEFT_MAP: the left URDFs are
   mirrors of the right ones up to these sign / finger-block changes; fingertip distances agree within 0-1.8 mm,
   dexverse_xlvla.check_left_hands 2026-11-15). No new retargeting.

Benchmarks: dexverse = shadow, inspire, sharpa, wuji, allegro, leap (env action finger order);
            dexjoco  = allegro, inspire, leap, sharpa, wuji(_dj), linkerhand, xhand (DexJoCo-X actuator order).

Output per (bench, side): ``<out>/<bench>_<side>_pairs.npy`` = dict {hand: float32 (N, n)} (robot_clip data format) and
``<out>/<bench>_<side>_pairs.meta.json``. Every --stride-th frame is kept (stride from --target_rows when 0).

  python -m dexverse_lad.build_pairs_bench --data /workspace/datasets/cross_dex/chunk_paired_dataset --out <dir>
"""
import argparse, glob, json, math, os

import numpy as np

SOURCES = ["dexcanvas", "hograspnet", "dexonomy"]
RUNS = ["feix400_v2", "feix400_v2_x"]
BENCH = {  # output hand -> (retarget hand, joint order into the retarget hand's finger joints or None)
    "dexverse": {h: (h, None) for h in ["shadow", "inspire", "sharpa", "wuji", "allegro", "leap"]},
    "dexjoco": {"allegro": ("allegro", None), "inspire": ("inspire", [4, 5, 0, 1, 3, 2]), "leap": ("leap", None),
                "sharpa": ("sharpa", [17, 18, 19, 20, 21, 0, 1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 16, 8, 9, 10, 11, 12]),
                "wuji": ("wuji_dj", None), "linkerhand": ("linkerhand", None), "xhand": ("xhand", None)},
}


def left_map(rt_hand: str, names: list[str], bench: str = "dexverse"):
    """(index, sign) arrays: left_q = right_q[index] * sign (name based, any joint order).

    The left convention depends on the benchmark's left-hand MODEL, not only on the hand:
      dexverse  DexVerse left URDFs (dexverse_xlvla.check_left_hands 2026-11-15, fingertip distances within 0-1.8 mm)
      dexjoco   DexJoCo-X bimanual MuJoCo models (2026-11-15). Left vs right hand compared inside MuJoCo on random poses
                (all hand bodies, one reflection): Allegro = mirror with the three abduction joints flipped (0.36 mm),
                LinkerHand = mirror with the four mcp_roll joints flipped (0.89 mm), Inspire / Sharpa / Wuji / XHand =
                mirror with the same joint values (0.05-0.58 mm). The left LEAP is NOT a mirror of the right one under any
                sign map (best 16 mm there); leap_4 and leap_13 flipped is the best sign map for the left demo actions
                (fingertips 6 mm mean, 33 mm max on the right model) -> the left LEAP pairs are approximate.
    """
    n = len(names)
    idx, sgn = np.arange(n), np.ones(n, np.float32)
    if bench == "dexjoco":
        flip = {"allegro": ["joint_0.0", "joint_4.0", "joint_8.0"], "leap": ["leap_4", "leap_13"],
                "linkerhand": [f"{f}_mcp_roll" for f in ("index", "middle", "ring", "pinky")]}.get(rt_hand, [])
        for j in flip:
            sgn[names.index(j)] = -1
        return idx, sgn
    if rt_hand == "leap":                       # abduction of the three fingers and the two thumb base joints
        for j in ("leap_0", "leap_4", "leap_8", "leap_12", "leap_13"):
            sgn[names.index(j)] = -1
    elif rt_hand == "wuji":                     # DexVerse Wuji: abduction (joint2) of fingers 2-5
        for f in range(2, 6):
            sgn[names.index(f"right_finger{f}_joint2")] = -1
    elif rt_hand == "allegro":                  # left URDF: joint_0-3 is the finger farthest from the thumb
        at = lambda k: names.index(f"joint_{k}.0")
        for k in range(4):
            idx[at(k)], idx[at(8 + k)] = at(8 + k), at(k)
        for k in (0, 4, 8):
            sgn[at(k)] = -1
    return idx, sgn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--drop_pen_mm", type=float, default=8.0)
    ap.add_argument("--stride", type=int, default=0)
    ap.add_argument("--target_rows", type=int, default=250000)
    ap.add_argument("--kf", default="", help="cross_dex build_keyframes.py dir (kf.npz: q_<hand> [N, 3, n]): every keyframe of "
                                             "every kept Dexonomy grasp is appended as one paired row (2026-11-15: use all data)")
    ap.add_argument("--sides", default="right,left", help="files to write; the left file can be made at load time instead "
                                                          "(train_cam --left) to save disk")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    metas = {r: json.load(open(f"{a.data}/{SOURCES[0]}/retarget/{r}/meta.json"))["hands"] for r in RUNS}
    run_of = {h: r for r in RUNS for h in metas[r]}
    names = {h: metas[run_of[h]][h]["finger_joint_names"] for h in run_of}
    for bench, hands in BENCH.items():
        rts = [rt for rt, _ in hands.values()]
        cols = {h: [] for h in hands}
        stat = {}
        for src in SOURCES:
            pen = {r: json.load(open(f"{a.data}/{src}/retarget/{r}_selfcollision.json"))["per_traj"] for r in RUNS}
            ids = sorted(set(pen[RUNS[0]]) & set(pen[RUNS[1]]))
            kept = dropped = 0
            for tid in ids:
                if a.drop_pen_mm and max(pen[run_of[rt]][tid][rt]["max_pen_mm"] for rt in rts) > a.drop_pen_mm:
                    dropped += 1
                    continue
                zs = {r: np.load(f"{a.data}/{src}/retarget/{r}/{tid}.npz") for r in {run_of[rt] for rt in rts}}
                q = {h: zs[run_of[rt]][f"{rt}_q"] for h, (rt, _) in hands.items()}
                T = {len(v) for v in q.values()}
                if len(T) != 1 or not all(np.isfinite(v).all() for v in q.values()):
                    dropped += 1
                    continue
                for h in hands:
                    cols[h].append(q[h].astype(np.float32))
                kept += 1
            stat[src] = {"trajectories": kept, "dropped": dropped}
            print(bench, src, stat[src], flush=True)
        full = {h: np.concatenate(v) for h, v in cols.items()}
        n_full = len(next(iter(full.values())))
        stride = a.stride or max(1, math.ceil(n_full / a.target_rows))
        n_kf = 0
        if a.kf:
            assert stride == 1, "--kf is meant for the all-data set (--stride 1)"
            z = np.load(os.path.join(a.kf, "kf.npz"))
            for h, (rt, _) in hands.items():
                k = z[f"q_{rt}"]                                   # [N, 3, n] pregrasp / grasp / squeeze
                full[h] = np.concatenate([full[h], k.reshape(-1, k.shape[-1]).astype(np.float32)])
                n_kf = k.shape[0] * k.shape[1]
            stat["dexonomy_keyframes"] = {"grasps": int(n_kf // 3), "rows": int(n_kf)}
            print(bench, "keyframes", stat["dexonomy_keyframes"], flush=True)
            n_full = len(next(iter(full.values())))
        for side in a.sides.split(","):
            out = {}
            for h, (rt, order) in hands.items():
                x = full[h][::stride]
                if side == "left":
                    idx, sgn = left_map(rt, names[rt], bench)
                    x = x[:, idx] * sgn
                out[h] = np.ascontiguousarray(x if order is None else x[:, order])
            p = f"{a.out}/{bench}_{side}_pairs.npy"
            np.save(p, out, allow_pickle=True)
            json.dump({"bench": bench, "side": side, "hands": list(out), "dims": {h: int(v.shape[1]) for h, v in out.items()},
                       "retarget_hand": {h: rt for h, (rt, _) in hands.items()},
                       "joint_names": {h: [names[rt][j] for j in (order or range(len(names[rt])))] for h, (rt, order) in hands.items()},
                       "left_conversion": "right pairs * LEFT_MAP (see module doc)" if side == "left" else None,
                       "rows_full": int(n_full), "stride": stride, "rows": int(len(next(iter(out.values())))),
                       "sources": stat, "drop_pen_mm": a.drop_pen_mm}, open(p[:-4] + ".meta.json", "w"), indent=1)
            print("wrote", p, {h: v.shape for h, v in out.items()}, flush=True)


if __name__ == "__main__":
    main()
