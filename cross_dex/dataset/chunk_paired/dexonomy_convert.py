"""Dexonomy (Shadow hand grasp poses, 31 Feix types) -> common keypoint format (input of the retargeting wrapper).

Each Dexonomy grasp is three static Shadow poses (pregrasp, grasp, squeeze; qpos = root pos 3 | quat wxyz 4 | 22 joints).
Trajectory (60 Hz): hold pregrasp 0.1 s -> min-jerk to grasp over D -> min-jerk to squeeze over 0.2 s -> hold 0.3 s, with D
drawn per grasp from the human closing-time distribution of DexCanvas (timing.json: 10-90 % closing time of that Feix type,
else of all types, quantiles p10/p50/p90 interpolated; a min-jerk spends 0.507 of its duration between 10 and 90 %).
Keypoints = forward kinematics of the Dexonomy MJCF (hand/right.xml, hand frame; the root pose only moves the whole
hand and is not used by the finger retargeting), mapped to handframe.DC:
  palm = rh_palm origin (wrist), long fingers mcp/pip/dip = <f>proximal / <f>middle / <f>distal origins, tip = distal +
  0.026 m along its z; thumb cmc = rh_thbase, mcp = rh_thhub, ip = rh_thdistal, tip = thdistal + 0.0275 m along z.
Wrist rotation = handframe.root_rotvec_from_keypoints (same convention as DexCanvas / HOGraspNet).

Selection: up to --per_feix grasps per type, one grasp per file, files shuffled (seed) so objects / scales vary.
Output <out>/D<feix>.npz with the common keys (trajectory_id, action_label, feix, offsets, t, joints, wrist_rotvec,
mano_shape) plus src_file / src_index.

  PYTHONPATH=/workspace/codes/cross_dex /workspace/envs/dexverse/openpi/bin/python -m cross_dex.dataset.chunk_paired.dexonomy_convert
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re

import mujoco
import numpy as np

from cross_dex.dataset.chunk_paired import DATA_ROOT, save_manifest
from cross_dex.dataset.chunk_paired.handframe import DCI, root_rotvec_from_keypoints

MINJERK_10_90 = 0.507
TIP = {"f": 0.026, "th": 0.0275}


def min_jerk(s):
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


class ShadowFK:
    def __init__(self, xml: str):
        self.m = mujoco.MjModel.from_xml_path(xml)
        self.d = mujoco.MjData(self.m)
        b = lambda n: mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, n)   # noqa: E731
        self.map = {"palm": (b("rh_palm"), 0.0)}
        for f, dc in (("ff", "index"), ("mf", "middle"), ("rf", "ring"), ("lf", "pinky")):
            self.map[f"{dc}_mcp"] = (b(f"rh_{f}proximal"), 0.0)
            self.map[f"{dc}_pip"] = (b(f"rh_{f}middle"), 0.0)
            self.map[f"{dc}_dip"] = (b(f"rh_{f}distal"), 0.0)
            self.map[f"{dc}_tip"] = (b(f"rh_{f}distal"), TIP["f"])
        self.map.update({"thumb_cmc": (b("rh_thbase"), 0.0), "thumb_mcp": (b("rh_thhub"), 0.0),
                         "thumb_ip": (b("rh_thdistal"), 0.0), "thumb_tip": (b("rh_thdistal"), TIP["th"])})
        assert all(i >= 0 for i, _ in self.map.values()), self.map

    def keypoints(self, joints: np.ndarray) -> np.ndarray:       # [T, 22] -> [T, 21, 3] DC order, hand frame
        out = np.zeros((len(joints), 21, 3))
        for t, q in enumerate(joints):
            self.d.qpos[:] = q
            mujoco.mj_kinematics(self.m, self.d)
            for name, (bid, off) in self.map.items():
                out[t, DCI[name]] = self.d.xpos[bid] + self.d.xmat[bid].reshape(3, 3)[:, 2] * off
        return out


def draw_duration(rng, q: list[float]) -> float:
    """Closing time (10-90 %) from piecewise-linear quantiles p10/p50/p90 -> min-jerk duration."""
    u = rng.uniform(0.1, 0.9)
    c = np.interp(u, [0.1, 0.5, 0.9], q)
    return float(c / MINJERK_10_90)


def trajectory(keys: np.ndarray, d_close: float, fps: float) -> np.ndarray:   # keys [3, 22]
    segs = [np.repeat(keys[:1], int(round(0.1 * fps)), 0)]
    for a, b, d in ((keys[0], keys[1], d_close), (keys[1], keys[2], 0.2)):
        s = min_jerk(np.linspace(0, 1, max(int(round(d * fps)), 2)))[1:]
        segs.append(a[None] + s[:, None] * (b - a)[None])
    segs.append(np.repeat(keys[2:], int(round(0.3 * fps)), 0))
    return np.concatenate(segs)


def convert_all_type(args):
    """--all: every grasp of one type, keyframes only (pregrasp, grasp, squeeze) -> shards of <= shard_size grasps."""
    tdir, root, out, shard_size = args
    fk = ShadowFK(os.path.join(root, "hand", "right.xml"))
    gdir = os.path.join(root, "Dexonomy_GRASP_shadow", "succ_collect")
    g = int(tdir.split("_")[0])
    name = re.sub(r"^\d+_", "", tdir)
    files = sorted(glob.glob(os.path.join(gdir, tdir, "*", "*", "*.npy")))
    buf = {"id": [], "kp": [], "q": [], "src": []}
    n_shard, n_total = 0, 0

    def flush():
        nonlocal n_shard
        if not buf["id"]:
            return
        kp = np.stack(buf["kp"]).astype(np.float32)                     # [N, 3, 21, 3]
        np.savez(os.path.join(out, f"D{g:02d}_{n_shard:03d}.npz"), trajectory_id=np.array(buf["id"]),
                 action_label=np.array([f"feix{g:02d}-{name}"] * len(buf["id"])), feix=np.full(len(buf["id"]), g),
                 offsets=np.arange(0, 3 * len(buf["id"]) + 1, 3, dtype=np.int64), t=np.tile(np.arange(3) / 60.0, len(buf["id"])),
                 joints=kp.reshape(-1, 21, 3), wrist_rotvec=root_rotvec_from_keypoints(kp.reshape(-1, 21, 3).astype(np.float64)).astype(np.float32),
                 mano_shape=np.zeros((len(buf["id"]), 10), np.float32), shadow_q=np.stack(buf["q"]).astype(np.float32),
                 src=np.array(buf["src"]), keyframes=np.array(True))
        n_shard += 1
        for k in buf:
            buf[k].clear()

    for f in files:
        try:
            d = np.load(f, allow_pickle=True).item()
        except Exception:   # noqa: BLE001
            continue
        rel = os.path.relpath(f, gdir)
        for k in range(len(d["grasp_qpos"])):
            keys = np.stack([d["pregrasp_qpos"][k, 7:], d["grasp_qpos"][k, 7:], d["squeeze_qpos"][k, 7:]]).astype(np.float64)
            kp = fk.keypoints(keys)
            if not np.isfinite(kp).all():
                continue
            buf["id"].append(f"D{g:02d}/{rel.replace('/', '__')[:-4]}/{k}")
            buf["kp"].append(kp)
            buf["q"].append(keys)
            buf["src"].append(f"{rel}:{k}")
            n_total += 1
            if len(buf["id"]) >= shard_size:
                flush()
    flush()
    return g, n_total, n_shard


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/datasets/dexverse/grasp_taxonomy/dexonomy")
    ap.add_argument("--timing", default=os.path.join(DATA_ROOT, "dexcanvas", "timing.json"))
    ap.add_argument("--out", default=os.path.join(DATA_ROOT, "dexonomy", "keypoints"))
    ap.add_argument("--per_feix", type=int, default=400)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all", action="store_true", help="every grasp, keyframes only (shards); trajectories are made in the robot "
                    "joint space after retargeting (build step)")
    ap.add_argument("--shard_size", type=int, default=20000)
    ap.add_argument("--workers", type=int, default=31)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.all:
        from concurrent.futures import ProcessPoolExecutor
        gdir = os.path.join(a.root, "Dexonomy_GRASP_shadow", "succ_collect")
        types = sorted(os.listdir(gdir), key=lambda s: int(s.split("_")[0]))
        stats = {}
        with ProcessPoolExecutor(a.workers) as ex:
            for g, n, ns in ex.map(convert_all_type, [(t, a.root, a.out, a.shard_size) for t in types]):
                stats[g] = {"grasps": n, "shards": ns}
                print(g, stats[g], flush=True)
        json.dump(stats, open(os.path.join(os.path.dirname(a.out.rstrip("/")), "convert_all_stats.json"), "w"), indent=1)
        save_manifest(a.out, a, per_feix=stats, total=sum(v["grasps"] for v in stats.values()))
        return
    fk = ShadowFK(os.path.join(a.root, "hand", "right.xml"))
    timing = json.load(open(a.timing))["close"]
    rng = np.random.default_rng(a.seed)
    gdir = os.path.join(a.root, "Dexonomy_GRASP_shadow", "succ_collect")
    stats = {}
    for tdir in sorted(os.listdir(gdir), key=lambda s: int(s.split("_")[0])):
        g = int(tdir.split("_")[0])
        name = re.sub(r"^\d+_", "", tdir)
        files = sorted(glob.glob(os.path.join(gdir, tdir, "*", "*", "*.npy")))
        rng.shuffle(files)
        q = timing.get(str(g), timing["all"])["p10_p50_p90"]
        ids, ts, js, rvs, srcf, srci = [], [], [], [], [], []
        for f in files:
            if len(ids) >= a.per_feix:
                break
            d = np.load(f, allow_pickle=True).item()
            k = int(rng.integers(len(d["grasp_qpos"])))
            keys = np.stack([d["pregrasp_qpos"][k, 7:], d["grasp_qpos"][k, 7:], d["squeeze_qpos"][k, 7:]]).astype(np.float64)
            traj = trajectory(keys, draw_duration(rng, q), a.fps)
            kp = fk.keypoints(traj)
            if not np.isfinite(kp).all():
                continue
            ids.append(f"D{g:02d}/{os.path.relpath(f, gdir).replace('/', '__')[:-4]}/{k}")
            ts.append(np.arange(len(traj)) / a.fps)
            js.append(kp.astype(np.float32))
            rvs.append(root_rotvec_from_keypoints(kp).astype(np.float32))
            srcf.append(os.path.relpath(f, gdir))
            srci.append(k)
        np.savez(os.path.join(a.out, f"D{g:02d}.npz"), trajectory_id=np.array(ids),
                 action_label=np.array([f"feix{g:02d}-{name}"] * len(ids)), feix=np.full(len(ids), g),
                 offsets=np.concatenate([[0], np.cumsum([len(t) for t in ts])]).astype(np.int64), t=np.concatenate(ts),
                 joints=np.concatenate(js), wrist_rotvec=np.concatenate(rvs), mano_shape=np.zeros((len(ids), 10), np.float32),
                 src_file=np.array(srcf), src_index=np.array(srci))
        stats[g] = {"selected": len(ids), "files": len(files), "mean_T_s": round(float(np.mean([t[-1] for t in ts])), 2)}
        print(g, name, stats[g], flush=True)
    json.dump(stats, open(os.path.join(os.path.dirname(a.out), "convert_stats.json"), "w"), indent=1)
    save_manifest(a.out, a, per_feix=stats)


if __name__ == "__main__":
    main()
