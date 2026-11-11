"""Retarget DexCanvas human hand trajectories to DexVerse 5-finger hands with the DexVerse TELEOP retargeter.

The finger path is the one used while recording teleop demos, unchanged:
``dexverse.devices.retargeters.simple_relative_retargeting.SimpleRelativeRetargeter`` is built from the same cfg the
teleop device uses (``SimpleRelativeRetargeterCfg(robot_type=floating_<hand>_right)``, defaults otherwise), and every
frame goes through its own ``_retarget_hand_fingers`` (OpenXR joints -> wrist canonical frame incl. the -15 deg z
heading -> DexPilot, hand YAML / patched URDF / finger scales) and ``_assign_hand_fingers`` (-> env action finger
order). Common code is only called, never modified.

What this wrapper adds (the only DexCanvas-specific part):

* joint mapping DexCanvas(SNAP/manotorch 21) -> OpenXR hand joint names (wrist + 4 per finger, the 21 joints the
  retargeter reads; metacarpals of the long fingers and the palm are not read and are filled with the wrist pose),
* the OpenXR wrist orientation (only its rotation is used by the finger path): ``--wrist_frame mano`` (default) =
  DexCanvas MANO root rotation x a fixed MANO->OpenXR axis map; ``keypoints`` = built from the MCPs in the OpenXR
  hand-joint convention (-Z: wrist -> middle MCP, +Y: back of the hand, X = Y x Z),
* resampling 200 Hz -> ``--fps`` (linear on positions) BEFORE retargeting, so DexPilot's warm start sees the same
  frame spacing as a teleop session at the policy rate.

Output per trajectory (npz): ``<hand>_q`` [T, n_finger] finger targets in env action order (= ``action`` finger slice
of a teleop demo of that hand), ``<hand>_err`` [T, n_vec] DexPilot vector residual |robot_vec - scaled human_vec| (m),
``human_canonical`` [T, 21, 3] (OpenXR order, retargeter canonical frame), ``t`` [T].

  source /workspace/codes/_setup/dexverse_env.sh
  cd /workspace/codes/dexverse/DexLatent
  python -m cross_dex.dataset.chunk_paired.retarget_dexcanvas --data <dexcanvas>/data --out <dir> --shards 0 --max_traj 4 --headless
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import time

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/workspace/datasets/dexverse/human_motion/dexcanvas_keypoints",
                help="npz shards from cross_dex.dataset.chunk_paired.extract_dexcanvas")
ap.add_argument("--out", required=True)
ap.add_argument("--hands", default="shadow,inspire,sharpa,wuji")
ap.add_argument("--fps", type=float, default=60.0)
ap.add_argument("--shards", default="0", help="comma list of shard indices or a:b range")
ap.add_argument("--max_traj", type=int, default=0, help="per shard, 0 = all")
ap.add_argument("--per_label", type=int, default=0,
                help="keep at most N trajectories per grasp type over all shards (0 = all); labels '01' and '001-Palmar-Pinch' are one type")
ap.add_argument("--wrist_frame", choices=["mano", "keypoints"], default="mano",
                help="mano: DexCanvas MANO root rotation (wrist_rotvec) x fixed MANO->OpenXR axes; keypoints: built from MCPs")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()
# as record_demos.py: import pinocchio before AppLauncher (Isaac Lab's build, required by dex-retargeting)
import pinocchio  # noqa: F401,E402

args.enable_pinocchio = True
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.devices.device_base import DeviceBase  # noqa: E402
from isaaclab.devices.openxr.common import HAND_JOINT_NAMES  # noqa: E402
from scipy.spatial.transform import Rotation as R, Slerp  # noqa: E402

from dexverse.devices.retargeters.simple_relative_retargeting import (  # noqa: E402
    SimpleRelativeRetargeter,
    SimpleRelativeRetargeterCfg,
)

RIGHT = DeviceBase.TrackingTarget.HAND_RIGHT

# DexCanvas joint order (DATA_SCHEMA.md)
DC = ["palm", "index_mcp", "index_pip", "index_dip", "middle_mcp", "middle_pip", "middle_dip", "ring_mcp", "ring_pip",
      "ring_dip", "pinky_mcp", "pinky_pip", "pinky_dip", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
      "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
DCI = {n: i for i, n in enumerate(DC)}
# OpenXR joint -> DexCanvas joint (positions). OpenXR joint poses sit at the proximal end of their bone.
FINGER_XR = {"thumb": ["metacarpal", "proximal", "distal", "tip"],
             "index": ["proximal", "intermediate", "distal", "tip"]}
DC_FINGER = {"thumb": ["thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip"]}
for f in ("index", "middle", "ring", "pinky"):
    DC_FINGER[f] = [f"{f}_mcp", f"{f}_pip", f"{f}_dip", f"{f}_tip"]
XR_FINGER = {"thumb": "thumb", "index": "index", "middle": "middle", "ring": "ring", "pinky": "little"}


def _xr_name_map() -> dict[str, str]:
    """OpenXR joint name -> DexCanvas joint name, for the names in Isaac Lab's HAND_JOINT_NAMES."""
    m = {}
    for name in HAND_JOINT_NAMES:
        low = name.lower()
        if "wrist" in low:
            m[name] = "palm"            # MANO root joint = wrist
            continue
        for f, xf in XR_FINGER.items():
            if low.startswith(xf):
                seg = low[len(xf):].strip("_")
                segs = FINGER_XR["thumb"] if f == "thumb" else FINGER_XR["index"]
                if seg in segs:
                    m[name] = DC_FINGER[f][segs.index(seg)]
    return m


def openxr_wrist_rotation(j: np.ndarray) -> np.ndarray:
    """[T,21,3] DexCanvas joints -> [T,3,3] wrist rotation (columns = X,Y,Z axes) in the OpenXR hand convention."""
    w = j[:, DCI["palm"]]
    fwd = j[:, DCI["middle_mcp"]] - w
    fwd /= np.linalg.norm(fwd, axis=-1, keepdims=True)
    dorsal = np.cross(j[:, DCI["pinky_mcp"]] - w, j[:, DCI["index_mcp"]] - w)   # right hand: back of the hand
    dorsal -= (dorsal * fwd).sum(-1, keepdims=True) * fwd
    dorsal /= np.linalg.norm(dorsal, axis=-1, keepdims=True)
    z = -fwd
    y = dorsal
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=-1)


# OpenXR wrist axes (columns X, Y, Z) in the MANO root frame: MANO +x = towards the fingers (= OpenXR -Z),
# MANO +z = back of the hand (= OpenXR +Y). The keypoint frame equals MANO root x this axis map rotated by ~5.6 deg
# (shard 0: per-frame spread < 1 deg, since MANO MCPs are rigid in the root frame).
MANO_TO_OPENXR = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def mano_wrist_rotation(rotvec: np.ndarray) -> np.ndarray:
    """[T,3] MANO root rotvec (world) -> [T,3,3] wrist rotation (columns = X,Y,Z) in the OpenXR hand convention."""
    return R.from_rotvec(rotvec).as_matrix() @ MANO_TO_OPENXR


def resample(t: np.ndarray, x: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    tn = np.arange(t[0], t[-1] + 1e-9, 1.0 / fps)
    flat = x.reshape(len(t), -1)
    out = np.stack([np.interp(tn, t, flat[:, k]) for k in range(flat.shape[1])], axis=-1)
    return tn, out.reshape((len(tn),) + x.shape[1:])


class TeleopFingerRetargeter:
    """The teleop retargeter of one hand, used for its finger path only."""

    def __init__(self, hand: str):
        self.hand = hand
        cfg = SimpleRelativeRetargeterCfg(robot_type=f"floating_{hand}_right", sim_device="cpu", debug_vis=False)
        self.rt = SimpleRelativeRetargeter(cfg)
        layout = self.rt._layout["hands"][RIGHT]
        self.finger_indices = np.asarray(layout["finger_indices"])
        self.finger_joint_names = list(layout.get("finger_joint_names", ()))
        self.output_dim = int(self.rt._layout["output_dim"])
        dex = self.rt._dex_retgt[RIGHT]
        self.dex = dex
        opt = dex.optimizer
        self.robot = opt.robot
        self.task_links = list(getattr(opt, "task_link_names", []))
        self.origin_links = list(getattr(opt, "origin_link_names", []))
        self.scaling = float(getattr(opt, "scaling", getattr(dex, "scaling", 1.0)) or 1.0)
        self.dex_joint_names = list(self.robot.dof_joint_names)                      # dex_q order (retarget URDF)
        self.urdf_orig = str(self.rt._get_dex_retargeting_spec()["hands"]["right"]["urdf_path"])
        patched = [p for p in getattr(self.rt, "_dex_config_temp_paths", []) if p.endswith(".urdf")]
        self.urdf_used = patched[-1] if patched else self.urdf_orig                  # limit/origin overrides applied

    def reset(self):
        self.dex.reset()

    def __call__(self, hand_data: dict) -> tuple[np.ndarray, np.ndarray]:
        dex_q = self.rt._retarget_hand_fingers(hand_data, RIGHT)            # dex/URDF DoF order
        action = np.zeros(self.output_dim, dtype=np.float32)
        self.rt._assign_hand_fingers(action, RIGHT, dex_q)                  # env action order
        return action[self.finger_indices], dex_q

    def vector_residual(self, dex_q: np.ndarray, hand_data: dict) -> np.ndarray:
        """|robot task vector - (finger-scaled) human ref vector * scaling| per DexPilot vector (m)."""
        canon = self.rt._convert_hand_to_canonical_joint_positions(hand_data, RIGHT)
        ref = self.rt._compute_dex_ref_value(self.dex, canon) * self.scaling
        self.robot.compute_forward_kinematics(np.asarray(dex_q, dtype=np.float64))   # retarget() returns all DoF
        pos = {n: self.robot.get_link_pose(self.robot.get_link_index(n))[:3, 3] for n in set(self.task_links + self.origin_links)}
        rv = np.stack([pos[t] - pos[o] for t, o in zip(self.task_links, self.origin_links)])
        self.last_links = np.stack([self.robot.get_link_pose(i)[:3, 3] for i in range(len(self.robot.link_names))])
        return np.linalg.norm(rv - ref, axis=-1)


def frame_hand_data(joints: np.ndarray, rot: np.ndarray, xr_map: dict[str, str]) -> dict:
    q = R.from_matrix(rot).as_quat()                     # xyzw
    wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=np.float32)
    data = {}
    wrist = joints[DCI["palm"]]
    for name in HAND_JOINT_NAMES:
        src = xr_map.get(name)
        pos = joints[DCI[src]] if src is not None else wrist
        data[name] = np.concatenate([pos, wxyz]).astype(np.float32)
    return data


def shard_list(spec: str, n: int) -> list[int]:
    if ":" in spec:
        a, b = spec.split(":")
        return list(range(int(a or 0), int(b or n)))
    return [int(s) for s in spec.split(",") if s != ""]


def main():
    os.makedirs(args.out, exist_ok=True)
    sim_utils.create_new_stage()
    xr_map = _xr_name_map()
    missing = [n for n in ("thumb", "index", "middle", "ring", "pinky") if not any(v.startswith(n) for v in xr_map.values())]
    assert not missing, f"OpenXR mapping misses fingers {missing}: {xr_map}"
    hands = {h: TeleopFingerRetargeter(h) for h in args.hands.split(",")}
    os.makedirs(os.path.join(args.out, "urdf"), exist_ok=True)
    for h, r in hands.items():      # the URDF the optimizer used (meshes stay next to the original URDF)
        shutil.copy(r.urdf_used, os.path.join(args.out, "urdf", f"{h}.urdf"))
    meta = {"fps": args.fps, "xr_map": xr_map, "wrist_frame": args.wrist_frame,
            "hands": {h: {"finger_joint_names": r.finger_joint_names, "dex_joint_names": r.dex_joint_names,
                          "urdf_orig": r.urdf_orig, "task_links": r.task_links,
                          "origin_links": r.origin_links, "scaling": r.scaling,
                          "link_names": list(r.robot.link_names)} for h, r in hands.items()}}
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    files = sorted(glob.glob(os.path.join(args.data, "train-*.npz")))
    per_label: dict[int, int] = {}
    for si in shard_list(args.shards, len(files)):
        z = dict(np.load(files[si]))
        off = z["offsets"]
        n = len(z["trajectory_id"]) if not args.max_traj else min(args.max_traj, len(z["trajectory_id"]))
        for i in range(n):
            tid, label = str(z["trajectory_id"][i]), str(z["action_label"][i])
            a, b = off[i], off[i + 1]
            dst = os.path.join(args.out, f"{tid}.npz")
            if args.per_label:
                key = int(label.split("-")[0]) if label[:1].isdigit() else -1
                if per_label.get(key, 0) >= args.per_label:
                    continue
                per_label[key] = per_label.get(key, 0) + 1
            if os.path.exists(dst):
                continue
            t = z["t"][a:b].astype(np.float64)
            j = z["joints"][a:b].astype(np.float64)
            tn, jn = resample(t, j, args.fps)
            if args.wrist_frame == "mano":
                # rotations are resampled by slerp, not by interpolating the rotation vectors
                rg = R.from_rotvec(z["wrist_rotvec"][a:b].astype(np.float64))
                rot = mano_wrist_rotation(Slerp(t, rg)(np.clip(tn, t[0], t[-1])).as_rotvec())
            else:
                rot = openxr_wrist_rotation(jn)
            t0 = time.time()
            out = {"t": tn.astype(np.float32), "action_label": np.array(label)}
            canon = []
            for h, r in hands.items():
                r.reset()
                qs, errs, links, dqs = [], [], [], []
                for k in range(len(tn)):
                    hd = frame_hand_data(jn[k], rot[k], xr_map)
                    q, dq = r(hd)
                    qs.append(q)
                    dqs.append(dq)
                    errs.append(r.vector_residual(dq, hd))
                    links.append(r.last_links)
                    if h == next(iter(hands)):
                        canon.append(r.rt._convert_hand_to_canonical_joint_positions(hd, RIGHT))
                out[f"{h}_q"] = np.asarray(qs, np.float32)
                out[f"{h}_dexq"] = np.asarray(dqs, np.float32)   # [T, n_dex] retarget URDF DoF order
                out[f"{h}_err"] = np.asarray(errs, np.float32)
                out[f"{h}_links"] = np.asarray(links, np.float32)   # [T, L, 3] all URDF links, retarget robot root frame
            out["human_canonical"] = np.asarray(canon, np.float32)
            np.savez_compressed(dst, **out)
            print(f"[{tid}] T={len(tn)} {time.time() - t0:.1f}s " + " ".join(
                f"{h}:err_mean={out[f'{h}_err'].mean() * 100:.2f}cm" for h in hands), flush=True)


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception:
        traceback.print_exc()   # app.close() exits the process before Python would print it
        import sys
        sys.stdout.flush(); sys.stderr.flush()
    finally:
        app.close()
