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

Input: any source in the common keypoint format (handframe: DC joint order, MANO-root ``wrist_rotvec``), e.g.
``chunk_paired_dataset/dexcanvas/keypoints`` or ``chunk_paired_dataset/hograspnet/keypoints`` (``--data``).

  source /workspace/codes/_setup/dexverse_env.sh; export PYTHONPATH=/workspace/codes/cross_dex:$PYTHONPATH
  python -m cross_dex.dataset.chunk_paired.retarget_dexcanvas --data <keypoints dir> --out <dir> --shards 0 --max_traj 4 --headless
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import time

from cross_dex.dataset.chunk_paired import DATA_ROOT, save_manifest

from isaaclab.app import AppLauncher

ap = argparse.ArgumentParser()
ap.add_argument("--data", default=os.path.join(DATA_ROOT, "dexcanvas", "keypoints"),
                help="npz shards from cross_dex.dataset.chunk_paired.extract_dexcanvas")
ap.add_argument("--out", required=True)
ap.add_argument("--hands", default="shadow,inspire,sharpa,wuji")
ap.add_argument("--fps", type=float, default=60.0)
ap.add_argument("--shards", default="0", help="comma list of shard indices or a:b range")
ap.add_argument("--max_traj", type=int, default=0, help="per shard, 0 = all")
ap.add_argument("--select", default="", help="select_trajs.py json: retarget exactly these trajectories (overrides --shards/--per_label)")
ap.add_argument("--part", default="0/1", help="i/P: this process takes every P-th selected trajectory starting at i")
ap.add_argument("--per_label", type=int, default=0,
                help="keep at most N trajectories per grasp type over all shards (0 = all); labels '01' and '001-Palmar-Pinch' are one type")
ap.add_argument("--finger_lengths", default=os.path.join(DATA_ROOT, "dexcanvas", "finger_lengths.json"),
                help="finger_lengths.py json of the source: per hand, lengthen each human finger (bones after its root) so its "
                     "wrist->tip length matches the robot finger proportions ('' = off)")
ap.add_argument("--no_palm_length", action="store_true",
                help="with --finger_lengths: keep the human palm (wrist -> knuckles) instead of matching the robot's")
ap.add_argument("--palm_hands", default="inspire",
                help="hands whose palm length is matched (2026-11-12: only inspire, whose origin link sits ~4.5 cm behind the "
                     "knuckles; on shadow / sharpa / wuji it moved the data away from the teleop demonstrations)")
ap.add_argument("--collision_project", action="store_true",
                help="move each solved dex_q to the closest pose without finger-finger penetration (collision_project.py); "
                     "raw DexPilot solution kept as <hand>_dexq_raw")
ap.add_argument("--pen_tol_mm", type=float, default=1.0, help="penetration left allowed after the projection")
ap.add_argument("--proj_max_iter", type=int, default=8,
                help="linearised iterations per frame; with the per-frame rate box more do not help (60 took 6-24 s/trajectory)")
ap.add_argument("--corr_rate_deg", type=float, default=5.0,
                help="max change of the collision correction per frame and joint (deg); deeper overlaps stay and are filtered")
ap.add_argument("--mesh_roots", default="/workspace/assets/dexverse,/workspace/assets/bench2dex",
                help="collision meshes missing next to the URDF are found by file name here")
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
from cross_dex.dataset.chunk_paired.handframe import DC, DCI, mano_wrist_rotation, openxr_wrist_rotation  # noqa: E402,F401
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

    def set_finger_lengths(self, human: dict[str, dict[str, float]], palm: bool = True):
        """Per-hand proportions of the human input (the retargeter itself is unchanged).

        DexPilot matches human wrist->tip vectors x scaling to robot origin->tip vectors (origin = the hand's palm
        origin link of the teleop config).
        * palm (wrist -> long-finger roots): the origin link is not at the same place on every hand (inspire
          hand_base_link sits ~4.5 cm behind the knuckles' wrist-equivalent: origin->index root 14.0 cm vs 9.3 cm human,
          scaling 1.0; shadow / sharpa / wuji 9.7-10.1 cm vs 10.7-11.2 cm human x scaling). The human wrist point is moved
          along the knuckles->wrist axis by palm_shift = robot_palm / scaling - human_palm (mean of index, middle, ring)
          so the palm matches the robot's in the scaled frame.
        * fingers: a robot finger that is longer relative to the hand than the human one (robot little fingers ~0.95 of
          the middle finger, human ~0.81) curls to reach a too-close target. Target human wrist->tip length per finger:
          L_f = robot_f / (scaling * c), c = mean robot / (human x scaling) ratio of index, middle, ring (overall size stays
          with the hand's own scaling); the bones after the finger root are scaled by k_f = (L_f - meta_f) / phal_f,
          meta_f = |root - shifted wrist| and phal_f measured per frame (adjust).
        """
        lo, hi = self.robot.joint_limits[:, 0], self.robot.joint_limits[:, 1]
        self.robot.compute_forward_kinematics(np.clip(np.zeros(self.robot.dof), lo, hi))
        pos = lambda n: self.robot.get_link_pose(self.robot.get_link_index(n))[:3, 3]   # noqa: E731
        origin = pos(self.origin_links[-1])
        robot = np.array([np.linalg.norm(pos(t) - pos(o)) for o, t in zip(self.origin_links[-5:], self.task_links[-5:])])
        # finger root = first joint of a tip's kinematic chain below the joints shared by all five fingers
        model, data = self.robot.model, self.robot.data
        fr = [model.frames[model.getFrameId(t)] for t in self.task_links[-5:]]
        sups = [list(model.supports[getattr(f, "parentJoint", None) or f.parent]) for f in fr]   # pinocchio 2: .parent
        c0 = 0
        while all(len(s) > c0 + 1 for s in sups) and len({s[c0] for s in sups}) == 1:
            c0 += 1
        roots = np.array([data.oMi[s[c0]].translation for s in sups])
        self.robot_palm = {f: round(float(np.linalg.norm(roots[i] - origin)), 4) for i, f in enumerate(FINGERS)}
        human_palm = np.mean([human[f]["meta"] for f in ("index", "middle", "ring")])
        robot_palm = np.mean([self.robot_palm[f] for f in ("index", "middle", "ring")])
        self.palm_shift = float(robot_palm / self.scaling - human_palm) if palm else 0.0
        tot = np.array([human[f]["total"] + (self.palm_shift if f != "thumb" else 0.0) for f in FINGERS])
        ratio = robot / (tot * self.scaling)
        c = ratio[1:4].mean()
        self.finger_target = {f: float(robot[i] / (self.scaling * c)) for i, f in enumerate(FINGERS)}
        # reported factors (median human, long fingers' meta + palm_shift); the applied ones are computed per frame
        self.finger_k = {f: float(np.clip((self.finger_target[f] - human[f]["meta"] - (self.palm_shift if f != "thumb" else 0.0))
                                          / human[f]["phal"], 0.7, 1.6)) for f in FINGERS}
        self.finger_ratio = {f: round(float(ratio[i] / c), 3) for i, f in enumerate(FINGERS)}

    def adjust(self, joints: np.ndarray) -> np.ndarray:
        """[21,3] DC joints -> wrist moved by palm_shift along knuckles->wrist, then each finger's bones after its root
        scaled so that |wrist -> tip| (along the chain) = L_f."""
        if not getattr(self, "finger_target", None):
            return joints
        j = joints.copy()
        knuck = joints[[DCI["index_mcp"], DCI["middle_mcp"], DCI["ring_mcp"]]].mean(0)
        u = joints[DCI["palm"]] - knuck
        j[DCI["palm"]] = joints[DCI["palm"]] + self.palm_shift * u / max(np.linalg.norm(u), 1e-9)
        for f, chain in CHAINS.items():
            root = joints[chain[0]]
            meta = np.linalg.norm(root - j[DCI["palm"]])
            phal = sum(np.linalg.norm(joints[chain[i + 1]] - joints[chain[i]]) for i in range(len(chain) - 1))
            k = float(np.clip((self.finger_target[f] - meta) / max(phal, 1e-6), 0.7, 1.6))
            j[chain[1:]] = root + k * (joints[chain[1:]] - root)
        return j

    def reset(self):
        self.dex.reset()

    def __call__(self, hand_data: dict) -> tuple[np.ndarray, np.ndarray]:
        dex_q = self.rt._retarget_hand_fingers(hand_data, RIGHT)            # dex/URDF DoF order
        return self.to_action(dex_q), dex_q

    def to_action(self, dex_q: np.ndarray) -> np.ndarray:
        action = np.zeros(self.output_dim, dtype=np.float32)
        self.rt._assign_hand_fingers(action, RIGHT, dex_q)                  # env action order
        return action[self.finger_indices]

    def make_projector(self, mesh_index: dict, tmp: str, tol_mm: float, rate_deg: float = 5.0, max_iter: int = 8):
        """Collision projection on the URDF the optimizer used, in the optimizer's own joints (mimic via its adaptor)."""
        from cross_dex.dataset.chunk_paired.collision_project import CollisionProjector
        from cross_dex.dataset.chunk_paired.self_collision import resolved_urdf
        opt = self.dex.optimizer
        ad = getattr(opt, "adaptor", None)
        lim = np.asarray(self.robot.joint_limits, float)
        self.projector = CollisionProjector(resolved_urdf(self.hand, self.urdf_used, self.urdf_orig, mesh_index, tmp),
                                            self.dex_joint_names, target_idx=opt.idx_pin2target,
                                            mimic_fn=(ad.forward_qpos if ad is not None else None),
                                            lo=lim[:, 0], hi=lim[:, 1], tol_mm=tol_mm, rate=np.deg2rad(rate_deg),
                                            max_iter=max_iter)

    def vector_residual(self, dex_q: np.ndarray, hand_data: dict) -> np.ndarray:
        """|robot task vector - (finger-scaled) human ref vector * scaling| per DexPilot vector (m)."""
        canon = self.rt._convert_hand_to_canonical_joint_positions(hand_data, RIGHT)
        ref = self.rt._compute_dex_ref_value(self.dex, canon) * self.scaling
        self.robot.compute_forward_kinematics(np.asarray(dex_q, dtype=np.float64))   # retarget() returns all DoF
        pos = {n: self.robot.get_link_pose(self.robot.get_link_index(n))[:3, 3] for n in set(self.task_links + self.origin_links)}
        rv = np.stack([pos[t] - pos[o] for t, o in zip(self.task_links, self.origin_links)])
        self.last_links = np.stack([self.robot.get_link_pose(i)[:3, 3] for i in range(len(self.robot.link_names))])
        return np.linalg.norm(rv - ref, axis=-1)


FINGERS = ["thumb", "index", "middle", "ring", "pinky"]   # order of the last 5 DexPilot palm->tip vectors
CHAINS = {"thumb": [DCI[n] for n in ("thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip")],
          **{f: [DCI[f"{f}_{n}"] for n in ("mcp", "pip", "dip", "tip")] for f in ("index", "middle", "ring", "pinky")}}


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
    if args.finger_lengths:
        human = json.load(open(args.finger_lengths))["fingers"]
        for r in hands.values():
            r.set_finger_lengths(human, palm=(not args.no_palm_length) and r.hand in args.palm_hands.split(","))
            print(r.hand, "palm shift (m)", round(r.palm_shift, 4), "robot palm", r.robot_palm, flush=True)
            print(r.hand, "finger length factors", {f: round(v, 3) for f, v in r.finger_k.items()},
                  "robot/human ratio (index-ring = 1)", r.finger_ratio, flush=True)
    if args.collision_project:
        import tempfile
        from cross_dex.dataset.chunk_paired.self_collision import _mesh_index
        mi, tmp = _mesh_index(args.mesh_roots.split(",")), tempfile.mkdtemp()
        for r in hands.values():
            r.make_projector(mi, tmp, args.pen_tol_mm, args.corr_rate_deg, args.proj_max_iter)
            print(r.hand, "collision projection: pairs", len(r.projector.pairs), "excluded at open pose",
                  r.projector.n_base_excluded, flush=True)
    os.makedirs(os.path.join(args.out, "urdf"), exist_ok=True)
    for h, r in hands.items():      # the URDF the optimizer used (meshes stay next to the original URDF)
        shutil.copy(r.urdf_used, os.path.join(args.out, "urdf", f"{h}.urdf"))
    meta = {"fps": args.fps, "xr_map": xr_map, "wrist_frame": args.wrist_frame, "finger_lengths": args.finger_lengths,
            "collision_project": {"pen_tol_mm": args.pen_tol_mm, "corr_rate_deg": args.corr_rate_deg, "max_iter": args.proj_max_iter} if args.collision_project else None,
            "hands": {h: {"finger_joint_names": r.finger_joint_names, "dex_joint_names": r.dex_joint_names,
                          "urdf_orig": r.urdf_orig, "task_links": r.task_links,
                          "origin_links": r.origin_links, "scaling": r.scaling,
                          "link_names": list(r.robot.link_names), "finger_k": getattr(r, "finger_k", None),
                          "palm_shift": getattr(r, "palm_shift", None), "robot_palm": getattr(r, "robot_palm", None),
                          "finger_target": getattr(r, "finger_target", None),
                          "finger_ratio": getattr(r, "finger_ratio", None)} for h, r in hands.items()}}
    tmp_meta = os.path.join(args.out, f".meta.{os.getpid()}.json")   # parallel parts write the same meta: atomic rename
    with open(tmp_meta, "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(tmp_meta, os.path.join(args.out, "meta.json"))
    files = sorted(glob.glob(os.path.join(args.data, "*.npz")))   # any source in the common keypoint format
    per_label: dict[int, int] = {}
    if args.select:
        sel = json.load(open(args.select))
        flat = [(f, i) for f, idx in sel["files"].items() for i in idx]
        pi, pn = (int(x) for x in args.part.split("/"))
        mine: dict[str, list[int]] = {}
        for f, i in flat[pi::pn]:
            mine.setdefault(f, []).append(i)
        jobs = [(os.path.join(args.data, f), idx) for f, idx in mine.items()]
        print(f"selection {args.select}: part {args.part} -> {sum(len(v) for v in mine.values())} trajectories", flush=True)
    else:
        jobs = [(files[si], None) for si in shard_list(args.shards, len(files))]
    for path, idx in jobs:
        z = dict(np.load(path))
        off = z["offsets"]
        n = len(z["trajectory_id"]) if not args.max_traj else min(args.max_traj, len(z["trajectory_id"]))
        for i in (idx if idx is not None else range(n)):
            tid, label = str(z["trajectory_id"][i]), str(z["action_label"][i])
            a, b = off[i], off[i + 1]
            dst = os.path.join(args.out, tid.replace("/", "__") + ".npz")   # HOGraspNet ids contain subject/trial/run
            if args.per_label and idx is None:
                if "feix" in z:
                    key = int(z["feix"][i])
                else:
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
            if "feix" in z:
                out["feix"] = np.array(int(z["feix"][i]))
            canon = []
            for h, r in hands.items():
                r.reset()
                if args.collision_project:
                    r.projector.reset()
                qs, errs, links, dqs, hcan, raws, pen = [], [], [], [], [], [], []
                for k in range(len(tn)):
                    hd = frame_hand_data(r.adjust(jn[k]), rot[k], xr_map)
                    q, dq = r(hd)
                    if args.collision_project:
                        raws.append(dq)
                        dq, info = r.projector.project(dq)
                        q = r.to_action(dq)
                        pen.append([info["depth_before"], info["depth_after"]])
                    qs.append(q)
                    dqs.append(dq)
                    errs.append(r.vector_residual(dq, hd))
                    links.append(r.last_links)
                    hcan.append(r.rt._convert_hand_to_canonical_joint_positions(hd, RIGHT))
                    if h == next(iter(hands)):   # unadjusted human, for reference
                        canon.append(r.rt._convert_hand_to_canonical_joint_positions(frame_hand_data(jn[k], rot[k], xr_map), RIGHT))
                out[f"{h}_q"] = np.asarray(qs, np.float32)
                out[f"{h}_dexq"] = np.asarray(dqs, np.float32)   # [T, n_dex] retarget URDF DoF order
                out[f"{h}_err"] = np.asarray(errs, np.float32)
                out[f"{h}_links"] = np.asarray(links, np.float32)   # [T, L, 3] all URDF links, retarget robot root frame
                out[f"{h}_human_canonical"] = np.asarray(hcan, np.float32)   # length-adjusted human input of this hand
                if args.collision_project:
                    out[f"{h}_dexq_raw"] = np.asarray(raws, np.float32)    # DexPilot solution before the projection
                    out[f"{h}_pen"] = np.asarray(pen, np.float32)          # [T, 2] deepest penetration before / after (m)
            out["human_canonical"] = np.asarray(canon, np.float32)
            np.savez_compressed(dst, **out)
            print(f"[{tid}] T={len(tn)} {time.time() - t0:.1f}s " + " ".join(
                f"{h}:err_mean={out[f'{h}_err'].mean() * 100:.2f}cm" + (
                    f",pen_max={out[f'{h}_pen'][:, 0].max() * 1e3:.1f}->{out[f'{h}_pen'][:, 1].max() * 1e3:.1f}mm"
                    if args.collision_project else "") for h in hands), flush=True)
    save_manifest(args.out, args, n_traj=len(glob.glob(os.path.join(args.out, "*.npz"))))


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
