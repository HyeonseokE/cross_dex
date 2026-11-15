#!/usr/bin/env python
"""Measure every floating hand's virtual-wrist kinematics in the DexVerse simulator (main env, Isaac Sim; read-only use
of DexVerse). Needed because the dataset wrist pose (``ee_abs``) is a displacement in each hand's own base frame, and the
base pose / home wrist values differ by hand, so it is not comparable across hands.

For each hand (task env of the given pickle):
  * action map: the env's action terms (absolute joint-position terms; DexVerse sets their offset to the joint default
    and orders joints per hand, e.g. Shadow rotation [z, y, x]) -> per action dim: joint name, scale, offset
    (joint = offset + scale * action). Stored for every action dim; the wrist uses the first six.
  * wrist kinematics: with all wrist joints at 0, the world pose M of the hand root link (the URDF root used by
    dexverse_xlvla.hands); then each wrist joint alone by +eps -> its world screw (omega, v) by the matrix log of the pose
    change. World pose of the hand root for wrist joints q (kinematic-chain order) is T(q) = exp(S_1 q_1)...exp(S_6 q_6) M.
    If the root link is not a moving simulator body (inspire's ``base`` is the floating URDF's world link, links joined
    by fixed joints may be merged), the first body on the hand URDF's fixed-joint chain from the root that moves with
    the wrist is measured and M is moved back to the root by the URDF transform.
  A random-configuration check against the simulator is stored as well.

  source /workspace/codes/_setup/dexverse_env.sh DexVerse-develop
  python measure_wrist.py --pkls <pkl> [<pkl> ...] --out wrist_calib.json --headless
"""
import argparse
import json
import pickle
import xml.etree.ElementTree as et

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--pkls", nargs="+", required=True)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import numpy as np  # noqa: E402
from scipy.linalg import expm, logm  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

ASSET_ROOT = "/workspace/assets/dexverse/robot_agents/"
# same files / roots as dexverse_xlvla.hands (kept here so this script needs no torch-side imports)
URDF = {"shadow": "shadow/retarget/floating_shadow_right.urdf",
        "allegro": "allegro/retarget/allegro_hand_right/allegro_hand_right.urdf",
        "inspire": "inspire/retarget/inspire_hand_right/inspire_hand_right.urdf",
        "leap": "leap/retarget/leap_hand_right_fixed.urdf",
        "sharpa": "sharpa/retarget/sharpa_wave_right/sharpa_wave_right.urdf",
        "wuji": "wuji/retarget/wuji_hand_right/wuji_hand_right.urdf"}
ROOT = {"allegro": "base_link", "inspire": "base", "leap": "base_link", "sharpa": "right_hand_C_MC",
        "wuji": "right_palm_link"}


def hand_of(robot_type: str) -> str:
    return robot_type.removeprefix("floating_").removesuffix("_right")


def root_link(hand: str) -> str:
    if hand != "shadow":
        return ROOT[hand]
    r = et.parse(ASSET_ROOT + URDF["shadow"]).getroot()
    return next(j.find("child").get("link") for j in r.findall("joint") if j.get("name") == "z_rotation_joint")


def fixed_chain(hand: str) -> list:
    """[(link, T_root_link)] along the hand URDF's fixed joints from the root (root itself first)."""
    r = et.parse(ASSET_ROOT + URDF[hand]).getroot()
    out, frontier = [], [(root_link(hand), np.eye(4))]
    while frontier:
        link, t = frontier.pop(0)
        out.append((link, t))
        for j in r.findall("joint"):
            if j.get("type") == "fixed" and j.find("parent").get("link") == link:
                o = j.find("origin")
                xyz = [float(x) for x in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
                rpy = [float(x) for x in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
                tj = np.eye(4)
                tj[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
                tj[:3, 3] = xyz
                frontier.append((j.find("child").get("link"), t @ tj))
    return out


def make_env(payload):
    import gymnasium as gym

    import dexverse.tasks  # noqa: F401
    from dexverse.tasks.utils import parse_env_cfg

    task, robot_type = payload["env_name"], payload["robot_type"]
    cfg = parse_env_cfg(task, device=args.device, num_envs=1, json_path=payload.get("json_path"))
    cfg = type(cfg)(robot_type=robot_type)
    cfg.sim.device = args.device
    cfg.scene.num_envs = 1
    cfg.seed = payload["seed"]
    cfg.recorders = {}
    return gym.make(task, cfg=cfg)


def pose_mat(tf):
    m = np.eye(4)
    m[:3, :3] = Rotation.from_quat(tf[3:7]).as_matrix()   # PhysX: position + quaternion (x, y, z, w)
    m[:3, 3] = tf[:3]
    return m


def twist(m):   # 4x4 se(3) matrix -> (omega, v)
    return np.array([m[2, 1], m[0, 2], m[1, 0]]), m[:3, 3]


def se3(w, v):
    m = np.zeros((4, 4))
    m[:3, :3] = [[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]
    m[:3, 3] = v
    return m


def poe(screws, m0, q):
    t = np.eye(4)
    for (w, v), x in zip(screws, q):
        t = t @ expm(se3(np.asarray(w), np.asarray(v)) * x)
    return t @ m0


def measure(pkl):
    import torch

    payload = pickle.load(open(pkl, "rb"))
    hand = hand_of(payload["robot_type"])
    env = make_env(payload)
    env.reset()
    base = env.unwrapped
    robot, sim = base.scene["robot"], base.sim
    amap = []                                                   # every action dim: joint, scale, offset
    for tname in base.action_manager.active_terms:
        term = base.action_manager.get_term(tname)
        kind = type(term).__name__
        for i, jn in enumerate(getattr(term, "_joint_names", [None] * term.action_dim)):
            sc, of = getattr(term, "_scale", None), getattr(term, "_offset", None)
            sc = sc if sc is None or isinstance(sc, float) else float(sc[0, i])
            of = of if of is None or isinstance(of, float) else float(of[0, i])
            amap.append({"term": tname, "kind": kind, "joint": jn, "scale": sc, "offset": of})
    if any(m["kind"] != "JointPositionAction" for m in amap[:6]):
        raise TypeError(f"{hand}: wrist action terms must be absolute joint-position terms: {amap[:6]}")
    action_joints = [m["joint"] for m in amap[:6]]
    names = sorted(action_joints, key=robot.joint_names.index)  # kinematic-chain order
    jid = [robot.joint_names.index(n) for n in names]
    view = robot.root_physx_view
    q0 = robot.data.joint_pos.clone()

    def pose_at(bid, wq):
        q = q0.clone()
        for i, x in zip(jid, wq):
            q[0, i] = float(x)
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        sim.forward()
        return pose_mat(view.get_link_transforms()[0, bid].detach().cpu().numpy().astype(np.float64))

    zero = np.zeros(6)
    for link, t_root_link in fixed_chain(hand):                 # first body that moves with the wrist joints
        if link not in robot.body_names:
            continue
        bid = robot.body_names.index(link)
        m0 = pose_at(bid, zero)
        screws = []
        for k in range(6):
            eps = 0.05 if k >= 3 else 0.02
            wq = zero.copy()
            wq[k] = eps
            w, v = twist(np.real(logm(pose_at(bid, wq) @ np.linalg.inv(m0))) / eps)
            screws.append((w.tolist(), v.tolist()))
        if min(np.linalg.norm(np.r_[w, v]) for w, v in screws) > 0.5:
            break
    else:
        raise RuntimeError(f"{hand}: no body on the root's fixed chain moves with the wrist ({robot.body_names})")
    m_body = m0
    m0 = m_body @ np.linalg.inv(t_root_link)                    # body pose -> root link pose
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(20):
        wq = np.concatenate([rng.uniform(-0.2, 0.2, 3), rng.uniform(-0.6, 0.6, 3)])
        t_sim, t_poe = pose_at(bid, wq) @ np.linalg.inv(t_root_link), poe(screws, m0, wq)
        errs.append((float(np.linalg.norm(t_sim[:3, 3] - t_poe[:3, 3])),
                     float(np.degrees(Rotation.from_matrix(t_sim[:3, :3].T @ t_poe[:3, :3]).magnitude()))))
    env.close()
    root = payload["episodes"][0]["initial_state"]["articulation"]["robot"]["root_pose"][0].tolist()
    print(f"[wrist] {hand}: body={link} check pos_err_max={max(e[0] for e in errs)*1000:.3f} mm "
          f"rot_err_max={max(e[1] for e in errs):.4f} deg", flush=True)
    return hand, {"robot_type": payload["robot_type"], "chain_joints": names, "action_joints": action_joints,
                  "action_scale": [m["scale"] for m in amap[:6]], "action_offset": [m["offset"] for m in amap[:6]],
                  "action_map": amap, "body": link, "body_from_root": t_root_link.tolist(), "M": m0.tolist(),
                  "screws": screws, "robot_root_pose": root,
                  "check_pos_err_max_m": max(e[0] for e in errs), "check_rot_err_max_deg": max(e[1] for e in errs)}


if __name__ == "__main__":
    try:
        out = dict(measure(p) for p in args.pkls)
        json.dump(out, open(args.out, "w"), indent=1)
    finally:
        simulation_app.close()
