#!/usr/bin/env python
"""Bimanual version of measure_wrist.py: read the action/state semantics of a floating bimanual DexVerse robot from the
simulator (main env, Isaac Sim; DexVerse is only called) and measure BOTH virtual wrists, so that
dexverse_xlvla.bimanual can map [right | left] x [wrist | fingers] between the env action and the policy space.

Per robot (task env of the given pickle, or --robot_types on the env of the first pickle):
  * action map, read from the env's action term objects: per action dim term name, term class, joint name, scale, offset
    (joint target = offset + scale * action); checked by pushing a random action through the env's action manager and
    reading the articulation's joint position targets (action_target_check). Also the articulation's joint names and
    the evaluator's state order (eval_lerobot.state_order: action-term joints first, then the rest), checked here to
    start with the action joints, so that state[i] <-> action[i].
  * sides: the dims of each hand found by joint-name prefix (right_/rh_, left_/lh_); the wrist = the 6 dims of the
    side's action terms whose name contains "wrist" (must be absolute JointPositionAction terms); fingers = the side's
    other dims.
  * per side, the wrist kinematics exactly as measure_wrist.py (root link of that side's hand URDF, found in the
    articulation as <link> / <prefix><link>; screws of the 6 wrist joints and the root pose M at wrist joints 0;
    random-configuration check of the product-of-exponentials model against the simulator).
  * per side, fk_samples: the recorded start joint state of a few episodes with random wrist offsets -> world pose of
    the root body and world positions of the fingertip bodies found in the articulation. Checked against the hand
    model's FK in the DexLatent env (dexverse_xlvla.verify_bimanual), which tests the root frame, the canonical frame
    and the finger joint name mapping at once.

  source /workspace/codes/_setup/dexverse_env.sh DexVerse
  python measure_wrist_bimanual.py --pkls <bimanual pkl> [...] --out wrist_calib_<task>.json --headless
         [--robot_types floating_wuji_bimanual,...]   (hands without a pickle: env of the first pickle, default reset state)
  An existing --out is updated (one robot per process is safer: a second env in the same Isaac process hung once).
"""
import argparse
import json
import os
import pickle
import xml.etree.ElementTree as et

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--pkls", nargs="+", required=True)
parser.add_argument("--robot_types", default="", help="comma list: measure these robots (not the pickles' own) on the env of the first pickle")
parser.add_argument("--out", required=True)
parser.add_argument("--fk_samples", type=int, default=6)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import numpy as np  # noqa: E402
from scipy.linalg import expm, logm  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

ASSET_ROOT = os.environ.get("DEXVERSE_ROBOT_AGENTS", "/workspace/assets/dexverse/robot_agents")
EXTRA_URDF_ROOT = os.environ.get("XLVLA_EXTRA_URDF", "/workspace/assets/dexverse/xlvla_extra")
# same files / roots / fingertips as dexverse_xlvla.hands HANDS / LEFT_HANDS (kept here: no torch-side imports in Isaac)
URDF = {("shadow", "right"): "shadow/retarget/floating_shadow_right.urdf",
        ("shadow", "left"): "shadow/retarget/floating_shadow_left.urdf",
        ("allegro", "right"): "allegro/retarget/allegro_hand_right/allegro_hand_right.urdf",
        ("allegro", "left"): "allegro/retarget/allegro_hand_left/allegro_hand_left.urdf",
        ("inspire", "right"): "inspire/retarget/inspire_hand_right/inspire_hand_right.urdf",
        ("inspire", "left"): os.path.join(EXTRA_URDF_ROOT, "inspire_hand_left.urdf"),
        ("leap", "right"): "leap/retarget/leap_hand_right_fixed.urdf",
        ("leap", "left"): "leap/retarget/leap_hand_left_fixed.urdf",
        ("sharpa", "right"): "sharpa/retarget/sharpa_wave_right/sharpa_wave_right.urdf",
        ("sharpa", "left"): "sharpa/retarget/sharpa_wave_left/sharpa_wave_left.urdf",
        ("wuji", "right"): "wuji/retarget/wuji_hand_right/wuji_hand_right.urdf",
        ("wuji", "left"): "wuji/retarget/wuji_hand_left/wuji_hand_left.urdf"}
ROOT = {("allegro", "right"): "base_link", ("allegro", "left"): "base_link", ("inspire", "right"): "base",
        ("inspire", "left"): "base", ("leap", "right"): "base_link", ("leap", "left"): "base_link",
        ("sharpa", "right"): "right_hand_C_MC", ("sharpa", "left"): "left_hand_C_MC",
        ("wuji", "right"): "right_palm_link", ("wuji", "left"): "left_palm_link"}
TIPS = {"shadow": ("thtip", "fftip", "mftip", "rftip", "lftip"),
        "allegro": ("link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip"),
        "inspire": ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"),
        "leap": ("thumb_tip_head", "index_tip_head", "middle_tip_head", "ring_tip_head"),
        "sharpa": tuple(f"{{s}}_{f}_fingertip" for f in ("thumb", "index", "middle", "ring", "pinky")),
        "wuji": tuple(f"{{s}}_finger{i}_tip_link" for i in range(1, 6))}
PREFIX = {"right": ("right_", "rh_", "r_"), "left": ("left_", "lh_", "l_")}


def side_of(name: str):
    for s, ps in PREFIX.items():
        if name.startswith(ps):
            return s
    return None


def urdf_path(hand, side):
    p = URDF[(hand, side)]
    return p if os.path.isabs(p) else os.path.join(ASSET_ROOT, p)


def root_link(hand, side):
    if hand != "shadow":
        return ROOT[(hand, side)]
    r = et.parse(urdf_path(hand, side)).getroot()
    return next(j.find("child").get("link") for j in r.findall("joint") if j.get("name") == "z_rotation_joint")


def fixed_chain(hand, side):
    """[(link, T_root_link)] along the hand URDF's fixed joints from the root (root itself first)."""
    r = et.parse(urdf_path(hand, side)).getroot()
    out, frontier = [], [(root_link(hand, side), np.eye(4))]
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


def sim_body(link, side, bodies):
    """Articulation body of a hand-URDF link: prefixed name first (both hands share one articulation), then with a
    _<side> suffix. Which body it is, is checked by the screw test and the fingertip FK check."""
    for base in (link, link.replace(".", "_")):
        own = base.startswith(PREFIX[side])                 # already side-named (sharpa, wuji)
        cands = [base] if own else [p + base for p in PREFIX[side]] + ([base] if side_of(base) is None else [])
        cands += [f"{c}_{side}" for c in cands]             # e.g. LEAP bimanual: left_palm_lower_left
        for c in cands:
            if c in bodies:
                return c
    return None


def make_env(payload, robot_type):
    import gymnasium as gym

    import dexverse.tasks  # noqa: F401
    from dexverse.tasks.utils import parse_env_cfg

    task = payload["env_name"]
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


def twist(m):
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


def state_order(base):   # same as the common evaluator (eval_lerobot.state_order), re-implemented read-only
    robot = base.scene["robot"]
    ids = []
    for name in base.action_manager.active_terms:
        term = base.action_manager.get_term(name)
        tid = term._joint_ids
        tid = list(range(robot.num_joints))[tid] if isinstance(tid, slice) else list(tid)
        ids += [int(i) for i in tid[: int(term.action_dim)]]
    return ids + [i for i in range(robot.num_joints) if i not in set(ids)]


def measure(payload, robot_type, use_recorded_starts):
    import torch

    hand = robot_type.removeprefix("floating_").removesuffix("_bimanual")
    env = make_env(payload, robot_type)
    env.reset()
    base = env.unwrapped
    robot, sim = base.scene["robot"], base.sim
    amap = []
    for tname in base.action_manager.active_terms:
        term = base.action_manager.get_term(tname)
        kind = type(term).__name__
        jn = list(getattr(term, "_joint_names", [None] * term.action_dim))
        if len(jn) != term.action_dim:
            raise RuntimeError(f"{robot_type}: term {tname} has {len(jn)} joints for {term.action_dim} dims")
        sc, of = getattr(term, "_scale", None), getattr(term, "_offset", None)
        for i, j in enumerate(jn):
            s = sc if sc is None or isinstance(sc, float) else float(sc[0, i])
            o = of if of is None or isinstance(of, float) else float(of[0, i])
            amap.append({"term": tname, "kind": kind, "joint": j, "scale": s, "offset": o,
                         "use_default_offset": bool(getattr(term.cfg, "use_default_offset", False)),
                         "side": side_of(j)})
    order = state_order(base)
    state_names = [robot.joint_names[i] for i in order]
    if state_names[: len(amap)] != [m["joint"] for m in amap]:
        raise RuntimeError(f"{robot_type}: evaluator state order != action order")
    if any(m["side"] is None for m in amap):
        raise RuntimeError(f"{robot_type}: action joints without a side prefix: {[m['joint'] for m in amap if not m['side']]}")
    # action -> joint target as the env applies it: random action through the action manager, then the articulation's
    # joint position targets (= what the drives track) vs offset + scale * action of the action map
    rng0 = np.random.default_rng(1)
    a_rand = torch.tensor(rng0.uniform(-0.3, 0.3, (1, len(amap))), dtype=torch.float32, device=base.device)
    base.action_manager.process_action(a_rand)
    base.action_manager.apply_action()
    tgt = robot.data.joint_pos_target[0].cpu().numpy()
    exp = np.array([(m["offset"] or 0.0) + (1.0 if m["scale"] is None else m["scale"]) * float(a_rand[0, i])
                    for i, m in enumerate(amap)])
    got = np.array([tgt[robot.joint_names.index(m["joint"])] for m in amap])
    target_check = {"max_abs_err": float(np.abs(got - exp).max()),
                    "max_abs_err_if_no_offset": float(np.abs(got - (exp - np.array([m["offset"] or 0.0 for m in amap]))).max())}
    print(f"[wrist-bi] {robot_type}: action -> joint target check {target_check}", flush=True)
    env.reset()
    view = robot.root_physx_view
    q0 = robot.data.joint_pos.clone()
    lim = robot.data.joint_pos_limits if hasattr(robot.data, "joint_pos_limits") else robot.data.joint_limits
    lo, hi = lim[0, :, 0].cpu().numpy(), lim[0, :, 1].cpu().numpy()
    out = {"robot_type": robot_type, "task": payload["env_name"], "action_map": amap,
           "action_joints": [m["joint"] for m in amap], "articulation_joints": list(robot.joint_names),
           "state_order_joints": state_names, "body_names": list(robot.body_names),
           "robot_root_pose": robot.data.root_pose_w[0].cpu().numpy().tolist(), "action_target_check": target_check,
           "sides": {}}

    def pose_at(bid, jid, wq, qbase=None):
        q = (q0 if qbase is None else qbase).clone()
        for i, x in zip(jid, wq):
            q[0, i] = float(x)
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        sim.forward()
        return pose_mat(view.get_link_transforms()[0, bid].detach().cpu().numpy().astype(np.float64))

    for side in ("right", "left"):
        dims = [i for i, m in enumerate(amap) if m["side"] == side]
        wdims = [i for i in dims if "wrist" in amap[i]["term"]]
        if len(wdims) != 6 or any(amap[i]["kind"] != "JointPositionAction" for i in wdims):
            raise RuntimeError(f"{robot_type} {side}: wrist dims must be 6 absolute JointPositionAction dims: "
                               f"{[amap[i] for i in wdims]}")
        fdims = [i for i in dims if i not in wdims]
        if any(amap[i]["kind"] != "JointPositionAction" for i in fdims):
            raise RuntimeError(f"{robot_type} {side}: finger terms are not JointPositionAction")
        action_joints = [amap[i]["joint"] for i in wdims]
        names = sorted(action_joints, key=robot.joint_names.index)           # kinematic-chain order
        jid = [robot.joint_names.index(n) for n in names]
        zero = np.zeros(6)
        for link, t_root_link in fixed_chain(hand, side):
            body = sim_body(link, side, robot.body_names)
            if body is None:
                continue
            bid = robot.body_names.index(body)
            m0 = pose_at(bid, jid, zero)
            screws = []
            for k in range(6):
                eps = 0.05 if k >= 3 else 0.02
                wq = zero.copy()
                wq[k] = eps
                dm = pose_at(bid, jid, wq) @ np.linalg.inv(m0)
                if not np.isfinite(dm).all():
                    raise RuntimeError(f"{robot_type} {side}: non-finite pose of {body}")
                w, v = twist(np.real(logm(dm)) / eps)
                screws.append((w.tolist(), v.tolist()))
            print(f"[wrist-bi] {robot_type} {side}: candidate {link} -> {body}, smallest screw "
                  f"{min(np.linalg.norm(np.r_[w, v]) for w, v in screws):.3f}", flush=True)
            if min(np.linalg.norm(np.r_[w, v]) for w, v in screws) > 0.5:
                break
        else:
            raise RuntimeError(f"{robot_type} {side}: no body of the hand root's fixed chain "
                               f"{[l for l, _ in fixed_chain(hand, side)]} moves with the wrist; bodies {robot.body_names}")
        m0 = m0 @ np.linalg.inv(t_root_link)
        rng = np.random.default_rng(0)
        errs = []
        for _ in range(20):
            wq = np.concatenate([rng.uniform(-0.2, 0.2, 3), rng.uniform(-0.6, 0.6, 3)])
            t_sim, t_poe = pose_at(bid, jid, wq) @ np.linalg.inv(t_root_link), poe(screws, m0, wq)
            errs.append((float(np.linalg.norm(t_sim[:3, 3] - t_poe[:3, 3])),
                         float(np.degrees(Rotation.from_matrix(t_sim[:3, :3].T @ t_poe[:3, :3]).magnitude()))))
        # fingertip bodies + FK samples (recorded starts: physically consistent finger joints, incl. passive ones)
        tip_links = [t.format(s=side) for t in TIPS[hand]]
        tip_bodies = [sim_body(t, side, robot.body_names) for t in tip_links]
        starts = []
        if use_recorded_starts:
            for e in payload["episodes"][: args.fk_samples]:
                jp = np.asarray(e["initial_state"]["articulation"]["robot"]["joint_position"], np.float64)
                if jp.shape[-1] == robot.num_joints:
                    starts.append(torch.tensor(jp.reshape(1, -1), dtype=q0.dtype, device=q0.device))
        starts = starts or [q0.clone() for _ in range(args.fk_samples)]
        samples = []
        for k, qs in enumerate(starts):
            wq = np.array([qs[0, i].item() for i in jid]) + np.concatenate([rng.uniform(-0.05, 0.05, 3),
                                                                            rng.uniform(-0.3, 0.3, 3)])
            pose_at(bid, jid, wq, qs)                                          # writes the state
            tf = view.get_link_transforms()[0].detach().cpu().numpy().astype(np.float64)
            qn = robot.data.joint_pos[0].cpu().numpy()                           # as written (sim.forward)
            samples.append({"joint_pos": {n: float(qn[i]) for i, n in enumerate(robot.joint_names)},
                            "root_body_pose": pose_mat(tf[bid]).tolist(),
                            "tips_world": {b: tf[robot.body_names.index(b), :3].tolist() for b in tip_bodies if b}})
        # random finger poses (action finger joints uniform in the sim limits; passive joints stay as reset, so tips
        # behind passive / mimic joints are expected to differ): tests the finger joint name mapping with bent fingers
        samples_rand = []
        for k in range(args.fk_samples):
            qs = q0.clone()
            for i in fdims:
                j = robot.joint_names.index(amap[i]["joint"])
                qs[0, j] = float(rng.uniform(lo[j], hi[j]))
            wq = np.array([qs[0, i].item() for i in jid]) + np.concatenate([rng.uniform(-0.05, 0.05, 3),
                                                                            rng.uniform(-0.3, 0.3, 3)])
            pose_at(bid, jid, wq, qs)
            tf = view.get_link_transforms()[0].detach().cpu().numpy().astype(np.float64)
            qn = robot.data.joint_pos[0].cpu().numpy()
            samples_rand.append({"joint_pos": {n: float(qn[i]) for i, n in enumerate(robot.joint_names)},
                                 "root_body_pose": pose_mat(tf[bid]).tolist(),
                                 "tips_world": {b: tf[robot.body_names.index(b), :3].tolist() for b in tip_bodies if b}})
        out["sides"][side] = {
            "hand_model": hand if side == "right" else f"{hand}_left",
            "action_dims": dims, "wrist_dims": wdims, "finger_dims": fdims,
            "finger_joints": [amap[i]["joint"] for i in fdims],
            "finger_scale": [amap[i]["scale"] for i in fdims], "finger_offset": [amap[i]["offset"] for i in fdims],
            "finger_limits": [[float(lo[robot.joint_names.index(amap[i]["joint"])]),
                               float(hi[robot.joint_names.index(amap[i]["joint"])])] for i in fdims],
            # the one-hand Wrist calibration fields (dexverse_xlvla.wrist.Wrist), for this side
            "chain_joints": names, "action_joints": action_joints,
            "action_scale": [amap[i]["scale"] for i in wdims], "action_offset": [amap[i]["offset"] for i in wdims],
            "body": body, "root_link": root_link(hand, side), "body_from_root": t_root_link.tolist(), "M": m0.tolist(),
            "screws": screws, "check_pos_err_max_m": max(e[0] for e in errs),
            "check_rot_err_max_deg": max(e[1] for e in errs),
            "tip_links": tip_links, "tip_bodies": tip_bodies, "fk_samples": samples, "fk_samples_random": samples_rand,
            "passive_joints": [n for n in robot.joint_names if side_of(n) == side and n not in
                               {m["joint"] for m in amap}]}
        print(f"[wrist-bi] {robot_type} {side}: body={body} wrist dims {wdims} fingers {len(fdims)} "
              f"poe pos_err_max={max(e[0] for e in errs)*1000:.3f} mm rot_err_max={max(e[1] for e in errs):.4f} deg "
              f"tips {sum(b is not None for b in tip_bodies)}/{len(tip_bodies)}", flush=True)
    env.close()
    return hand, out


if __name__ == "__main__":
    if os.environ.get("XLVLA_DUMP_STACK_S"):   # debugging a hang: stack of every thread every N s on stderr
        import faulthandler
        import sys
        faulthandler.dump_traceback_later(int(os.environ["XLVLA_DUMP_STACK_S"]), repeat=True, file=sys.stderr)
    try:
        res = json.load(open(args.out)) if os.path.exists(args.out) else {}   # one robot per process: entries add up
        extra = [r for r in args.robot_types.split(",") if r]       # given: only these robots, on the first pickle's env
        jobs = [(args.pkls[0], r) for r in extra] if extra else [(p, None) for p in args.pkls]
        task = None
        for p, rt in jobs:
            payload = pickle.load(open(p, "rb"))
            task = task or payload["env_name"]
            if payload["env_name"] != task:
                raise ValueError(f"one task per calibration file: {p}")
            hand, r = measure(payload, rt or payload["robot_type"], rt is None)
            r["pkl"] = p if rt is None else None
            res[hand] = r
            json.dump(res, open(args.out, "w"), indent=1)                         # partial results survive a crash
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(1)            # simulation_app.close() after an exception hung (replicator stop -> render), seen 2026-11-16
    simulation_app.close()
