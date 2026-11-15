"""Hands that are not DexVerse robot agents (DexJoCo-X models) retargeted with the same DexVerse teleop finger path.

``CustomHandRetargeter`` subclasses DexVerse's ``SimpleRelativeRetargeter`` (called, not modified) and only swaps the hand
specification: the action layout (finger joints = the DexJoCo actuator order) and the dex-retargeting spec (URDF +
DexPilot YAML generated here). Everything else (OpenXR joints -> canonical wrist frame incl. the finger heading rotation,
DexPilot optimisation, warm start) is the teleop code path.

The URDF is prepared per hand: mesh paths made absolute, missing fingertip links added (LinkerHand: DexJoCo's teacher
fingertip points), and a fixed root rotation ``cross_dex_rot`` added so that, in the optimiser frame, the fingers point
along +x, +y points from the little finger to the index side and +z out of the back of the hand (the canonical human
frame; DexVerse's own retarget URDFs carry the same kind of ``dex_rot`` root). The rotation is computed from the hand's
own zero-pose kinematics (wrist -> middle-finger base, little -> index finger base).
"""

from __future__ import annotations

import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation as R

DEXJOCO_ROOT = os.environ.get("DEXJOCO_ROOT", "/workspace/codes/dexjoco/DexJoCo-X")
_X = os.path.join(DEXJOCO_ROOT, "simulator", "dexjoco", "sim", "envs", "xmls")

# finger joint order = DexJoCo actuator order of the hand (evaluation/contracts, dexjoco_mapping.md 2026-11-12)
CUSTOM_HANDS = {
    "wuji_dj": {
        "urdf": os.path.join(DEXJOCO_ROOT, "simulator", "teleoperation", "wuji", "assets", "right", "main.urdf"),
        "wrist": "base_link",
        "tips": ["fingertip_thumb", "fingertip_index", "fingertip_middle", "fingertip_ring", "fingertip_pinky"],
        "bases": {"middle": "MCP_joint_middle", "index": "MCP_joint_index", "pinky": "MCP_joint_pinky"},
        "joints": [f"F{f}J{j}" for f in range(1, 6) for j in range(1, 5)],
        "scaling": 1.15,
    },
    "linkerhand": {
        "urdf": os.path.join(_X, "linkerhand_l20", "right", "source.urdf"),
        "wrist": "hand_base_link",
        "add_tips": {"thumb_tip": ("thumb_distal", (0.0, 0.0, 0.025)),
                     **{f"{f}_tip": (f"{f}_distal", (-0.006, 0.0, 0.02)) for f in ("index", "middle", "ring", "pinky")}},
        "tips": ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"],
        "bases": {"middle": "middle_mcp_pitch", "index": "index_mcp_pitch", "pinky": "pinky_mcp_pitch"},
        "joints": ["thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch", "thumb_mcp",
                   *[f"{f}_{j}" for f in ("index", "middle", "ring", "pinky") for j in ("mcp_roll", "mcp_pitch", "pip")]],
        "scaling": 1.2,
    },
    "xhand": {
        "urdf": os.path.join(_X, "xhand", "right", "source.urdf"),
        "wrist": "right_hand_link",
        "tips": ["right_hand_thumb_rota_tip", "right_hand_index_rota_tip", "right_hand_mid_tip", "right_hand_ring_tip",
                 "right_hand_pinky_tip"],
        "bases": {"middle": "right_hand_mid_joint1", "index": "right_hand_index_joint1", "pinky": "right_hand_pinky_joint1"},
        "joints": ["right_hand_thumb_bend_joint", "right_hand_thumb_rota_joint1", "right_hand_thumb_rota_joint2",
                   "right_hand_index_bend_joint", "right_hand_index_joint1", "right_hand_index_joint2",
                   "right_hand_mid_joint1", "right_hand_mid_joint2", "right_hand_ring_joint1", "right_hand_ring_joint2",
                   "right_hand_pinky_joint1", "right_hand_pinky_joint2"],
        "scaling": 1.1,
    },
}


def _joint_world(urdf_root, names: list[str]) -> dict[str, np.ndarray]:
    """Zero-pose positions (root frame) of the child frames of the named joints / links (fixed + movable chain)."""
    joints = {j.find("child").get("link"): j for j in urdf_root.findall("joint")}
    by_name = {j.get("name"): j for j in urdf_root.findall("joint")}

    def tf(j):
        o = j.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()]) if o is not None else np.zeros(3)
        rpy = [float(v) for v in (o.get("rpy") or "0 0 0").split()] if o is not None else [0, 0, 0]
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R.from_euler("xyz", rpy).as_matrix(), xyz
        return T

    def link_T(link):
        T = np.eye(4)
        while link in joints:
            j = joints[link]
            T = tf(j) @ T
            link = j.find("parent").get("link")
        return T

    out = {}
    for n in names:
        link = by_name[n].find("child").get("link") if n in by_name else n
        out[n] = link_T(link)[:3, 3]
    return out


def prepare_urdf(name: str, out_dir: str) -> str:
    spec = CUSTOM_HANDS[name]
    src = spec["urdf"]
    tree = ET.parse(src)
    root = tree.getroot()
    sdir = os.path.dirname(os.path.abspath(src))
    for m in root.iter("mesh"):
        fn = m.get("filename")
        if fn and not fn.startswith(("/", "package://")):
            p = os.path.join(sdir, fn)
            if not os.path.exists(p) and os.path.exists(os.path.join(sdir, os.path.basename(fn))):
                p = os.path.join(sdir, os.path.basename(fn))         # Wuji: URDF says meshes/, files sit next to it
            m.set("filename", p)
    for tip, (parent, xyz) in spec.get("add_tips", {}).items():
        ET.SubElement(root, "link", {"name": tip})
        j = ET.SubElement(root, "joint", {"name": f"{tip}_fixed", "type": "fixed"})
        ET.SubElement(j, "parent", {"link": parent})
        ET.SubElement(j, "child", {"link": tip})
        ET.SubElement(j, "origin", {"xyz": " ".join(map(str, xyz)), "rpy": "0 0 0"})
    # canonical alignment root (from the hand's own zero pose)
    links = {l.get("name") for l in root.findall("link")}
    children = {j.find("child").get("link") for j in root.findall("joint")}
    old_root = [l for l in links if l not in children][0]
    p = _joint_world(root, [spec["wrist"], spec["bases"]["middle"], spec["bases"]["index"], spec["bases"]["pinky"]])
    x = p[spec["bases"]["middle"]] - p[spec["wrist"]]
    x /= np.linalg.norm(x)
    y = p[spec["bases"]["index"]] - p[spec["bases"]["pinky"]]
    y -= (y @ x) * x
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    R_new_old = np.stack([x, y, z])                      # rows: old-frame axes -> canonical axes
    rpy = R.from_matrix(R_new_old).as_euler("xyz")
    ET.SubElement(root, "link", {"name": "cross_dex_rot_base"})
    j = ET.SubElement(root, "joint", {"name": "cross_dex_rot", "type": "fixed"})
    ET.SubElement(j, "parent", {"link": "cross_dex_rot_base"})
    ET.SubElement(j, "child", {"link": old_root})
    ET.SubElement(j, "origin", {"xyz": "0 0 0", "rpy": " ".join(f"{v:.8f}" for v in rpy)})
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, f"{name}_cross_dex.urdf")
    tree.write(dst)
    return dst


def dexpilot_yaml(name: str, urdf: str, out_dir: str) -> str:
    spec = CUSTOM_HANDS[name]
    lines = ["retargeting:", "  type: DexPilot", f"  urdf_path: {urdf}", f"  wrist_link_name: {spec['wrist']}",
             "  finger_tip_link_names:", *[f"  - {t}" for t in spec["tips"]], "  low_pass_alpha: 0.8",
             f"  scaling_factor: {spec['scaling']}", "  target_joint_names:", *[f"  - {j}" for j in spec["joints"]]]
    dst = os.path.join(out_dir, f"{name}_right_dexpilot.yml")
    open(dst, "w").write("\n".join(lines) + "\n")
    return dst


def make_custom_retargeter(name: str, sim_device: str = "cpu"):
    from dexverse.devices.retargeters.simple_relative_retargeting import (
        HAND_NAME_TO_TARGET,
        SimpleRelativeRetargeter,
        SimpleRelativeRetargeterCfg,
    )

    out_dir = tempfile.mkdtemp(prefix=f"cross_dex_{name}_")
    urdf = prepare_urdf(name, out_dir)
    yml = dexpilot_yaml(name, urdf, out_dir)
    names = CUSTOM_HANDS[name]["joints"]

    class CustomHandRetargeter(SimpleRelativeRetargeter):
        def _get_action_layout(self, robot_type):
            n = len(names)
            return {"output_dim": 6 + n,
                    "hands": {HAND_NAME_TO_TARGET["right"]: {
                        "wrist_trans_indices": (0, 1, 2), "wrist_rot_indices": (3, 4, 5), "wrist_rot_order": "xyz",
                        "wrist_rot_signs": (1.0, 1.0, 1.0), "finger_indices": tuple(range(6, 6 + n)),
                        "finger_joint_names": list(names)}}}

        def _get_dex_retargeting_spec(self):
            return {"hands": {"right": {"urdf_path": urdf, "config_paths": {"dexpilot": yml}}}}

    # robot_type only names the hand here; the layout / spec come from the overrides above
    cfg = SimpleRelativeRetargeterCfg(robot_type="floating_inspire_right", sim_device=sim_device, debug_vis=False)
    return CustomHandRetargeter(cfg), urdf
