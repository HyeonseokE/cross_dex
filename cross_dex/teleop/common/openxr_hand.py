"""Hand conventions shared by every device: OpenXR joints, the wrist frame, a template hand.

Every device in this package hands the teleop code the same thing: the 26 OpenXR hand joints as
``{joint_name: [x, y, z, qw, qx, qy, qz]}`` in the simulator world, with the wrist orientation in the
OpenXR joint convention:

* +Y points out of the back of the hand,
* -Z points along the bones toward the fingertips (+Z toward the forearm),
* X = Y x Z (thumb side for a left hand, little-finger side for a right hand).

This is what Isaac Lab's ``OpenXRDevice`` produces for headset hand tracking, so retargeters written
for hand tracking take glove + controller input unchanged.
"""

from __future__ import annotations

import numpy as np

# XR_EXT_hand_tracking joint order (same as isaaclab.devices.openxr.common.HAND_JOINT_NAMES; duplicated
# so this module imports without Isaac Sim).
HAND_JOINT_NAMES = (
    "palm", "wrist",
    "thumb_metacarpal", "thumb_proximal", "thumb_distal", "thumb_tip",
    "index_metacarpal", "index_proximal", "index_intermediate", "index_distal", "index_tip",
    "middle_metacarpal", "middle_proximal", "middle_intermediate", "middle_distal", "middle_tip",
    "ring_metacarpal", "ring_proximal", "ring_intermediate", "ring_distal", "ring_tip",
    "little_metacarpal", "little_proximal", "little_intermediate", "little_distal", "little_tip",
)
SIDES = ("right", "left")

# Bones of the hand skeleton (pairs of joint names), plus the knuckle line, for drawing.
_FINGERS = ("index", "middle", "ring", "little")
BONES = [("wrist", "thumb_metacarpal"), ("thumb_metacarpal", "thumb_proximal"),
         ("thumb_proximal", "thumb_distal"), ("thumb_distal", "thumb_tip")]
for _f in _FINGERS:
    _chain = ["wrist"] + [f"{_f}_{j}" for j in ("metacarpal", "proximal", "intermediate", "distal", "tip")]
    BONES += list(zip(_chain[:-1], _chain[1:]))
BONES += [("index_proximal", "middle_proximal"), ("middle_proximal", "ring_proximal"), ("ring_proximal", "little_proximal")]


def palm_down_frame(forward: np.ndarray, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """OpenXR wrist frame (columns X, Y, Z) of a palm-down hand whose fingers point along ``forward``."""
    up = np.asarray(up, dtype=np.float64)
    f = np.asarray(forward, dtype=np.float64) - np.dot(forward, up) * up
    f /= max(np.linalg.norm(f), 1e-9)
    y, z = up, -f
    return np.stack([np.cross(y, z), y, z], axis=1)


# Palm down, fingers along world +X (Z-up world).
PALM_DOWN_FINGERS_X = palm_down_frame(np.array([1.0, 0.0, 0.0]))


def flat_hand_template(side: str) -> dict[str, np.ndarray]:
    """Open flat hand (fingers together), joint positions in the OpenXR wrist frame (metres)."""
    s = -1.0 if side == "right" else 1.0  # the thumb is at -X for a right hand, +X for a left hand
    pts = {"wrist": (0.0, 0.0, 0.0),
           "thumb_metacarpal": (0.022, 0.0, -0.025), "thumb_proximal": (0.042, -0.005, -0.050),
           "thumb_distal": (0.052, -0.008, -0.080), "thumb_tip": (0.058, -0.010, -0.100)}
    for f, x, length in (("index", 0.022, 1.0), ("middle", 0.0, 1.08), ("ring", -0.020, 1.0), ("little", -0.038, 0.82)):
        pts[f"{f}_metacarpal"] = (x * 0.5, 0.0, -0.02)
        z = -0.085 * (0.95 if f == "little" else 1.0)
        pts[f"{f}_proximal"] = (x, 0.0, z)
        for j, seg in (("intermediate", 0.042), ("distal", 0.026), ("tip", 0.022)):
            z -= seg * length
            pts[f"{f}_{j}"] = (x, 0.0, z)
    pts["palm"] = (0.0, 0.0, -0.05)
    return {k: np.array([s * v[0], v[1], v[2]]) for k, v in pts.items()}


def place_hand(local: dict[str, np.ndarray], pos: np.ndarray, rot: np.ndarray) -> dict[str, np.ndarray]:
    """Joint positions given in a wrist frame -> world, for a wrist at ``pos`` with frame ``rot``."""
    return {k: pos + rot @ v for k, v in local.items()}


def skeleton_points(joints: dict, bone_dots: int = 6) -> tuple[np.ndarray, np.ndarray]:
    """Points for drawing a hand: joint positions, then ``bone_dots`` points along each bone.

    Returns ``(points (N, 3), is_bone (N,))``. ``joints`` values may be 3-vectors or 7-vector poses.
    """
    pos = {k: np.asarray(v, dtype=np.float64)[:3] for k, v in joints.items()}
    joint_pts = [pos[n] for n in HAND_JOINT_NAMES if n != "palm" and n in pos]
    t = np.linspace(0.0, 1.0, bone_dots + 2)[1:-1, None]
    bone_pts = [pos[a] * (1 - t) + pos[b] * t for a, b in BONES if a in pos and b in pos]
    points = np.concatenate([np.stack(joint_pts), *bone_pts])
    return points, np.r_[np.zeros(len(joint_pts), bool), np.ones(len(points) - len(joint_pts), bool)]
