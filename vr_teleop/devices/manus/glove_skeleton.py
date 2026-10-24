"""Manus glove skeleton -> OpenXR hand joints (glove only; numpy / scipy, no simulator imports).

Input: the 25 raw-skeleton nodes of one glove in WebXR order (0 wrist, thumb 1-4, index 5-9,
middle 10-14, ring 15-19, little 20-24, metres) and the wrist-node quaternion (x, y, z, w), as the
bridge client returns them (``bridge/manus_bridge_client.py``).

* :class:`GloveWristFrame`: the glove's wrist frame in the OpenXR joint convention (common/openxr_hand.py).
* :func:`glove_to_openxr_joints`: the finger shape re-placed at any wrist pose as the 26 OpenXR joints.
  Where the wrist is comes from elsewhere (gloves have no position tracking).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

from ...common.openxr_hand import HAND_JOINT_NAMES
from ...common.rotation_utils import snap_signed_permutation, wxyz_from_rot

# WebXR 25-node index -> OpenXR 26-joint name (OpenXR adds "palm" in front).
WEBXR_TO_OPENXR = HAND_JOINT_NAMES[1:]
_INDEX_PROXIMAL, _MIDDLE_METACARPAL, _MIDDLE_PROXIMAL, _LITTLE_PROXIMAL = 6, 10, 11, 21


def palm_keypoint_frame(kp: np.ndarray, side: str) -> np.ndarray | None:
    """Wrist frame (columns X/Y/Z, OpenXR convention) from the palm keypoints of 25 WebXR nodes."""
    z = kp[0] - kp[_MIDDLE_PROXIMAL]  # toward the wrist, so -Z points to the fingers
    a = kp[_INDEX_PROXIMAL] - kp[0]
    b = kp[_LITTLE_PROXIMAL] - kp[0]
    back = np.cross(b, a) if side == "right" else np.cross(a, b)
    zn = np.linalg.norm(z)
    if zn < 1e-6:
        return None
    z = z / zn
    y = back - back.dot(z) * z
    yn = np.linalg.norm(y)
    if yn < 1e-6:
        return None
    y = y / yn
    return np.stack([np.cross(y, z), y, z], axis=1)


class GloveWristFrame:
    """Glove wrist frame in the OpenXR convention, from the Manus wrist node (as Bench2Dex does).

    Which local axis of the wrist node is the finger / back-of-hand direction depends on the Manus
    SDK coordinate-system settings, so the map is found once per glove side from the palm keypoints
    and snapped to a signed axis permutation. Without a quaternion the palm-keypoint frame is used.
    """

    def __init__(self, verbose: bool = True):
        self._local_to_openxr: dict[str, np.ndarray] = {}
        self.verbose = verbose

    def __call__(self, kp: np.ndarray, side: str, wq: np.ndarray | None) -> np.ndarray | None:
        palm = palm_keypoint_frame(kp, side)
        if wq is None or palm is None:
            return palm
        node = R.from_quat(wq).as_matrix()
        if side not in self._local_to_openxr:
            m = node.T @ palm
            p = snap_signed_permutation(m)
            self._local_to_openxr[side] = p
            if self.verbose:
                err = np.degrees(np.arccos(np.clip((np.trace(p.T @ m) - 1) / 2, -1.0, 1.0)))
                print(f"[manus] {side} wrist-node axes -> OpenXR wrist frame:\n{p.astype(int)}  (palm-vs-node {err:.1f} deg)")
        return node @ self._local_to_openxr[side]


def glove_to_openxr_joints(
    kp: np.ndarray, frame: np.ndarray, wrist_pos: np.ndarray, wrist_rot: np.ndarray
) -> dict[str, np.ndarray]:
    """Glove keypoints (25, 3) with glove wrist frame ``frame`` -> the 26 OpenXR joints of a hand whose
    wrist is at ``wrist_pos`` with frame ``wrist_rot`` (3x3, OpenXR convention), as 7-vector poses."""
    local = (kp - kp[0]) @ frame  # keypoints in the glove wrist frame
    world = wrist_pos[None, :] + local @ wrist_rot.T
    quat = wxyz_from_rot(wrist_rot)
    joints = {name: np.concatenate([world[i], quat]).astype(np.float32) for i, name in enumerate(WEBXR_TO_OPENXR)}
    palm = 0.5 * (world[_MIDDLE_METACARPAL] + world[_MIDDLE_PROXIMAL])
    joints["palm"] = np.concatenate([palm, quat]).astype(np.float32)
    return joints
