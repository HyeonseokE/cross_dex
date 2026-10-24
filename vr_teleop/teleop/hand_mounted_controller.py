"""Wrist pose of a hand that wears a Quest controller on its back (glove mount; numpy only).

    wrist position = t_ctrl + R_ctrl @ offset      (offset: pivot calibration)
    wrist frame    = R_ctrl @ mount_rot            (OpenXR wrist convention, common/openxr_hand.py)

Both transforms come from ``tools/calibration`` (``mount_calibration.py``). Simulator-independent:
give it controller poses ``[x, y, z, qw, qx, qy, qz]`` from any XR runtime.
"""

from __future__ import annotations

import numpy as np

from ..common.openxr_hand import SIDES
from ..common.rotation_utils import rot_from_wxyz


class HandMountedController:
    """Per-side controller -> wrist transform (pivot offset + mount rotation)."""

    def __init__(self, offsets: dict[str, np.ndarray] | None = None, mount_rots: dict[str, np.ndarray] | None = None):
        self.offset = {s: np.zeros(3) for s in SIDES} | {k: np.asarray(v, float) for k, v in (offsets or {}).items()}
        self.mount_rot = {s: np.eye(3) for s in SIDES} | {k: np.asarray(v, float) for k, v in (mount_rots or {}).items()}
        self.ctrl_rot: dict[str, np.ndarray] = {}  # latest raw controller orientation per side

    def wrist(self, side: str, ctrl_pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(position (3,), OpenXR wrist frame (3, 3))`` for a controller pose ``[x, y, z, qw, qx, qy, qz]``."""
        r_ctrl = rot_from_wxyz(ctrl_pose[3:])
        self.ctrl_rot[side] = r_ctrl
        return ctrl_pose[:3] + r_ctrl @ self.offset[side], r_ctrl @ self.mount_rot[side]
