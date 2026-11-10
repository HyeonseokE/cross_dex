"""Rotation helpers shared by the devices and the teleop code (numpy / scipy only).

Quaternion order: ``wxyz`` for poses coming from OpenXR / Isaac Lab, ``xyzw`` where scipy is used
directly (Manus wrist-node quaternion). Function names say which one they take.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R


def rot_from_wxyz(q) -> np.ndarray:
    """3x3 rotation from a ``(w, x, y, z)`` quaternion."""
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()


def wxyz_from_rot(m: np.ndarray) -> np.ndarray:
    """``(w, x, y, z)`` quaternion from a 3x3 rotation."""
    x, y, z, w = R.from_matrix(m).as_quat()
    return np.array([w, x, y, z])


def angle_deg(m: np.ndarray) -> float:
    """Rotation angle of a 3x3 rotation, degrees."""
    return float(np.degrees(np.linalg.norm(R.from_matrix(m).as_rotvec())))


def mean_rotation(ms) -> np.ndarray:
    """Mean of several 3x3 rotations."""
    return R.from_matrix(np.stack(ms)).mean().as_matrix()


def snap_signed_permutation(m: np.ndarray) -> np.ndarray:
    """Nearest signed permutation matrix (each column / row has a single +-1)."""
    p = np.zeros((3, 3))
    used: set[int] = set()
    for col in np.argsort(-np.abs(m).max(axis=0)):
        row = max((r for r in range(3) if r not in used), key=lambda r: abs(m[r, col]))
        used.add(row)
        p[row, col] = np.sign(m[row, col])
    return p
