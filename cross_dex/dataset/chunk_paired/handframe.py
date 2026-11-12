"""Common hand representation of all sources (numpy only, no simulator imports).

Joint order = DexCanvas / SNAP order (``DC``): wrist(palm), index/middle/ring/pinky MCP-PIP-DIP, thumb CMC-MCP-IP, then
the 5 tips. Wrist rotation = MANO root rotation (``wrist_rotvec``, world, axis-angle); the retargeting wrapper turns it
into the OpenXR wrist frame with the fixed axis map ``MANO_TO_OPENXR``. Sources without a MANO root rotation build it
from keypoints: ``openxr_wrist_rotation`` (OpenXR convention from the MCPs) x ``KP_TO_MANO_ROOT``.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

DC = ["palm", "index_mcp", "index_pip", "index_dip", "middle_mcp", "middle_pip", "middle_dip", "ring_mcp", "ring_pip",
      "ring_dip", "pinky_mcp", "pinky_pip", "pinky_dip", "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
      "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
DCI = {n: i for i, n in enumerate(DC)}

# OpenPose / MANO-21 order (wrist, thumb 1-4, index 1-4, middle 1-4, ring 1-4, pinky 1-4; 1 = CMC/MCP ... 4 = tip)
# -> DC order: DC[i] = OP[OPENPOSE_TO_DC[i]]
OPENPOSE_TO_DC = [0, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 1, 2, 3, 4, 8, 12, 16, 20]

# OpenXR wrist axes (columns X, Y, Z) in the MANO root frame: MANO +x = towards the fingers (= OpenXR -Z),
# MANO +z = back of the hand (= OpenXR +Y).
MANO_TO_OPENXR = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])

# Keypoint-built OpenXR frame expressed in the MANO root frame (R_kp = R_root @ KP_IN_ROOT), measured on DexCanvas
# shard 0 (2026-11-11, 74k frames: per-frame spread < 1 deg, since MANO MCPs are rigid in the root frame). It differs
# from MANO_TO_OPENXR by a constant ~5.6 deg. Used to give keypoint-only sources the same wrist frame as DexCanvas.
KP_IN_ROOT = np.array([[-0.036, 0.011, -0.999], [-0.995, -0.097, 0.035], [-0.097, 0.995, 0.014]])
_u, _, _vt = np.linalg.svd(KP_IN_ROOT)
KP_IN_ROOT = _u @ _vt                                    # re-orthonormalise the rounded values


def openxr_wrist_rotation(j: np.ndarray) -> np.ndarray:
    """[T,21,3] DC-order joints -> [T,3,3] wrist rotation (columns X,Y,Z) in the OpenXR hand convention
    (-Z: wrist -> middle MCP, +Y: back of the hand, X = Y x Z), right hand."""
    w = j[:, DCI["palm"]]
    fwd = j[:, DCI["middle_mcp"]] - w
    fwd = fwd / np.linalg.norm(fwd, axis=-1, keepdims=True)
    dorsal = np.cross(j[:, DCI["pinky_mcp"]] - w, j[:, DCI["index_mcp"]] - w)
    dorsal -= (dorsal * fwd).sum(-1, keepdims=True) * fwd
    dorsal /= np.linalg.norm(dorsal, axis=-1, keepdims=True)
    z = -fwd
    y = dorsal
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=-1)


def root_rotvec_from_keypoints(j: np.ndarray) -> np.ndarray:
    """[T,21,3] DC-order joints -> [T,3] MANO-root-equivalent wrist rotation (axis-angle), same convention as
    DexCanvas ``wrist_rotvec``."""
    return R.from_matrix(openxr_wrist_rotation(j) @ KP_IN_ROOT.T).as_rotvec()


def mano_wrist_rotation(rotvec: np.ndarray) -> np.ndarray:
    """[T,3] MANO root rotvec (world) -> [T,3,3] wrist rotation (columns X,Y,Z) in the OpenXR hand convention."""
    return R.from_rotvec(rotvec).as_matrix() @ MANO_TO_OPENXR
