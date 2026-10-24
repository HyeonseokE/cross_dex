"""Finger retargeting with Bench2Dex's Manus-tuned ``RetargetBridge`` (simulator-independent).

``bench2dex/`` is the Bench2Dex teleop code (``bench2dex_teleop/teleop``, commit 511cbf3), kept unmodified:
``retarget_bridge.py`` maps 21 MediaPipe keypoints in a canonical wrist frame to robot joint angles with
DexPilot, using per-hand YAMLs (``bench2dex/hand_cfgs``) whose scale / offset / joint locks were tuned
on the Manus skeleton, and the Bench2Dex URDF-Zoo URDFs (directory in ``$URDF_ZOO_DIR``).

:class:`Bench2DexHand` feeds it the OpenXR 26-joint hand (``common/openxr_hand.py``) every device produces.
"""

from __future__ import annotations

import contextlib

import numpy as np

from ...common.rotation_utils import rot_from_wxyz

# OpenXR joint names in MediaPipe 21-keypoint order (wrist, thumb CMC/MCP/IP/TIP, then MCP/PIP/DIP/TIP
# per finger; the finger metacarpals are skipped, as in Bench2Dex).
MEDIAPIPE_FROM_OPENXR = (
    "wrist",
    "thumb_metacarpal", "thumb_proximal", "thumb_distal", "thumb_tip",
    "index_proximal", "index_intermediate", "index_distal", "index_tip",
    "middle_proximal", "middle_intermediate", "middle_distal", "middle_tip",
    "ring_proximal", "ring_intermediate", "ring_distal", "ring_tip",
    "little_proximal", "little_intermediate", "little_distal", "little_tip",
)

# Columns = RetargetBridge canonical wrist axes expressed in the OpenXR wrist joint frame: (+Y, -X, +Z),
# the joint frame turned +90 deg about Z (Bench2Dex xr_hand_source.OPENXR_JOINT_TO_CANONICAL, verified
# against the URDF-Zoo hands' zero-pose fingertips).
OPENXR_JOINT_TO_CANONICAL = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

# Bench2Dex hand configs (hand_cfgs/<name>.yml).
HAND_TYPES = ("shadow", "allegro", "leap", "sharpa", "wuji", "rh56dfx", "rh5dg2", "schunk_svh", "orca",
              "ability", "dexhand021", "revo2")


class Bench2DexHand:
    """One hand: OpenXR joints -> robot joint angles (``joint_names`` order)."""

    def __init__(self, hand_type: str, side: str, input_device: str = "manus", reload_every: int = 120):
        """``input_device="manus"`` uses the Manus-tuned scale / offset, ``"xr"`` the OpenXR hand-tracking ones.
        Edits to ``hand_cfgs/<hand_type>.yml`` scale / offset are reloaded every ``reload_every`` calls."""
        from .bench2dex.retarget_bridge import RetargetBridge

        self.bridge = RetargetBridge(hand_type, side, input_device=input_device)
        self.joint_names = [str(n) for n in self.bridge._retargeting.optimizer.robot.dof_joint_names]
        self._reload_every, self._calls = reload_every, 0

    def retarget(self, joints: dict) -> np.ndarray | None:
        """``joints``: ``{name: [x, y, z, qw, qx, qy, qz]}`` (OpenXR). Returns joint angles or None."""
        try:
            keypoints = np.stack([np.asarray(joints[n], dtype=np.float32)[:3] for n in MEDIAPIPE_FROM_OPENXR])
        except KeyError:
            return None
        wrist_frame = (rot_from_wxyz(np.asarray(joints["wrist"], dtype=np.float64)[3:]) @ OPENXR_JOINT_TO_CANONICAL)
        self._calls += 1
        if self._reload_every and self._calls % self._reload_every == 0:
            self.bridge.reload_tunables_if_changed()
        with _grad_enabled():  # dex-retargeting optimisers use torch autograd
            qpos = self.bridge.retarget_keypoints(keypoints, wrist_frame.astype(np.float32))
        return None if qpos is None else np.asarray(qpos, dtype=np.float32)


@contextlib.contextmanager
def _grad_enabled():
    try:
        import torch
    except ImportError:
        yield
        return
    with torch.enable_grad(), torch.inference_mode(False):  # teleop loops often run under inference_mode
        yield
