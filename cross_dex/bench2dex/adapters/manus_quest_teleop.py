"""Bench2Dex teleop with Manus gloves + glove-mounted Quest controllers.   [Isaac Lab only, Bench2Dex]

``ManusQuestTeleopController`` is Bench2Dex's ``XrTeleopController`` with the finger retargeting taken from cross_dex
(``Bench2DexHand``: the Manus-tuned hand_cfgs / angle prior used for DexVerse collection) instead of the fork's
``teleop/hand_cfgs`` with the OpenXR ("xr") scale. The wrist path (``XrIsaacLabBridge``: anchor-relative mapping
into the arm IK) is unchanged.

``start_manus_quest_teleop(args_cli, hand_type, robot_art)`` is what ``main.py --teleop-device manus-quest`` calls.
Environment: VR_TELEOP_CALIB (calibration JSON), VR_TELEOP_BRIDGE_PORT (Manus bridge, 47000), VR_TELEOP_TRACE
(controller CSV, optional).
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

from cross_dex.bench2dex.adapters.manus_quest_hand_source import ManusQuestHandSource
from cross_dex.teleop.retarget.bench2dex_finger_retargeter import Bench2DexHand


def _xr_base():
    from teleop.xr_hand_source import HAND_JOINT_NAMES, SIDES
    from teleop.xr_teleop import XrIsaacLabBridge, XrTeleopController

    return XrTeleopController, XrIsaacLabBridge, HAND_JOINT_NAMES, SIDES


def make_controller_class():
    """Built lazily: the bench2dex package is importable only inside the bench2dex repo (main.py)."""
    XrTeleopController, _, HAND_JOINT_NAMES, SIDES = _xr_base()

    class ManusQuestTeleopController(XrTeleopController):
        """XrTeleopController whose finger retargeting is cross_dex's Bench2DexHand (Manus-tuned)."""

        def start(self, wait_timeout: float = 0.0) -> bool:
            if self._started:
                return True
            if not self._source.start():
                return False
            self._hands: Dict[str, Bench2DexHand] = {}
            for side, on in (("right", self._enable_right), ("left", self._enable_left)):
                if on:
                    self._hands[side] = Bench2DexHand(self._hand_type, side, input_device="manus")
            self._bridge_right = self._hands["right"].bridge if "right" in self._hands else None
            self._bridge_left = self._hands["left"].bridge if "left" in self._hands else None
            self._started = True
            print(f"[ManusQuestTeleopController] Started: hand_type={self._hand_type}, "
                  f"right={'ON' if self._enable_right else 'OFF'}, left={'ON' if self._enable_left else 'OFF'} "
                  "(fingers: cross_dex Bench2DexHand, Manus-tuned configs)")
            return True

        def step(self) -> Optional[Dict[str, np.ndarray]]:
            if not self._started:
                return None
            self._samples = self._source.poll()
            result: Dict[str, np.ndarray] = {}
            for side in SIDES:
                sample = self._samples.get(side)
                if sample is None or not sample.tracked:
                    continue
                self._filter_wrist(side, sample)
                hand = self._hands.get(side)
                if hand is None:
                    continue
                joints = {n: sample.joints[i] for i, n in enumerate(HAND_JOINT_NAMES)}
                q = hand.retarget(joints)
                if q is not None:
                    result[side] = q
            return result if result else None

    return ManusQuestTeleopController


def start_manus_quest_teleop(args_cli, hand_type: str, robot_art):
    """Bridge for ``main.py --teleop-device manus-quest`` (same role as main.py's _start_quest_teleop)."""
    _, XrIsaacLabBridge, _, _ = _xr_base()
    source = ManusQuestHandSource(
        calibration_json=os.environ.get("VR_TELEOP_CALIB"),
        anchor_pos=args_cli.xr_anchor_pos,
        anchor_rot=args_cli.xr_anchor_rot,
        port=int(os.environ.get("VR_TELEOP_BRIDGE_PORT", "47000")),
        controller_trace=os.environ.get("VR_TELEOP_TRACE") or None,
        stream_log_every=args_cli.xr_stream_log,
        sim_device=str(getattr(robot_art, "device", "cpu")),
    )
    ctrl = make_controller_class()(source, hand_type, enable_right=True, enable_left=True,
                                   wrist_smoothing=args_cli.xr_wrist_smoothing)
    # Jump rejection is per poll: at the ~6 Hz the Bench2Dex loop ran on bs1 (2026-11-12, ithor background) a normal
    # hand move exceeds the 8 cm default and re-anchors every few polls. B2D_MAX_STEP_M / B2D_RELOCK_POLLS override.
    bridge = XrIsaacLabBridge(ctrl, robot_art, pos_scale=args_cli.xr_pos_scale, log_every=args_cli.xr_stream_log,
                              max_target_step_m=float(os.environ.get("B2D_MAX_STEP_M", "0.25")),
                              relock_after_lost_polls=int(os.environ.get("B2D_RELOCK_POLLS", "6")))
    if not bridge.start():
        print("[WARN] Manus + Quest teleop failed to start. Running without teleop.")
        bridge.stop()
        return None
    print(f"[INFO] Teleop active (manus-quest): {bridge}")
    return bridge
