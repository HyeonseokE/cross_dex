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


def make_bridge_class():
    """XrIsaacLabBridge whose wrist ORIENTATION is absolute (robot hand faces where the operator's hand faces), as in
    DexVerse; the position stays anchor-relative. Built lazily (bench2dex importable only inside main.py)."""
    import numpy as _np
    from teleop.math_utils import quat_wxyz_to_rotmat, rotmat_to_quat_wxyz
    from teleop.xr_teleop import XrIsaacLabBridge, _WristAnchor

    class ManusQuestIsaacLabBridge(XrIsaacLabBridge):
        """2026-11-12: with the relative mapping the robot hand kept the arm's home orientation plus the rotation since
        the anchor, so it faced elsewhere than the operator's hand (left arm especially). Absolute: R_wrist_target =
        R_openxr_wrist @ N, N derived from the arm's URDF (fingers / index side / back of hand in the wrist frame)."""

        def _wrist_abs_rotation(self, side: str):
            """N for ``side`` (3x3) from the IK's pinocchio model at the neutral pose, or None (-> relative mapping)."""
            cache = getattr(self, "_wrist_abs_cache", None)
            if cache is None:
                cache = self._wrist_abs_cache = {}
            if side in cache:
                return cache[side]
            N = None
            try:
                import pinocchio as pin

                ik, _, _ = self._arm(side)
                m, d = ik._pin_model, ik._pin_data
                q = pin.neutral(m)
                pin.forwardKinematics(m, d, q)
                pin.updateFramePlacements(m, d)
                hand = self._teleop._hands[side]
                tips = list(hand.bridge.finger_tip_link_names)  # [thumb, index, middle, ring, (pinky)]
                side_cfg = getattr(hand.bridge, "_side_cfg", None) or {}
                prefix = side_cfg.get("articulation_prefix", "") if isinstance(side_cfg, dict) else ""
                names = None
                for pre in (prefix, "", "l_" if side == "left" else "", side + "_"):
                    cand = [pre + t for t in tips]
                    if all(m.existFrame(n) for n in cand):
                        names = cand
                        break
                if names is None:
                    raise ValueError(f"fingertip frames {tips} (prefix {prefix!r}) not in the arm URDF")
                W = d.oMf[m.getFrameId(ik._pin_ee_frame_name)] if hasattr(ik, "_pin_ee_frame_name") else d.oMf[ik._ee_frame_id]
                to_w = lambda n: W.rotation.T @ (d.oMf[m.getFrameId(n)].translation - W.translation)
                tip = {k: to_w(n) for k, n in zip(("thumb", "index", "middle", "ring", "pinky")[: len(names)], names)}
                f = (tip["index"] + tip["middle"] + tip.get("ring", tip["middle"])) / 3.0
                f /= _np.linalg.norm(f)
                i = tip["index"] - tip.get("pinky", tip.get("ring"))
                i -= f * (i @ f)
                i /= _np.linalg.norm(i)
                b = _np.cross(f, i) if side == "right" else _np.cross(i, f)  # back of the hand
                Z, Y = -f, b
                X = _np.cross(Y, Z)
                N = _np.stack([X, Y, Z])  # rows: OpenXR axes in the wrist frame -> R_wrist = R_openxr @ N
                print(f"[manus-quest] {side} wrist: absolute orientation, N (OpenXR->{ik.ee_body}) = {_np.round(N, 2).tolist()}")
            except Exception as exc:  # noqa: BLE001
                print(f"[manus-quest] {side} wrist: absolute orientation unavailable ({type(exc).__name__}: {exc}); relative mapping")
            cache[side] = N
            return N

        def _update_arm_side(self, side: str, target) -> bool:
            if os.environ.get("B2D_WRIST_ROT", "absolute") != "absolute":
                return super()._update_arm_side(side, target)
            ik, arm_cols, ee_idx = self._arm(side)
            if ik is None:
                return False
            lost = self._teleop.lost_polls(side)
            if lost is None:
                return False
            if lost > 0:
                if lost >= self._relock_after:
                    self._anchors[side] = None
                return False
            pose = self._teleop.wrist_pose(side)
            if pose is None:
                return False
            p_h, R_h = pose
            N = self._wrist_abs_rotation(side)
            if N is None:
                return super()._update_arm_side(side, target)
            art = self._articulation
            ee_pos_w = art.data.body_pos_w[:, ee_idx]
            ee_quat_w = art.data.body_quat_w[:, ee_idx]
            anchor = self._anchors[side]
            if anchor is None:
                R_ee0 = quat_wxyz_to_rotmat(ee_quat_w[0].cpu().numpy())
                p_r0 = ee_pos_w[0].cpu().numpy().astype(_np.float64)
                ee_offset = getattr(ik, "_ee_offset", None)
                if ee_offset is not None:
                    p_r0 = p_r0 + R_ee0 @ ee_offset
                self._anchors[side] = _WristAnchor(p_h, R_h, p_r0, R_ee0)
                print(f"[XrIsaacLabBridge] {side} wrist anchored (position only; orientation absolute): human={_np.round(p_h, 3)}, robot palm={_np.round(p_r0, 3)}")
                return False
            palm_target = anchor.p_r0 + self._pos_scale * (p_h - anchor.p_h0)
            if anchor.last_target is not None and _np.linalg.norm(palm_target - anchor.last_target) > self._max_step:
                anchor.rejected += 1
                if anchor.rejected >= self._max_rejected:
                    print(f"[XrIsaacLabBridge] {side}: {anchor.rejected} consecutive jumps, re-anchoring")
                    self._anchors[side] = None
                return False
            anchor.rejected = 0
            R_target = R_h @ N
            if anchor.last_target is not None:
                self._record_tracking_error(side, ik, ee_pos_w, ee_quat_w, anchor)
            anchor.last_target = palm_target
            anchor.last_R_target = R_target
            joint_pos_des = ik.compute(
                target_pos=palm_target, target_quat=rotmat_to_quat_wxyz(R_target),
                current_joint_pos=art.data.joint_pos[:, arm_cols],
                ee_pos_w=ee_pos_w, ee_quat_w=ee_quat_w,
                root_pos_w=art.data.root_pos_w, root_quat_w=art.data.root_quat_w,
            )
            target[0, arm_cols] = joint_pos_des[0, :len(arm_cols)]
            self._visualize_arm(side, palm_target, ee_pos_w)
            return True

    return ManusQuestIsaacLabBridge


def start_manus_quest_teleop(args_cli, hand_type: str, robot_art):
    """Bridge for ``main.py --teleop-device manus-quest`` (same role as main.py's _start_quest_teleop)."""
    XrIsaacLabBridge = make_bridge_class()  # absolute wrist orientation (B2D_WRIST_ROT=relative for the stock mapping)
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
