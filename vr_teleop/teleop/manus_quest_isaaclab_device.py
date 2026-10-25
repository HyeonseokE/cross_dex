"""Isaac Lab teleop device: Manus glove fingers + Quest 3 wrist.   [Isaac Lab only]

:class:`ManusDevice` emits what Isaac Lab's ``OpenXRDevice`` emits for headset hand tracking:
``{HAND_LEFT/HAND_RIGHT: {joint_name: [x, y, z, qw, qx, qy, qz]}}`` over the 26 OpenXR joints, so any
retargeter written for hand tracking works with it. The finger shape comes from the glove
(``devices/manus``), re-placed at a wrist pose chosen by ``wrist_mode``:

* ``"controller"``: the Quest controller mounted on the back of the same hand (``hand_mounted_controller.py``):
  pivot offset and mount rotation from ``calibration_json`` (``tools/calibration``).
* ``"handtracking"``: the headset's optical hand tracking (wrist joint). Gloves can degrade it.
* ``"fixed"``: a fixed wrist, palm down with the fingers along +X (fingers only, no headset).

With ``xr_cfg`` (a VR session) the device also places the XR anchor and relays the headset menu's
START / STOP / RESET. On START it (1) takes the controller-to-hand rotation if none is saved (or
``calibrate_rotation``), with the hand held like the robot hand, (2) moves the XR anchor so the
operator's hand sits on the robot hand (``recenter_on_start``), then (3) fires the START callbacks
(the retargeter's wrist calibration) a few frames later, once the updated wrist reached it.

Robot wrist pose protocol: a retargeter that exposes ``robot_wrist_pose_w`` ([x, y, z, qw, qx, qy, qz],
world) enables (1) and (2); ``robot_wrist_from_openxr`` (quaternion wxyz, ``R_robot = R_openxr * N``)
says how a robot wrist orientation relates to an OpenXR wrist orientation.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from isaaclab.devices.device_base import DeviceBase, DeviceCfg

from ..common.openxr_hand import HAND_JOINT_NAMES, PALM_DOWN_FINGERS_X, SIDES, palm_down_frame
from ..common.rotation_utils import angle_deg, rot_from_wxyz
from ..devices.manus.bridge.manus_bridge_client import ManusBridgeClient
from ..devices.manus.glove_skeleton import GloveWristFrame, glove_to_openxr_joints
from ..tools.calibration.mount_calibration import MountCalibrationFile, mount_rotation
from .hand_mounted_controller import HandMountedController

_TARGET = {"right": DeviceBase.TrackingTarget.HAND_RIGHT, "left": DeviceBase.TrackingTarget.HAND_LEFT}
_START_DELAY_FRAMES = 3  # frames between moving / re-orienting the wrist and the retargeter calibration
_FROZEN_POS_M = 1e-6  # a tracked controller position always jitters more than this between new poses


class ManusDevice(DeviceBase):
    """Manus glove device emitting OpenXR-style hand joints."""

    def __init__(self, cfg: ManusDeviceCfg, retargeters: list | None = None):
        super().__init__(retargeters)
        if cfg.wrist_mode not in ("fixed", "controller", "handtracking"):
            raise ValueError(f"wrist_mode must be 'fixed', 'controller' or 'handtracking', got '{cfg.wrist_mode}'")
        self.cfg = cfg
        self._glove = ManusBridgeClient(cfg.host, cfg.port, cfg.stale_s)
        self._glove_frame = GloveWristFrame()
        self._callbacks: dict[str, Callable] = {}
        default = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)
        self._last = {side: {name: default.copy() for name in HAND_JOINT_NAMES} for side in SIDES}
        self._seen = {side: False for side in SIDES}  # glove data received
        self._wrist: dict[str, tuple[np.ndarray, np.ndarray]] = {}  # tracked wrist: position, OpenXR frame
        self._tracked: dict = {}
        self._ctrl_prev: dict[str, np.ndarray] = {}  # last raw controller pose
        self._frozen = {side: 0 for side in SIDES}  # consecutive new poses with a frozen position
        self._lost_since: dict[str, float] = {}  # side -> time the controller position was lost
        self._lost_count = {side: 0 for side in SIDES}
        self._trace = None
        self._trace_prev: dict[str, np.ndarray] = {}
        if cfg.controller_trace:
            self._trace = open(cfg.controller_trace, "a", buffering=1)
            self._trace.write("t,side,x,y,z,qw,qx,qy,qz,lost,wrist_x,wrist_y,wrist_z\n")
            print(f"[vr_teleop] controller trace: {cfg.controller_trace}")
        self._started = False
        self._pending_start = 0
        self._last_log = 0.0

        self._calib = MountCalibrationFile(cfg.calibration_json) if cfg.calibration_json else None
        self._ctrl = HandMountedController()
        self._mount_saved: set[str] = set()
        if self._calib is not None:
            for side in SIDES:
                self._ctrl.offset[side] = self._calib.offset(side)
                rot = self._calib.mount_rot(side)
                if rot is not None:
                    self._ctrl.mount_rot[side] = rot
                    self._mount_saved.add(side)
            print(f"[vr_teleop] calibration {cfg.calibration_json}: offsets "
                  + ", ".join(f"{s}={np.round(self._ctrl.offset[s], 4).tolist()}" for s in SIDES)
                  + f"; mount rotation saved for {sorted(self._mount_saved) or 'none'}")
        if cfg.calibrate_rotation and self._calib is None:
            raise ValueError("calibrate_rotation saves into calibration_json: set it")

        self._quest = None
        if cfg.xr_cfg is not None:
            try:
                from ..devices.metaquest.quest_openxr_isaaclab import QuestXR, TrackedPose

                self._tracked = {side: TrackedPose() for side in SIDES}
                self._quest = QuestXR(cfg.xr_cfg, controllers=cfg.wrist_mode == "controller",
                                      hand_tracking=cfg.wrist_mode == "handtracking", sim_device=cfg.sim_device)
            except Exception as exc:  # noqa: BLE001 - XR extensions are absent outside an XR app
                print(f"[vr_teleop] no XR session ({type(exc).__name__}: {exc}); running without headset commands")
        if cfg.wrist_mode != "fixed" and self._quest is None:
            raise RuntimeError(f"wrist_mode='{cfg.wrist_mode}' needs an XR session (xr_cfg and a running CloudXR runtime)")

    def __str__(self) -> str:
        xr = ", xr" if self._quest is not None else ""
        return f"ManusDevice(bridge={self.cfg.host}:{self.cfg.port}, wrist_mode={self.cfg.wrist_mode}{xr})"

    def status(self) -> str:
        """Manus bridge connection and the gloves currently streaming."""
        return self._glove.status()

    def reset(self) -> None:
        self._started = False
        if self._quest is not None:
            self._quest.reset()

    def add_callback(self, key: Any, func: Callable) -> None:
        self._callbacks[key] = func
        if self._quest is not None:
            # START from the headset menu goes through _request_start so it can align first.
            self._quest.add_callback(key, self._request_start if key == "START" else func)

    # ------------------------------------------------------------------ START: align operator and robot

    def _request_start(self) -> None:
        rotated = False
        if self.cfg.wrist_mode == "controller" and (self.cfg.calibrate_rotation or not set(self._robot_sides()) <= self._mount_saved):
            rotated = self._calibrate_mount_rotation()
        moved = self.cfg.recenter_on_start and self.cfg.wrist_mode != "fixed" and self._recenter()
        if rotated or moved:
            self._pending_start = _START_DELAY_FRAMES
        elif "START" in self._callbacks:
            self._callbacks["START"]()

    def _robot_wrist_pose(self) -> np.ndarray | None:
        for r in self._retargeters:
            pose = getattr(r, "robot_wrist_pose_w", None)
            if pose is not None:
                return np.asarray(pose, dtype=np.float64)
        return None

    def _robot_sides(self) -> list[str]:
        """Tracked sides the robot is driven by (sides a retargeter tracks; all if unknown)."""
        robot = [side for side, target in _TARGET.items() for r in self._retargeters
                 if target in getattr(r, "_tracked_hands", ())]
        return [s for s in (robot or SIDES) if s in self._wrist]

    def _calibrate_mount_rotation(self) -> bool:
        """Controller-to-hand rotation, taking the hand as held like the robot hand (or, without a robot
        wrist pose, palm down with the fingers toward the headset's horizontal forward)."""
        sides = [s for s in self._robot_sides() if s in self._ctrl.ctrl_rot]
        if not sides:
            print("[vr_teleop] rotation calibration skipped (controller not tracked yet)")
            return False
        pose = self._robot_wrist_pose()
        if pose is not None:
            n = rot_from_wxyz(self.cfg.robot_wrist_from_openxr)
            ref, how = rot_from_wxyz(pose[3:]) @ n.T, "robot hand orientation"
        else:
            ref, how = palm_down_frame(-rot_from_wxyz(self._quest.head()[3:])[:, 2]), "palm down, fingers forward"
        for side in sides:
            self._ctrl.mount_rot[side] = mount_rotation(self._ctrl.ctrl_rot[side], ref)
            print(f"[vr_teleop] {side} controller->hand rotation from {how} ({angle_deg(self._ctrl.mount_rot[side]):.0f} deg)")
            if self.cfg.calibrate_rotation:
                self._calib.set_mount_rot(side, self._ctrl.mount_rot[side], f"ManusDevice START ({how})")
                self._mount_saved.add(side)
        if self.cfg.calibrate_rotation:
            self._calib.save()
            print(f"[vr_teleop] controller->hand rotation saved to {self._calib.path}")
        return True

    def _recenter(self) -> bool:
        """Translate the XR anchor by (robot wrist - operator wrist) so the two coincide."""
        pose = self._robot_wrist_pose()
        sides = [s for s in self._robot_sides() if self._seen[s]]
        if pose is None or not sides:
            print("[vr_teleop] recenter skipped (no robot wrist pose, or the robot's hand is not tracked yet)")
            return False
        delta = pose[:3] - self._wrist[sides[0]][0]
        self._quest.move_anchor(delta)
        print(f"[vr_teleop] recentered: XR anchor moved by {np.round(delta, 3).tolist()} m ({sides[0]} hand -> robot wrist)")
        return True

    # ------------------------------------------------------------------ wrist sources

    def _update_wrists(self) -> None:
        self._quest.poll()
        for side in SIDES:
            if self.cfg.wrist_mode == "controller":
                ctrl = self._quest.controller(side)
                if ctrl is None:
                    continue
                pose, newly = ctrl[0], not self._tracked[side].tracked
                if not self._tracked[side].update(pose):
                    continue
                lost = self._position_lost(side, pose)
                if not lost:
                    self._wrist[side] = self._ctrl.wrist(side, pose)
                if self._trace is not None and not np.array_equal(pose, self._trace_prev.get(side)):
                    self._trace_prev[side] = pose.copy()
                    wrist = self._wrist[side][0] if side in self._wrist else np.full(3, np.nan)
                    self._trace.write(f"{time.monotonic():.4f},{side}," + ",".join(f"{v:.6f}" for v in pose)
                                      + f",{int(lost)}," + ",".join(f"{v:.5f}" for v in wrist) + "\n")
                if lost:
                    continue  # hold the last wrist pose
                if newly:
                    print(f"[vr_teleop] {side} controller tracked at {np.round(pose[:3], 3).tolist()}")
                    if self._started:
                        self._request_start()  # recalibrate on the real controller pose
            else:  # handtracking
                joints = self._quest.hand_joints(side)
                wrist = None if joints is None else joints.get("wrist")
                if wrist is None:
                    continue
                pose, newly = np.asarray(wrist, dtype=np.float64), not self._tracked[side].tracked
                if self._tracked[side].update(pose):
                    self._wrist[side] = (pose[:3], rot_from_wxyz(pose[3:]))
                    if newly:
                        print(f"[vr_teleop] {side} hand tracked by the headset at {np.round(pose[:3], 3).tolist()}")

    def _position_lost(self, side: str, pose: np.ndarray) -> bool:
        """True while the controller is not seen by the headset cameras: the runtime then keeps updating the
        orientation (IMU) but freezes the position, which would swing the wrist about the controller."""
        prev = self._ctrl_prev.get(side)
        self._ctrl_prev[side] = pose.copy()
        if prev is None or np.array_equal(prev, pose):
            return side in self._lost_since  # no new pose this frame
        frozen = np.abs(pose[:3] - prev[:3]).max() < _FROZEN_POS_M
        self._frozen[side] = self._frozen[side] + 1 if frozen else 0
        lost = self._frozen[side] >= self.cfg.lost_frames
        if lost and side not in self._lost_since:
            self._lost_since[side] = time.monotonic()
            self._lost_count[side] += 1
            print(f"[vr_teleop] WARNING {side} controller position lost (out of the headset cameras' view?): "
                  "holding the robot wrist")
        elif not lost and side in self._lost_since:
            held = time.monotonic() - self._lost_since.pop(side)
            jump = 0.0
            if side in self._wrist:
                jump = np.linalg.norm(self._ctrl.wrist(side, pose)[0] - self._wrist[side][0]) * 100
            print(f"[vr_teleop] {side} controller position back after {held:.1f} s (wrist moved {jump:.1f} cm meanwhile)")
        return lost

    # ------------------------------------------------------------------ DeviceBase

    def _get_raw_data(self) -> dict:
        if self._pending_start:
            self._pending_start -= 1
            if not self._pending_start and "START" in self._callbacks:
                self._callbacks["START"]()
        if self.cfg.wrist_mode != "fixed":
            self._update_wrists()
        for side, (kp, wq) in self._glove.gloves().items():
            if not np.isfinite(kp).all():
                continue
            if self.cfg.wrist_mode == "fixed":
                offset = np.array(self.cfg.left_offset if side == "left" else (0.0, 0.0, 0.0))
                pos, rot = np.asarray(self.cfg.wrist_pos, dtype=np.float64) + offset, PALM_DOWN_FINGERS_X
            elif side in self._wrist:
                pos, rot = self._wrist[side]
            else:
                continue  # wait for the headset so START calibrates on a tracked wrist
            frame = self._glove_frame(kp, side, wq)
            if frame is not None:
                self._last[side] = glove_to_openxr_joints(kp, frame, pos, rot)
                self._seen[side] = True
        if self.cfg.auto_start and not self._started and any(self._seen.values()):
            self._started = True
            self._request_start()
        now = time.monotonic()
        if self.cfg.log_every_s and now - self._last_log > self.cfg.log_every_s:
            self._last_log = now
            lost = ", ".join(f"{s} lost {n}x" for s, n in self._lost_count.items() if n)
            print(f"[vr_teleop] {self._glove.status()}" + (f" | controller position {lost}" if lost else ""))
        return {target: self._last[side] for side, target in _TARGET.items()}


@dataclass
class ManusDeviceCfg(DeviceCfg):
    """Configuration for :class:`ManusDevice`."""

    host: str = "127.0.0.1"  # Manus bridge
    port: int = 47000
    stale_s: float = 0.5
    wrist_mode: str = "fixed"  # "fixed" | "controller" | "handtracking"
    calibration_json: str | None = None  # glove-mounted controller calibration (tools/calibration/mount_calibration.py)
    calibrate_rotation: bool = False  # on START take and save the controller-to-hand rotation (hand like the robot hand)
    robot_wrist_from_openxr: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # wxyz, R_robot = R_openxr * N
    recenter_on_start: bool = True  # on START move the XR anchor so your hand sits on the robot hand
    auto_start: bool = True  # start on the first glove frame instead of the headset START
    wrist_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)  # "fixed" mode wrist position
    left_offset: tuple[float, float, float] = (0.0, 0.3, 0.0)  # "fixed" mode left-hand offset
    log_every_s: float = 5.0
    controller_trace: str | None = None  # CSV of every new controller pose (diagnosis)
    lost_frames: int = 3  # new controller poses with a frozen position before the wrist is held
    xr_cfg: Any = None  # isaaclab XrCfg; set in a VR session
    retargeters: list = field(default_factory=list)
    class_type: type[DeviceBase] = ManusDevice
