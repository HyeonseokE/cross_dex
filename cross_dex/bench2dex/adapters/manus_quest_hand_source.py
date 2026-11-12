"""Manus gloves (fingers) + glove-mounted Quest 3 controllers (wrist) as a Bench2Dex hand source.   [Isaac Lab only]

Same interface as ``bench2dex/teleop/xr_hand_source.py::XrHandSource`` (``start / stop / add_command_callback /
ensure_anchor / poll / reset_tracking``), so ``XrTeleopController`` / ``XrIsaacLabBridge`` and the collector in
``main.py`` are unchanged. The hand pose is built exactly as in the DexVerse device
(``cross_dex.dexverse.adapters.manus_quest_isaaclab_device.ManusDevice``):

  Manus bridge (TCP) ──► 25 glove keypoints + wrist quaternion ──► GloveWristFrame
  Quest controller pose (OpenXR, sim world frame, XrCfg anchor applied)
      ──► HandMountedController (pivot offset + mount rotation from the calibration JSON) ──► wrist (pos, R)
  glove_to_openxr_joints(kp, frame, pos, R) ──► 26 OpenXR joints [x y z qw qx qy qz] ──► HandSample

Differences from the DexVerse device: no START recentering (XrIsaacLabBridge anchors the wrist relatively at every
START / homing), no mount-rotation calibration at START (the saved calibration is used; run the calibration app
first), and the headset START / STOP / RESET go straight to the Bench2Dex episode commands.
"""

from __future__ import annotations

import os
import time
from typing import Callable, Dict, Optional, Sequence

import numpy as np

from cross_dex.teleop.common.openxr_hand import HAND_JOINT_NAMES as CROSS_DEX_JOINT_NAMES
from cross_dex.teleop.common.rotation_utils import rot_from_wxyz
from cross_dex.teleop.devices.manus.bridge.manus_bridge_client import ManusBridgeClient
from cross_dex.teleop.devices.manus.glove_skeleton import GloveWristFrame, glove_to_openxr_joints
from cross_dex.teleop.hand_mounted_controller import HandMountedController
from cross_dex.teleop.tools.calibration.mount_calibration import MountCalibrationFile

SIDES = ("right", "left")
_FROZEN_POS_M = 1e-5  # a new controller pose whose position moved less than this is "position lost" (IMU-only)


def _hand_sample_cls():
    """Bench2Dex's HandSample (imported lazily: the bench2dex repo is on sys.path only inside main.py)."""
    from teleop.xr_hand_source import HAND_JOINT_NAMES, HandSample

    return HandSample, HAND_JOINT_NAMES


class ManusQuestHandSource:
    """Manus fingers + controller wrist -> ``{side: HandSample}`` per poll."""

    def __init__(
        self,
        calibration_json: Optional[str] = None,
        anchor_pos: Sequence[float] = (0.0, -0.55, 0.0),
        anchor_rot: Sequence[float] = (1.0, 0.0, 0.0, 0.0),
        near_plane: float = 0.15,
        host: str = "127.0.0.1",
        port: int = 47000,
        stale_s: float = 0.5,
        lost_frames: int = 3,
        controller_trace: Optional[str] = None,
        stream_log_every: int = 0,
        sim_device: str = "cpu",
        quest=None,
    ):
        """``calibration_json``: glove-mounted controller calibration (tools/calibration); default $VR_TELEOP_CALIB.
        ``quest``: an object with the ``QuestXR`` interface (poll / controller / add_callback / reset / device) to use
        instead of creating one (tests)."""
        self._calib_path = calibration_json or os.environ.get("VR_TELEOP_CALIB")
        self._anchor_pos = tuple(float(v) for v in anchor_pos)
        self._anchor_rot = tuple(float(v) for v in anchor_rot)
        self._near_plane = float(near_plane)
        self._host, self._port, self._stale_s = host, int(port), float(stale_s)
        self._lost_frames = int(lost_frames)
        self._trace_path = controller_trace
        self._log_every = int(stream_log_every)
        self._sim_device = sim_device
        self._quest = quest
        self._glove = None
        self._glove_frame = GloveWristFrame()
        self._ctrl = HandMountedController()
        self._callbacks: Dict[str, Callable[[], None]] = {}
        self._tracked: Dict[str, object] = {}
        self._placeholder: Dict[str, np.ndarray] = {}  # pose the runtime reports before / while a controller is untracked
        self._wrist: Dict[str, tuple] = {}  # side -> (pos (3,), R (3,3)) last valid wrist
        self._rate_t0, self._rate_n = time.monotonic(), 0
        self._ctrl_prev: Dict[str, np.ndarray] = {}
        self._frozen = {s: 0 for s in SIDES}
        self._lost_since: Dict[str, float] = {}
        self._lost_count = {s: 0 for s in SIDES}
        self._lost_polls = {s: 0 for s in SIDES}
        self._last_joints: Dict[str, np.ndarray] = {}
        self._polls = 0
        self._trace = None
        self._last_log = 0.0

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self) -> bool:
        if self._calib_path and os.path.isfile(self._calib_path):
            calib = MountCalibrationFile(self._calib_path)
            saved = []
            for side in SIDES:
                self._ctrl.offset[side] = calib.offset(side)
                rot = calib.mount_rot(side)
                if rot is not None:
                    self._ctrl.mount_rot[side] = rot
                    saved.append(side)
            print(f"[manus-quest] calibration {self._calib_path}: offsets "
                  + ", ".join(f"{s}={np.round(self._ctrl.offset[s], 4).tolist()}" for s in SIDES)
                  + f"; mount rotation saved for {saved or 'none'}")
            if len(saved) < 2:
                print("[manus-quest] WARNING: mount rotation missing for "
                      f"{[s for s in SIDES if s not in saved]}: run cross_dex/teleop/tools/calibration first")
        else:
            print(f"[manus-quest] WARNING: no calibration file ({self._calib_path!r}); controller = wrist (no mount offset)")
        if self._quest is None:
            from isaaclab.devices.openxr import XrCfg

            from cross_dex.teleop.devices.metaquest.quest_openxr_isaaclab import QuestXR

            xr_cfg = XrCfg(anchor_pos=self._anchor_pos, anchor_rot=self._anchor_rot, near_plane=self._near_plane)
            self._quest = QuestXR(xr_cfg, controllers=True, hand_tracking=False, sim_device=self._sim_device)
        from cross_dex.teleop.devices.metaquest.quest_openxr_isaaclab import TrackedPose

        self._tracked = {s: TrackedPose() for s in SIDES}
        for key, fn in self._callbacks.items():
            self._quest.add_callback(key, fn)
        self._glove = ManusBridgeClient(self._host, self._port, self._stale_s)
        if self._trace_path:
            self._trace = open(self._trace_path, "a", buffering=1)
            self._trace.write("t,side,x,y,z,qw,qx,qy,qz,lost,wrist_x,wrist_y,wrist_z\n")
        print(f"[manus-quest] ready: Manus bridge {self._host}:{self._port}, controllers via OpenXR, "
              f"anchor_pos={self._anchor_pos}, anchor_rot={self._anchor_rot}")
        return True

    def stop(self):
        if self._trace is not None:
            self._trace.close()
            self._trace = None
        self._glove = None

    def add_command_callback(self, key: str, fn: Callable[[], None]):
        """key: "START" | "STOP" | "RESET" (IsaacTeleop client buttons)."""
        self._callbacks[key] = fn
        if self._quest is not None:
            self._quest.add_callback(key, fn)

    def ensure_anchor(self):
        """Re-create the XR anchor prim if a scene rebuild removed it (same as XrHandSource)."""
        if self._quest is None:
            return
        import omni.usd

        path = self._quest.device._xr_anchor_headset_path
        stage = omni.usd.get_context().get_stage()
        if stage is not None and stage.GetPrimAtPath(path).IsValid():
            return
        from isaacsim.core.prims import SingleXFormPrim

        SingleXFormPrim(path, position=self._anchor_pos, orientation=self._anchor_rot)
        print(f"[manus-quest] XR anchor re-created at {path}")

    def reset_tracking(self):
        for s in SIDES:
            self._lost_polls[s] = 0

    def status(self) -> str:
        return self._glove.status() if self._glove is not None else "bridge not started"

    # ── data ─────────────────────────────────────────────────────────────
    def _update_wrists(self) -> None:
        self._quest.poll()
        for side in SIDES:
            ctrl = self._quest.controller(side)
            if ctrl is None:
                continue
            pose, newly = ctrl[0], not self._tracked[side].tracked
            if not self._tracked[side].update(pose):
                self._placeholder[side] = pose.copy()  # constant pose before tracking = the XR anchor placeholder
                continue
            if side in self._placeholder and np.allclose(pose, self._placeholder[side], atol=1e-6):
                # The runtime falls back to the placeholder when it loses the controller: hold the wrist (2026-11-12,
                # the Bench2Dex session re-anchored on every placeholder <-> real transition).
                if side not in self._lost_since:
                    self._lost_since[side] = time.monotonic()
                    self._lost_count[side] += 1
                    print(f"[manus-quest] WARNING {side} controller untracked (placeholder pose): holding the wrist")
                continue
            lost = self._position_lost(side, pose)
            if not lost:
                self._wrist[side] = self._ctrl.wrist(side, pose)
            if self._trace is not None:
                wrist = self._wrist[side][0] if side in self._wrist else np.full(3, np.nan)
                self._trace.write(f"{time.time():.4f},{side}," + ",".join(f"{v:.6f}" for v in pose)
                                  + f",{int(lost)}," + ",".join(f"{v:.5f}" for v in wrist) + "\n")
            if newly and not lost:
                print(f"[manus-quest] {side} controller tracked at {np.round(pose[:3], 3).tolist()}")

    def _position_lost(self, side: str, pose: np.ndarray) -> bool:
        """True while the headset cameras do not see the controller: the runtime keeps rotating it (IMU) but freezes
        the position, which would swing the wrist about the controller. The last wrist pose is held meanwhile."""
        prev = self._ctrl_prev.get(side)
        self._ctrl_prev[side] = pose.copy()
        if prev is None or np.array_equal(prev, pose):
            return side in self._lost_since
        frozen = np.abs(pose[:3] - prev[:3]).max() < _FROZEN_POS_M
        self._frozen[side] = self._frozen[side] + 1 if frozen else 0
        lost = self._frozen[side] >= self._lost_frames
        if lost and side not in self._lost_since:
            self._lost_since[side] = time.monotonic()
            self._lost_count[side] += 1
            print(f"[manus-quest] WARNING {side} controller position lost (out of the headset cameras' view?): "
                  "holding the wrist")
        elif not lost and side in self._lost_since:
            held = time.monotonic() - self._lost_since.pop(side)
            print(f"[manus-quest] {side} controller position back after {held:.1f} s")
        return lost

    def poll(self) -> Dict[str, object]:
        HandSample, names = _hand_sample_cls()
        out: Dict[str, object] = {}
        if self._quest is None or self._glove is None:
            return out
        self._update_wrists()
        gloves = self._glove.gloves()
        for side in SIDES:
            fresh = False
            if side in gloves and side in self._wrist and side not in self._lost_since:
                kp, wq = gloves[side]
                if np.isfinite(kp).all():
                    frame = self._glove_frame(kp, side, wq)
                    if frame is not None:
                        pos, rot = self._wrist[side]
                        joints = glove_to_openxr_joints(kp, frame, pos, rot)
                        self._last_joints[side] = np.stack([np.asarray(joints[n], dtype=np.float32) for n in names])
                        fresh = True
            if side not in self._last_joints:
                continue  # never seen: no sample (XrHandSource reports untracked only after a first pose)
            self._lost_polls[side] = 0 if fresh else self._lost_polls[side] + 1
            out[side] = HandSample(joints=self._last_joints[side], tracked=fresh, lost_polls=self._lost_polls[side])
        self._polls += 1
        now = time.monotonic()
        if self._log_every > 0 and self._polls % self._log_every == 0 or now - self._last_log > 5.0:
            rate = (self._polls - self._rate_n) / max(now - self._rate_t0, 1e-6); self._rate_t0, self._rate_n = now, self._polls
            self._last_log = now
            lost = ", ".join(f"{s} lost {n}x" for s, n in self._lost_count.items() if n)
            print(f"[manus-quest] poll={self._polls:>6} {rate:4.1f} Hz {self._glove.status()} | hands "
                  + ", ".join(f"{s}:{'ok' if (s in out and out[s].tracked) else 'held' if s in out else '-'}" for s in SIDES)
                  + (f" | controller position {lost}" if lost else ""))
        return out


__all__ = ["ManusQuestHandSource", "SIDES", "CROSS_DEX_JOINT_NAMES", "rot_from_wxyz"]
