"""Quest 3 input through Isaac Sim's OpenXR (CloudXR 6 + IsaacTeleop web client).   [Isaac Sim only]

The headset streams to an Isaac Sim XR app; this wraps Isaac Lab's ``OpenXRDevice`` (no retargeters)
to read the controllers, the headset and optical hand tracking in sim-world coordinates, read the
controller buttons, move the XR anchor, and relay the headset menu's START / STOP / RESET.
Import it only inside a running Isaac Sim app (after ``AppLauncher``). For another simulator or XR
runtime, write a class with the same methods.

On this CloudXR / WebXR path the headset streams either controllers or hand tracking, not both at
once (it follows the input mode the headset had when it connected).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ...common.openxr_hand import SIDES

BUTTONS = {"trigger": 2, "squeeze": 3, "primary": 4, "secondary": 5}  # primary = A / X, secondary = B / Y
XR_CONTROLLER_GUI = "/_xr/gui/controllers"  # where omni.kit.xr.ui.stage loads the controller models


def request_ar_session(simulation_app, num_updates: int = 10) -> bool:
    """Start the XR (AR profile) session without the viewport button. Call after the app launched."""
    try:
        from omni.kit.xr.core import XRCore

        XRCore.get_singleton().request_enable_profile("ar")
    except Exception as exc:  # noqa: BLE001 - XR extension missing
        print(f"[quest] could not enable the AR profile: {exc}")
        return False
    for _ in range(num_updates):
        simulation_app.update()
    return True


class TrackedPose:
    """Before an XR runtime streams a controller (or hand) its pose is a constant placeholder (zeros, or
    the XR anchor). A tracked pose always jitters, so a source counts as tracked once it first changes."""

    def __init__(self):
        self._first: np.ndarray | None = None
        self.tracked = False

    def update(self, pose: np.ndarray) -> bool:
        if self.tracked:
            return True
        if not np.isfinite(pose).all():
            return False
        if self._first is None:
            self._first = pose.copy()
            return False
        self.tracked = bool(np.abs(pose - self._first).max() > 1e-6)
        return self.tracked


class QuestXR:
    """Controllers, headset and hand tracking of a Quest 3 inside an Isaac Sim XR app."""

    def __init__(self, xr_cfg=None, controllers: bool = True, hand_tracking: bool = False, sim_device: str = "cpu"):
        from isaaclab.devices import DeviceBase
        from isaaclab.devices.openxr import OpenXRDevice, OpenXRDeviceCfg, XrCfg
        from isaaclab.devices.retargeter_base import RetargeterBase

        self._db = DeviceBase
        self.device = OpenXRDevice(OpenXRDeviceCfg(xr_cfg=xr_cfg or XrCfg(), sim_device=sim_device))
        # No retargeter is attached, so request the streams explicitly.
        features = set()
        if controllers:
            features.add(RetargeterBase.Requirement.MOTION_CONTROLLER)
        if hand_tracking:
            features.add(RetargeterBase.Requirement.HAND_TRACKING)
        self.device._required_features = features
        self._controller = {"right": DeviceBase.TrackingTarget.CONTROLLER_RIGHT, "left": DeviceBase.TrackingTarget.CONTROLLER_LEFT}
        self._hand = {"right": DeviceBase.TrackingTarget.HAND_RIGHT, "left": DeviceBase.TrackingTarget.HAND_LEFT}
        self._data: dict = {}
        self._ghosted: set[str] = set()  # controller model prims made see-through
        self._ghost_listed, self._ghost_misses = False, 0

    def poll(self) -> None:
        """Read the latest XR data; call once per frame before the getters."""
        self._data = self.device._get_raw_data()

    def controller(self, side: str) -> tuple[np.ndarray, np.ndarray] | None:
        """``(pose [x, y, z, qw, qx, qy, qz], inputs)`` of one controller; inputs indexed by ``BUTTONS``."""
        ctrl = self._data.get(self._controller[side])
        if ctrl is None or len(ctrl) == 0:
            return None
        rows = self._db.MotionControllerDataRowIndex
        return np.asarray(ctrl[rows.POSE.value], dtype=np.float64), np.asarray(ctrl[rows.INPUTS.value], dtype=np.float64)

    def hand_joints(self, side: str) -> dict | None:
        """Optical hand tracking: ``{joint_name: [x, y, z, qw, qx, qy, qz]}`` (last valid pose when lost)."""
        joints = self._data.get(self._hand[side])
        return joints if isinstance(joints, dict) else None

    def head(self) -> np.ndarray:
        """Headset pose ``[x, y, z, qw, qx, qy, qz]`` (zeros before the headset connects)."""
        return np.asarray(self.device._calculate_headpose(), dtype=np.float64)

    def move_anchor(self, delta: np.ndarray) -> None:
        """Translate the XR anchor (where the headset's tracking space sits in the scene) by ``delta``."""
        from isaacsim.core.prims import SingleXFormPrim

        anchor = SingleXFormPrim(self.device._xr_anchor_headset_path)
        pos, rot = anchor.get_world_pose()
        anchor.set_world_pose(position=np.asarray(pos, dtype=np.float64) + delta, orientation=rot)

    def set_controller_model_opacity(self, opacity: float) -> list[str]:
        """Make the controller models Isaac Sim's XR draws at the controllers see-through (0 hides them, 1 leaves
        them). They load when the headset reports a controller, and again on a model change, so call this
        periodically; it returns the model prims it changed this time."""
        import carb
        import isaaclab.sim as sim_utils
        from pxr import Usd

        if opacity <= 0.0:
            settings = carb.settings.get_settings()
            for root in ("/persistent/xr", "/xr"):
                for profile in ("ar", "vr"):
                    settings.set_bool(f"{root}/profile/{profile}/controllers/visible", False)
            return []
        stage = sim_utils.get_current_stage()
        material = "/Visuals/vr_teleop_controller_ghost"
        if not stage.GetPrimAtPath(material).IsValid():
            cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.85, 0.9), opacity=float(opacity))
            cfg.func(material, cfg)
        root = stage.GetPrimAtPath(XR_CONTROLLER_GUI)  # the XR app's controller GUI layer
        if not root.IsValid():
            self._ghost_misses += 1
            if self._ghost_misses == 15:  # ~30 s of calls without the layer: show what is there
                xr = stage.GetPrimAtPath("/_xr")
                print(f"[quest] no {XR_CONTROLLER_GUI}; /_xr has "
                      f"{[p.GetPath().pathString for p in Usd.PrimRange(xr)][:40] if xr.IsValid() else 'nothing'}")
            return []
        models = [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                  if p.GetName() == "Model" and p.GetParent().GetName() == "Root"]
        if not self._ghost_listed and models:
            self._ghost_listed = True
            print(f"[quest] controller GUI prims: {[p.GetPath().pathString for p in Usd.PrimRange(root)][:40]}")
        changed = []
        for model in models:
            path = model.GetPath().pathString
            if path in self._ghosted:
                continue
            # A material binding cannot be authored inside an instance: make the instancing prim a plain one.
            prim = model
            while prim.IsInstanceProxy():
                prim = prim.GetParent()
                if prim.IsInstance():
                    prim.SetInstanceable(False)
            sim_utils.bind_visual_material(path, material, stage=stage, stronger_than_descendants=True)
            self._ghosted.add(path)
            changed.append(path)
        return changed

    def add_callback(self, key: str, func: Callable) -> None:
        """Headset menu commands: ``"START"``, ``"STOP"``, ``"RESET"``."""
        self.device.add_callback(key, func)

    def reset(self) -> None:
        self.device.reset()


__all__ = ["BUTTONS", "QuestXR", "SIDES", "TrackedPose", "request_ar_session"]
