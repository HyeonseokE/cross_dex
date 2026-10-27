"""DexVerse integration: ``--teleop_device manus`` in DexVerse's teleop_agent.py / record_demos.py.   [DexVerse only]

* :class:`Bench2DexFingerRetargeter`: DexVerse's ``SimpleRelativeRetargeter`` (wrist: displacement from
  the START pose, robot action layout) with the fingers retargeted by :class:`Bench2DexHand`.
* :func:`add_manus_device`: adds a ``"manus"`` entry to an env cfg's ``teleop_devices`` that reuses the
  hand-tracking entry's retargeters and XR anchor.
* :func:`add_manus_device_from_env`: the same, configured from ``VR_TELEOP_*`` environment variables
  (what the DexVerse scripts call)::

      VR_TELEOP_WRIST=controller|handtracking|fixed   (default fixed)
      VR_TELEOP_CALIB=<calibration json>              (tools/calibration/mount_calibration.py)
      VR_TELEOP_CALIBRATE_ROTATION=1                  save the controller-to-hand rotation on START
      VR_TELEOP_FINGERS=bench2dex|dexverse            (default bench2dex)
      VR_TELEOP_AUTO_START=1                          start on the first glove frame (default: headset START)
      VR_TELEOP_TRACE=<csv>                           write every new controller pose (diagnosis)
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
from dexverse.devices.retargeters import SimpleRelativeRetargeterCfg
from dexverse.devices.retargeters.simple_relative_retargeting import SimpleRelativeRetargeter
from isaaclab.devices.device_base import DeviceBase
from isaaclab.devices.retargeter_base import RetargeterBase
from scipy.spatial.transform import Rotation as R

from .manus_quest_isaaclab_device import ManusDeviceCfg
from .retarget.bench2dex_finger_retargeter import Bench2DexHand

# DexVerse hand family -> Bench2Dex hand config.
BENCH2DEX_HAND_TYPE = {"shadow": "shadow", "allegro": "allegro", "leap": "leap", "sharpa": "sharpa",
                       "wuji": "wuji", "inspire": "rh56dfx"}
# DexVerse retargeters normalise the OpenXR wrist with R_hand * Ry(90) * Rx(-90) to get the robot wrist.
_N = R.from_euler("y", 90, degrees=True) * R.from_euler("x", -90, degrees=True)
DEXVERSE_ROBOT_WRIST_FROM_OPENXR = tuple(float(v) for v in np.roll(_N.as_quat(), 1))  # wxyz
_SIDE = {DeviceBase.TrackingTarget.HAND_LEFT: "left", DeviceBase.TrackingTarget.HAND_RIGHT: "right"}
# DexVerse action finger joint -> Bench2Dex (URDF-Zoo) joint, where the two name the same joint differently.
# Inspire: DexVerse drives the 6 actuated joints (the rest are mimic), URDF-Zoo RH56DFX calls them <side>_<finger>_<n>_joint.
_INSPIRE_TO_BENCH2DEX = {"thumb_proximal_yaw_joint": "thumb_1_joint", "thumb_proximal_pitch_joint": "thumb_2_joint",
                         "index_proximal_joint": "index_1_joint", "middle_proximal_joint": "middle_1_joint",
                         "ring_proximal_joint": "ring_1_joint", "pinky_proximal_joint": "little_1_joint"}


def bench2dex_joint_name(action_name: str, bench2dex_names: list[str], side: str) -> str | None:
    """The Bench2Dex joint driving the DexVerse action joint ``action_name`` (None if there is none)."""
    if action_name in bench2dex_names:
        return action_name
    bare = re.sub(r"^(right|left)_", "", action_name)
    leap = re.fullmatch(r"(?:leap_)?(\d+)", bare)  # LEAP: DexVerse leap_<n> / <side>_<n>, URDF-Zoo "<n>"
    if leap and leap.group(1) in bench2dex_names:
        return leap.group(1)
    if bare in _INSPIRE_TO_BENCH2DEX:
        name = f"{side}_{_INSPIRE_TO_BENCH2DEX[bare]}"
        return name if name in bench2dex_names else None
    return None


class Bench2DexFingerRetargeter(SimpleRelativeRetargeter):
    """Relative wrist (displacement from the START pose) + Bench2Dex fingers, matched to the action by name."""

    def __init__(self, cfg: Bench2DexFingerRetargeterCfg):
        cfg.initialize_dex_retargeting = False  # Bench2Dex replaces the DexVerse dex-retargeting YAMLs
        super().__init__(cfg)
        parts = cfg.robot_type.split("_")  # floating_<hand>_<side>
        if len(parts) < 3 or parts[0] != "floating":
            raise ValueError(f"Bench2Dex fingers expect robot_type floating_<hand>_<side>, got '{cfg.robot_type}'")
        hand_type = cfg.bench2dex_hand_type or BENCH2DEX_HAND_TYPE.get(parts[1])
        if hand_type is None:
            raise ValueError(f"No Bench2Dex hand config for hand family '{parts[1]}'")
        self._hands = {}
        for hand in self._tracked_hands:
            b2d = Bench2DexHand(hand_type, _SIDE[hand], input_device=cfg.bench2dex_input_device)
            self._hands[hand] = b2d
            self._dex_output_joint_names[hand] = b2d.joint_names
            b2d_names = list(b2d.joint_names)
            self._dex_to_action_finger_indices[hand] = [
                None if (b := bench2dex_joint_name(n, b2d_names, _SIDE[hand])) is None else b2d_names.index(b)
                for n in self._layout["hands"][hand].get("finger_joint_names", ())]
            names = self._layout["hands"][hand].get("finger_joint_names", ())
            missing = [n for n, i in zip(names, self._dex_to_action_finger_indices[hand]) if i is None]
            print(f"[vr_teleop] {hand.name}: {b2d.bridge}" + (f", unmatched action joints {missing}" if missing else ""))
        self.robot_wrist_pose_w: np.ndarray | None = None

    def set_robot_wrist_pose(self, pose_w: np.ndarray) -> None:
        """Robot wrist world pose [x, y, z, qw, qx, qy, qz], pushed every step by teleop_agent / record_demos."""
        self.robot_wrist_pose_w = np.asarray(pose_w, dtype=np.float64)

    def _retarget_hand_fingers(self, hand_data: Any, hand: DeviceBase.TrackingTarget) -> np.ndarray:
        n_out = len(self._dex_output_joint_names.get(hand, []))
        qpos = self._hands[hand].retarget(hand_data) if hand in self._hands and isinstance(hand_data, dict) else None
        return np.zeros(n_out, dtype=np.float32) if qpos is None else qpos


@dataclass
class Bench2DexFingerRetargeterCfg(SimpleRelativeRetargeterCfg):
    """Configuration for :class:`Bench2DexFingerRetargeter`."""

    bench2dex_hand_type: str | None = None  # override hand_cfgs/<name>.yml; default from robot_type
    bench2dex_input_device: str = "manus"  # "manus": Manus-tuned scale / offset, "xr": OpenXR hand-tracking ones
    retargeter_type: type[RetargeterBase] = Bench2DexFingerRetargeter


def _bench2dex_fingers(retargeter_cfg):
    if type(retargeter_cfg) is not SimpleRelativeRetargeterCfg:
        print(f"[vr_teleop] {type(retargeter_cfg).__name__} kept (Bench2Dex fingers replace the relative retargeter only)")
        return retargeter_cfg
    kwargs = {f.name: getattr(retargeter_cfg, f.name) for f in fields(SimpleRelativeRetargeterCfg) if f.name != "retargeter_type"}
    return Bench2DexFingerRetargeterCfg(**kwargs)


def add_manus_device(devices_cfg, name: str = "manus", source: str = "handtracking", use_xr: bool = False,
                     fingers: str = "bench2dex", **overrides) -> None:
    """Add a Manus device reusing the retargeters (and, with ``use_xr``, the XR anchor) of ``source``.

    ``fingers="bench2dex"`` retargets the fingers with Bench2Dex's Manus-tuned ``RetargetBridge``;
    ``fingers="dexverse"`` keeps the DexVerse dex-retargeting YAMLs (tuned for Quest hand tracking).
    """
    src = devices_cfg.devices[source]
    if use_xr:
        overrides.setdefault("xr_cfg", getattr(src, "xr_cfg", None))
    overrides.setdefault("robot_wrist_from_openxr", DEXVERSE_ROBOT_WRIST_FROM_OPENXR)
    retargeters = list(src.retargeters)
    if fingers == "bench2dex":
        retargeters = [_bench2dex_fingers(r) for r in retargeters]
    elif fingers != "dexverse":
        raise ValueError(f"fingers must be 'bench2dex' or 'dexverse', got '{fingers}'")
    devices_cfg.devices[name] = ManusDeviceCfg(retargeters=retargeters, sim_device=src.sim_device, **overrides)


def add_manus_device_from_env(devices_cfg, use_xr: bool) -> None:
    """:func:`add_manus_device` configured from the ``VR_TELEOP_*`` environment variables (module docstring)."""
    env = os.environ
    add_manus_device(
        devices_cfg,
        use_xr=use_xr,
        fingers=env.get("VR_TELEOP_FINGERS", "bench2dex"),
        wrist_mode=env.get("VR_TELEOP_WRIST", "fixed"),
        calibration_json=env.get("VR_TELEOP_CALIB") or None,
        calibrate_rotation=env.get("VR_TELEOP_CALIBRATE_ROTATION", "0") == "1",
        # The robot moves only after the headset START, so the operator can align the hand first.
        auto_start=env.get("VR_TELEOP_AUTO_START", "0") == "1",
        controller_trace=env.get("VR_TELEOP_TRACE") or None,
    )


def show_goal_marker(env_cfg) -> None:
    """Show the goal sphere (red -> green as the tilt condition is met) in the DexVerse pour-style tasks.

    DexVerse v1 baselines show it by default; the v0 task configs turn it off (``pour_show_progress_marker`` and
    ``enable_debug_vis`` are False). This adds the same observation-term visual the config would add with both on,
    to an already built env cfg. Tasks without that marker (no ``pour_show_progress_marker``) are left as they are.
    """
    if not hasattr(env_cfg, "pour_show_progress_marker") or getattr(env_cfg.scene, "success_marker", None) is None:
        return
    if env_cfg.observations.scene_vis is not None and hasattr(env_cfg.observations.scene_vis, "pour_progress_marker_vis"):
        return
    from dexverse.tasks import dexverse_base_env_cfg as base
    from dexverse.tasks import mdp
    from isaaclab.managers import ObservationTermCfg as ObsTerm
    from isaaclab.managers import SceneEntityCfg

    if env_cfg.observations.scene_vis is None:
        env_cfg.observations.scene_vis = base.ObservationsCfg.SceneVisObsCfg()
    env_cfg.observations.scene_vis.pour_progress_marker_vis = ObsTerm(
        func=mdp.pour_progress_marker_vis,
        params={
            "goal_asset_cfg": SceneEntityCfg("success_marker"),
            "object_cfg": SceneEntityCfg("object"),
            "primary_threshold_rad": env_cfg.pour_angle_rad,
            "primary_axis_local": env_cfg.pour_axis_local,
            "primary_world_axis": env_cfg.pour_world_axis,
            "primary_tilt_ge": env_cfg.pour_tilt_ge,
            "plane_threshold_rad": env_cfg.pour_plane_angle_threshold_rad,
            "plane_axis_local": env_cfg.pour_plane_axis_local,
            "plane_normal": env_cfg.pour_plane_normal,
            "radius": env_cfg.pour_progress_marker_radius,
            "opacity": env_cfg.pour_progress_marker_opacity,
            "num_color_steps": env_cfg.pour_progress_marker_num_color_steps,
            "prim_path_prefix": "/Visuals/PourProgressMarker",
        },
    )
    print("[vr_teleop] goal marker on (sphere at the goal, red -> green as the tilt condition is met)")
