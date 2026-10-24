"""Mount-rotation calibration of a glove-mounted Quest controller (Isaac Sim XR app, passthrough).

The scene is empty with a transparent background, so with passthrough you see your real hand. It shows

* green: a fixed open flat hand skeleton in front of you (palm down, fingers forward), its wrist (big
  sphere) and dotted wrist axes (X magenta, Y cyan, Z white);
* yellow: your wrist (controller pose + pivot offset, from the calibration JSON);
* red: after a capture, your hand (live glove finger shape) with the new rotation and its wrist axes
  (frame marker: X red, Y green, Z blue), to check the result.

Put your real hand on the green hand (yellow sphere on the green wrist) and press A (right) / X (left),
holding still for 0.5 s: the controller-to-hand rotation (controller frame -> OpenXR wrist frame) is
saved as ``mount_rot_ctrl_to_wrist`` for that side. A rotation already in the JSON is not used, so a
calibration never builds on an earlier one. B / Y: move the target in front of your view; hold 2 s: quit.

    <isaac python> calibrate_rotation.py --side right --calib <calibration json>   (run calibrate_pivot.py first)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Mount-rotation calibration of a glove-mounted Quest controller.")
parser.add_argument("--side", choices=("right", "left"), required=True)
parser.add_argument("--calib", default=os.environ.get("VR_TELEOP_CALIB"), help="calibration JSON (default $VR_TELEOP_CALIB)")
parser.add_argument("--host", default="127.0.0.1", help="Manus bridge host")
parser.add_argument("--port", type=int, default=47000, help="Manus bridge port")
parser.add_argument("--target_forward", type=float, default=0.35, help="target distance in front of the head (m)")
parser.add_argument("--target_down", type=float, default=0.12, help="target height below the head (m)")
parser.add_argument("--no_passthrough", action="store_true", help="draw a ground and sky instead of a transparent background")
# Accepted for DexVerse run_teleop.sh compatibility; unused here.
parser.add_argument("--teleop_device", type=str, default=None, help=argparse.SUPPRESS)
parser.add_argument("--enable_pinocchio", action="store_true", help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.calib:
    parser.error("--calib (or $VR_TELEOP_CALIB) is required")
args.xr = True
simulation_app = AppLauncher(args).app

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.markers.config import FRAME_MARKER_CFG  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

from vr_teleop.common.openxr_hand import flat_hand_template, palm_down_frame, place_hand, skeleton_points  # noqa: E402
from vr_teleop.common.rotation_utils import angle_deg, mean_rotation, rot_from_wxyz, wxyz_from_rot  # noqa: E402
from vr_teleop.devices.manus.bridge.manus_bridge_client import ManusBridgeClient  # noqa: E402
from vr_teleop.devices.manus.glove_skeleton import GloveWristFrame, glove_to_openxr_joints  # noqa: E402
from vr_teleop.devices.metaquest.quest_openxr_isaaclab import BUTTONS, QuestXR, TrackedPose, request_ar_session  # noqa: E402
from vr_teleop.teleop.hand_mounted_controller import HandMountedController  # noqa: E402
from vr_teleop.tools.calibration.mount_calibration import MountCalibrationFile, mount_rotation  # noqa: E402

CAPTURE_FRAMES = 45  # ~0.5 s at 90 Hz
QUIT_HOLD_S = 2.0
TARGET_AXIS_COLORS = ((1.0, 0.0, 1.0), (0.0, 0.9, 1.0), (1.0, 1.0, 1.0))  # X magenta, Y cyan, Z white
AXIS_LENGTH, AXIS_DOTS = 0.12, 10


def spheres(path: str, colors_radii: dict[str, tuple[tuple[float, float, float], float]]) -> VisualizationMarkers:
    return VisualizationMarkers(VisualizationMarkersCfg(prim_path=path, markers={
        name: sim_utils.SphereCfg(radius=r, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c))
        for name, (c, r) in colors_radii.items()}))


def skeleton_markers(path: str, color) -> VisualizationMarkers:
    return spheres(path, {"joint": (color, 0.007), "bone": (color, 0.003)})


def draw_skeleton(markers: VisualizationMarkers, joints: dict) -> None:
    points, is_bone = skeleton_points(joints)
    markers.visualize(translations=points, marker_indices=is_bone.astype(int))


def draw_dotted_axes(markers: VisualizationMarkers, pos: np.ndarray, rot: np.ndarray) -> None:
    steps = np.linspace(AXIS_LENGTH / AXIS_DOTS, AXIS_LENGTH, AXIS_DOTS)
    points = np.concatenate([pos + steps[:, None] * rot[:, i] for i in range(3)])
    markers.visualize(translations=points, marker_indices=np.repeat(np.arange(3), AXIS_DOTS))


def frame_marker(path: str) -> VisualizationMarkers:
    cfg = FRAME_MARKER_CFG.copy()
    cfg.markers["frame"].scale = (0.12, 0.12, 0.12)
    return VisualizationMarkers(cfg.replace(prim_path=path))


def place_target(head: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Target wrist in front of / below the head; palm down, fingers along the horizontal view direction."""
    fwd = -rot_from_wxyz(head[3:])[:, 2]
    rot = palm_down_frame(fwd)
    fwd_h = -rot[:, 2]
    return head[:3] + args.target_forward * fwd_h - np.array([0.0, 0.0, args.target_down]), rot


def setup_scene() -> None:
    if args.no_passthrough:
        sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
        sim_utils.DomeLightCfg(intensity=2500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2500.0))
        return
    # Passthrough: no ground / sky and a zero-alpha background, so the headset shows the real room.
    import carb

    settings = carb.settings.get_settings()
    settings.set_bool("/rtx/post/backgroundZeroAlpha/enabled", True)
    settings.set_bool("/rtx/post/backgroundZeroAlpha/backgroundComposite", False)
    sim_utils.DistantLightCfg(intensity=3000.0).func("/World/light", sim_utils.DistantLightCfg(intensity=3000.0))


def main() -> None:
    calib = MountCalibrationFile(args.calib)
    if args.side not in calib.data:
        raise SystemExit(f"{args.calib} has no '{args.side}' entry: run calibrate_pivot.py for this hand first")
    mounted = HandMountedController(offsets={args.side: calib.offset(args.side)})  # mount rotation: identity
    print(f"[rot] {args.side}: pivot offset {np.round(mounted.offset[args.side], 4).tolist()} m")

    sim = SimulationContext(SimulationCfg(dt=1 / 90))
    setup_scene()
    quest = QuestXR(controllers=True)
    target_skel = skeleton_markers("/Visuals/calib_target_hand", (0.1, 0.9, 0.2))
    target_wrist = spheres("/Visuals/calib_target_wrist", {"wrist": ((0.1, 0.9, 0.2), 0.02)})
    target_axes = spheres("/Visuals/calib_target_axes", {a: (c, 0.004) for a, c in zip("xyz", TARGET_AXIS_COLORS)})
    live_skel = skeleton_markers("/Visuals/calib_live_hand", (0.95, 0.15, 0.1))
    live_wrist = spheres("/Visuals/calib_live_wrist", {"wrist": ((1.0, 0.85, 0.0), 0.02)})
    live_axes = frame_marker("/Visuals/calib_live_axes")
    sim.reset()
    request_ar_session(simulation_app)
    live_skel.set_visibility(False)
    live_axes.set_visibility(False)

    glove, glove_frame, tracked = ManusBridgeClient(args.host, args.port), GloveWristFrame(), TrackedPose()
    template = flat_hand_template(args.side)
    target, calibrated, capture = None, False, []
    a_prev, b_prev, b_since, frame = False, False, None, 0
    print("[rot] Yellow = your wrist. Put it on the green wrist, hold your hand like the green hand, then A/X. "
          "B/Y: re-place the target, hold 2 s to quit.")
    while simulation_app.is_running():
        simulation_app.update()
        frame += 1
        quest.poll()
        ctrl, gloves = quest.controller(args.side), glove.gloves()
        if ctrl is None or args.side not in gloves or not tracked.update(ctrl[0]):
            if frame % 270 == 0:
                print(f"[rot] waiting: controller={'ok' if ctrl is not None else 'none'} glove={glove.status()}")
            continue
        pose, inputs = ctrl
        if target is None:
            head = quest.head()
            if not np.any(head[:3]):
                continue
            target = place_target(head)
            print(f"[rot] target placed at {np.round(target[0], 3).tolist()}")

        wrist_pos, wrist_rot = mounted.wrist(args.side, pose)
        draw_skeleton(target_skel, place_hand(template, target[0], target[1]))
        target_wrist.visualize(translations=target[0][None])
        draw_dotted_axes(target_axes, target[0], target[1])
        live_wrist.visualize(translations=wrist_pos[None])
        kp, wq = gloves[args.side]
        frame_g = glove_frame(kp, args.side, wq)
        if calibrated and frame_g is not None and np.isfinite(kp).all():
            draw_skeleton(live_skel, glove_to_openxr_joints(kp, frame_g, wrist_pos, wrist_rot))
            live_axes.visualize(translations=wrist_pos[None], orientations=wxyz_from_rot(wrist_rot)[None])
        if frame % 90 == 0:
            msg = f"[rot] wrist to target wrist: {np.linalg.norm(wrist_pos - target[0]) * 100:.1f} cm"
            if calibrated:
                msg += f", hand vs target rotation {angle_deg(target[1] @ wrist_rot.T):.0f} deg"
            print(msg)

        a_now = inputs[BUTTONS["primary"]] > 0.5
        if a_now and not a_prev and not capture:
            capture = [mounted.ctrl_rot[args.side]]
            print("[rot] capturing, hold still ...")
        elif capture:
            capture.append(mounted.ctrl_rot[args.side])
            if len(capture) >= CAPTURE_FRAMES:
                r_ctrl = mean_rotation(capture)
                spread = max(angle_deg(c @ r_ctrl.T) for c in capture)
                rot = mount_rotation(r_ctrl, target[1])
                mounted.mount_rot[args.side] = rot
                calib = MountCalibrationFile(args.calib)
                calib.set_mount_rot(args.side, rot, "calibrate_rotation.py (target hand: palm down, fingers forward)")
                calib.save()
                print(f"[rot] saved controller->hand rotation to {calib.path} ({angle_deg(rot):.0f} deg; "
                      f"hand moved up to {spread:.1f} deg while capturing)")
                calibrated, capture = True, []
                live_skel.set_visibility(True)
                live_axes.set_visibility(True)
        a_prev = a_now

        b_now = inputs[BUTTONS["secondary"]] > 0.5
        if b_now and not b_prev:
            b_since = time.monotonic()
        if b_now and b_since is not None and time.monotonic() - b_since > QUIT_HOLD_S:
            print("[rot] quit")
            break
        if not b_now and b_prev:
            target = place_target(quest.head())
            print(f"[rot] target moved to {np.round(target[0], 3).tolist()}")
        b_prev = b_now


if __name__ == "__main__":
    main()
    # SimulationApp.close() can hang in an XR session and keep the headset session: exit directly.
    sys.stdout.flush()
    os._exit(0)
