"""Mount-rotation calibration of a glove-mounted Quest controller (Isaac Sim XR app, passthrough).

The scene is empty with a transparent background, so with passthrough you see your real hand. It shows

* green: after B / Y, a fixed open flat hand skeleton in front of you (palm down, fingers forward), its
  wrist (big sphere) and dotted wrist axes (X magenta, Y cyan, Z white);
* yellow: your wrist (controller pose + pivot offset, from the calibration JSON).

B (right) / Y (left) places the target in front of your view (again: re-place it). Put your real hand on the
green hand (yellow sphere on the green wrist) and press A (right) / X (left), holding still for 0.5 s: the
controller-to-hand rotation (controller frame -> OpenXR wrist frame) is saved as ``mount_rot_ctrl_to_wrist``
for that side and the app quits. A rotation already in the JSON is not used, so a calibration never builds
on an earlier one. Same buttons as calibrate_pivot.py: B / Y starts, A / X finishes.

    <isaac python> calibrate_rotation.py --side right --calib <calibration json>   (run calibrate_pivot.py first)
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Mount-rotation calibration of a glove-mounted Quest controller.")
parser.add_argument("--side", choices=("right", "left"), required=True)
parser.add_argument("--calib", default=os.environ.get("VR_TELEOP_CALIB"), help="calibration JSON (default $VR_TELEOP_CALIB)")
parser.add_argument("--target_forward", type=float, default=0.35, help="target distance in front of the head (m)")
parser.add_argument("--target_down", type=float, default=0.12, help="target height below the head (m)")
parser.add_argument("--panel_lang", choices=("ko", "en"), default="ko", help="language of the headset guide panel")
parser.add_argument("--no_passthrough", action="store_true", help="draw a ground and sky instead of a transparent background")
# Accepted for DexVerse run_teleop.sh compatibility; unused here.
parser.add_argument("--teleop_device", type=str, default=None, help=argparse.SUPPRESS)
parser.add_argument("--enable_pinocchio", action="store_true", help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.calib:
    parser.error("--calib (or $VR_TELEOP_CALIB) is required")
args.xr = True
if args.panel_lang == "ko":
    from vr_teleop.tools.calibration import step_guide

    args.kit_args = f"{args.kit_args or ''} {step_guide.korean_font_kit_args(__file__)}".strip()
simulation_app = AppLauncher(args).app

import isaaclab.sim as sim_utils  # noqa: E402
import numpy as np  # noqa: E402
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

from vr_teleop.common.openxr_hand import flat_hand_template, palm_down_frame, place_hand, skeleton_points  # noqa: E402
from vr_teleop.common.rotation_utils import angle_deg, mean_rotation, rot_from_wxyz  # noqa: E402
from vr_teleop.devices.metaquest.quest_openxr_isaaclab import BUTTONS, QuestXR, TrackedPose, request_ar_session  # noqa: E402
from vr_teleop.teleop.hand_mounted_controller import HandMountedController  # noqa: E402
from vr_teleop.tools.calibration.mount_calibration import MountCalibrationFile, mount_rotation  # noqa: E402
from vr_teleop.tools.calibration.step_guide import StepGuide  # noqa: E402

CAPTURE_FRAMES = 45  # ~0.5 s at 90 Hz
TARGET_AXIS_COLORS = ((1.0, 0.0, 1.0), (0.0, 0.9, 1.0), (1.0, 1.0, 1.0))  # X magenta, Y cyan, Z white
AXIS_LENGTH, AXIS_DOTS = 0.12, 10


def spheres(path: str, colors_radii: dict[str, tuple[tuple[float, float, float], float]]) -> VisualizationMarkers:
    return VisualizationMarkers(VisualizationMarkersCfg(prim_path=path, markers={
        name: sim_utils.SphereCfg(radius=r, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c))
        for name, (c, r) in colors_radii.items()}))


def draw_skeleton(markers: VisualizationMarkers, joints: dict) -> None:
    points, is_bone = skeleton_points(joints)
    markers.visualize(translations=points, marker_indices=is_bone.astype(int))


def draw_dotted_axes(markers: VisualizationMarkers, pos: np.ndarray, rot: np.ndarray) -> None:
    steps = np.linspace(AXIS_LENGTH / AXIS_DOTS, AXIS_LENGTH, AXIS_DOTS)
    points = np.concatenate([pos + steps[:, None] * rot[:, i] for i in range(3)])
    markers.visualize(translations=points, marker_indices=np.repeat(np.arange(3), AXIS_DOTS))


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


HAND = {"right": ("오른", "B", "A"), "left": ("왼", "Y", "X")}
GOOD_DIST_CM, GOOD_SPREAD_DEG = 3.0, 3.0  # wrist-to-target distance / hand motion while capturing


def main() -> None:
    hand, start_btn, end_btn = HAND[args.side]
    guide = StepGuide("회전", 4, args.panel_lang)
    calib = MountCalibrationFile(args.calib)
    if args.side not in calib.data:
        raise SystemExit(f"[회전] {args.calib}에 {hand}손 피벗 값이 없습니다: calibrate_pivot.py를 먼저 하세요")
    mounted = HandMountedController(offsets={args.side: calib.offset(args.side)})  # mount rotation: identity
    print(f"[회전] {hand}손 피벗 오프셋 {np.round(mounted.offset[args.side], 4).tolist()} m 사용")

    sim = SimulationContext(SimulationCfg(dt=1 / 90))
    setup_scene()
    quest = QuestXR(controllers=True)
    green = (0.1, 0.9, 0.2)
    target_skel = spheres("/Visuals/calib_target_hand", {"joint": (green, 0.007), "bone": (green, 0.003)})
    target_wrist = spheres("/Visuals/calib_target_wrist", {"wrist": (green, 0.02)})
    target_axes = spheres("/Visuals/calib_target_axes", {a: (c, 0.004) for a, c in zip("xyz", TARGET_AXIS_COLORS)})
    live_wrist = spheres("/Visuals/calib_live_wrist", {"wrist": ((1.0, 0.85, 0.0), 0.02)})
    sim.reset()
    request_ar_session(simulation_app)

    tracked = TrackedPose()
    template = flat_hand_template(args.side)
    target, capture = None, []
    ready, a_prev, b_prev, frame = False, False, False, 0
    guide.step(1, "헤드셋 연결", ("헤드셋을 쓰고 Quest 브라우저에서 Disconnect → Connect → AR(패스스루)로 접속",
                                f"{hand}쪽 컨트롤러가 잡히면 다음 단계로 넘어갑니다."),
               en="Rotation: connect the headset in AR (passthrough)")
    while simulation_app.is_running():
        simulation_app.update()
        frame += 1
        quest.poll()
        ctrl = quest.controller(args.side)
        if ctrl is None or not tracked.update(ctrl[0]):
            if frame % 270 == 0:
                print(f"[회전]   ... {hand}쪽 컨트롤러 대기 중 (헤드셋 접속 확인)", flush=True)
            continue
        if not ready:
            ready = True
            guide.step(2, "목표 손 표시", ("노란 점 = 내 손목.",
                                           f"{start_btn}를 한 번 누르면 내 앞에 초록 손(손바닥 아래, 손가락 앞)이 나타납니다."),
                       en=f"Yellow = your wrist. Press {start_btn} to show the green target hand")
        pose, inputs = ctrl
        wrist_pos, _ = mounted.wrist(args.side, pose)
        live_wrist.visualize(translations=wrist_pos[None])

        b_now = inputs[BUTTONS["secondary"]] > 0.5
        if b_now and not b_prev and np.any(quest.head()[:3]):
            if target is None:
                guide.step(3, "손 맞추기", (f"실제 {hand}손을 초록 손에 겹치고, 노란 점을 초록 큰 점에 맞추세요.",
                                            f"맞으면 {end_btn}를 한 번 누르고 0.5초 동안 움직이지 마세요 → 저장 후 종료.",
                                            f"{start_btn}: 초록 손을 지금 보는 방향 앞으로 다시 놓기"),
                           en=f"Put your hand on the green hand, yellow dot on the big green dot. "
                              f"{end_btn}: capture (hold still 0.5 s)  {start_btn}: re-place")
            else:
                print("[회전]   초록 손을 다시 놓았습니다.", flush=True)
            target = place_target(quest.head())
        b_prev = b_now
        if target is None:
            a_prev = inputs[BUTTONS["primary"]] > 0.5
            continue

        draw_skeleton(target_skel, place_hand(template, target[0], target[1]))
        target_wrist.visualize(translations=target[0][None])
        draw_dotted_axes(target_axes, target[0], target[1])
        if frame % 90 == 0 and not capture:
            dist = np.linalg.norm(wrist_pos - target[0]) * 100
            guide.status(f"손목 ↔ 초록 손목 거리 {dist:.1f} cm", en=f"wrist to target: {dist:.1f} cm")

        a_now = inputs[BUTTONS["primary"]] > 0.5
        if a_now and not a_prev and not capture:
            capture = [mounted.ctrl_rot[args.side]]
            guide.step(4, "캡처", ("움직이지 마세요 (0.5초) ...",), en="Hold still (0.5 s) ...")
        elif capture:
            capture.append(mounted.ctrl_rot[args.side])
            if len(capture) >= CAPTURE_FRAMES:
                r_ctrl = mean_rotation(capture)
                spread = max(angle_deg(c @ r_ctrl.T) for c in capture)
                rot = mount_rotation(r_ctrl, target[1])
                calib = MountCalibrationFile(args.calib)
                calib.set_mount_rot(args.side, rot, "calibrate_rotation.py (target hand: palm down, fingers forward)")
                calib.save()
                dist = np.linalg.norm(wrist_pos - target[0]) * 100
                ok = dist <= GOOD_DIST_CM and spread <= GOOD_SPREAD_DEG
                guide.step(4, f"저장: {'정상' if ok else '저장은 했지만 다시 하는 것을 권장'}", (
                    f"손목 ↔ 초록 손목 거리 {dist:.1f} cm (기준 {GOOD_DIST_CM:.0f} cm 이하) → "
                    f"{'OK' if dist <= GOOD_DIST_CM else '멀다: 노란 점을 초록 큰 점에 더 가깝게'}",
                    f"캡처 중 손 움직임 {spread:.1f}° (기준 {GOOD_SPREAD_DEG:.0f}° 이하) → "
                    f"{'OK' if spread <= GOOD_SPREAD_DEG else '흔들림: 더 가만히'}",
                    f"컨트롤러 → 손 회전 {angle_deg(rot):.0f}°. 결과는 텔레옵에서 확인하세요."),
                    en=f"Saved. distance {dist:.1f} cm {'OK' if dist <= GOOD_DIST_CM else '(redo: closer)'}, "
                       f"motion {spread:.1f} deg {'OK' if spread <= GOOD_SPREAD_DEG else '(redo: hold still)'}")
                print(f"[회전]   저장: {calib.path}", flush=True)
                guide.linger(simulation_app)
                break
        a_prev = a_now


if __name__ == "__main__":
    main()
    # SimulationApp.close() can hang in an XR session and keep the headset session: exit directly.
    sys.stdout.flush()
    os._exit(0)
