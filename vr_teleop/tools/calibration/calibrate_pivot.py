"""Pivot calibration of a glove-mounted Quest controller (Isaac Sim XR app): record + solve + save.

Rest the wrist on a fixed point (table edge), press B (right) / Y (left) to start recording and rotate the
hand in every direction (flex / extend, side to side, twist) for 20-30 s. Pressing B / Y again restarts the
recording. A (right) / X (left) finishes (no buttons to hold, so the other hand does not cover the
controller's tracking LEDs): the offset of the
wrist rotation centre in the controller frame is solved (``pivot_solver.py``) and saved for that side
into the calibration JSON (``mount_calibration.py``; the other side and the mount rotation are kept,
the previous file goes to ``history/``). The raw poses are kept next to it as ``history/pivot_<side>_<time>.npz``.

    <isaac python> calibrate_pivot.py --side right --calib <calibration json>
    (DexVerse: scripts/teleop_tools/run_teleop.sh <this file> --side right --calib ...)

A good fit has RMS <= 5 mm and the smallest rotation span >= 30 deg.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Pivot calibration of a glove-mounted Quest controller.")
parser.add_argument("--side", choices=("right", "left"), required=True)
parser.add_argument("--calib", default=os.environ.get("VR_TELEOP_CALIB"), help="calibration JSON (default $VR_TELEOP_CALIB)")
parser.add_argument("--panel_lang", choices=("ko", "en"), default="ko", help="language of the headset guide panel")
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

import numpy as np  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

from vr_teleop.devices.metaquest.quest_openxr_isaaclab import BUTTONS, QuestXR, TrackedPose, request_ar_session  # noqa: E402
from vr_teleop.tools.calibration.mount_calibration import MountCalibrationFile  # noqa: E402
from vr_teleop.tools.calibration.pivot_solver import GOOD_RMS_MM, MIN_ROTATION_SPAN_DEG, solve_pivot  # noqa: E402
from vr_teleop.tools.calibration.step_guide import StepGuide  # noqa: E402


HAND = {"right": ("오른", "B", "A"), "left": ("왼", "Y", "X")}


def main() -> None:
    hand, start_btn, end_btn = HAND[args.side]
    guide = StepGuide("피벗", 4, args.panel_lang)
    sim = SimulationContext(SimulationCfg(dt=1 / 90))
    quest = QuestXR(controllers=True)
    sim.reset()
    request_ar_session(simulation_app)
    tracked = TrackedPose()
    samples: list[np.ndarray] = []
    ready, recording, b_prev, frame, t_start = False, False, False, 0, 0.0
    guide.step(1, "헤드셋 연결", ("Quest 브라우저에서 Disconnect → Connect → Enter VR (헤드셋은 쓰지 않아도 됨, 켜서 접속만)",
                                f"{hand}쪽 컨트롤러가 잡히면 다음 단계로 넘어갑니다."),
               en="Pivot: connect the headset")
    while simulation_app.is_running():
        simulation_app.update()
        frame += 1
        quest.poll()
        ctrl = quest.controller(args.side)
        if ctrl is None or not tracked.update(ctrl[0]):
            if frame % 270 == 0:
                print(f"[피벗]   ... {hand}쪽 컨트롤러 대기 중 (헤드셋 접속 확인)", flush=True)
            continue
        if not ready:
            ready = True
            guide.step(2, "준비", (f"{hand}손 손목(팔뚝 끝)을 책상 모서리에 꽉 고정하세요.",
                                   f"{start_btn}를 한 번 누르면 기록이 시작됩니다."),
                       en=f"Fix the wrist on the desk edge, then press {start_btn} to record")
        pose, inputs = ctrl
        b_now = inputs[BUTTONS["secondary"]] > 0.5
        if b_now and not b_prev:
            guide.step(3, "기록 중" + (" (다시 시작)" if recording else ""),
                       ("손목은 고정한 채 위아래 꺾기 · 좌우 꺾기 · 비틀기를 세 방향 모두 크게 20~30초.",
                        f"{end_btn}: 기록 종료 → 계산 → 저장   /   {start_btn}: 처음부터 다시 기록"),
                       en=f"Recording: bend, tilt, twist the hand 20-30 s | {end_btn}: finish  {start_btn}: restart")
            recording, samples, t_start = True, [], time.monotonic()
        b_prev = b_now
        if recording:
            samples.append(pose.copy())
            if frame % 90 == 0:
                sec = time.monotonic() - t_start
                guide.status(f"기록 {sec:.0f}초, 샘플 {len(samples)}개", en=f"{sec:.0f} s, {len(samples)} samples",
                             echo=frame % 180 == 0)
        if recording and inputs[BUTTONS["primary"]] > 0.5:
            break

    calib = MountCalibrationFile(args.calib)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    npz = calib.path.parent / "history" / f"pivot_{args.side}_{stamp}.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz, poses=np.asarray(samples).reshape(-1, 7), side=args.side)
    try:
        res = solve_pivot(np.asarray(samples).reshape(-1, 7))
    except ValueError as exc:
        guide.step(4, "저장 안 함", (str(exc), f"다시 실행해서 {start_btn} → 20~30초 돌리기 → {end_btn}"),
                   en=f"Not saved: too few samples. Run again: {start_btn}, rotate 20-30 s, {end_btn}")
        guide.linger(simulation_app)
        return
    r, span = res["offset_ctrl_m"], res["rotation_span_deg"]
    rms_ok, span_ok = res["rms_mm"] <= GOOD_RMS_MM, min(span) >= MIN_ROTATION_SPAN_DEG
    calib.set_pivot(args.side, res, source=str(npz))
    calib.save()
    verdict = "정상" if rms_ok and span_ok else "저장은 했지만 다시 하는 것을 권장"
    guide.step(4, f"계산 · 저장: {verdict}", (
        f"RMS {res['rms_mm']:.1f} mm (기준 {GOOD_RMS_MM:.0f} mm 이하) → {'OK' if rms_ok else '부족: 손목이 움직임, 더 꽉 고정'}",
        f"회전 범위 {', '.join(f'{v:.0f}' for v in span)}° (가장 작은 값 {MIN_ROTATION_SPAN_DEG:.0f}° 이상) → "
        f"{'OK' if span_ok else '부족: 세 방향 모두 더 크게'}",
        f"손목 오프셋 (컨트롤러 기준): ({r[0]:+.4f}, {r[1]:+.4f}, {r[2]:+.4f}) m, 거리 {res['offset_norm_m'] * 100:.1f} cm",
        "이어서 회전 캘리브레이션(calibrate_rotation.py)을 하세요."),
        en=f"Saved. RMS {res['rms_mm']:.1f} mm {'OK' if rms_ok else '(redo: keep wrist fixed)'}, "
           f"span {min(span):.0f} deg {'OK' if span_ok else '(redo: rotate more)'}")
    print(f"[피벗]   샘플 {res['n_used']}/{res['n_valid']}개 사용", f"[피벗]   저장: {calib.path}", f"[피벗]   원본 포즈: {npz}",
          sep="\n", flush=True)
    guide.linger(simulation_app)


if __name__ == "__main__":
    main()
    # SimulationApp.close() can hang in an XR session and keep the headset session: exit directly.
    sys.stdout.flush()
    os._exit(0)
