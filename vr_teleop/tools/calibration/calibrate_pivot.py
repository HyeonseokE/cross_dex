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
# Accepted for DexVerse run_teleop.sh compatibility; unused here.
parser.add_argument("--teleop_device", type=str, default=None, help=argparse.SUPPRESS)
parser.add_argument("--enable_pinocchio", action="store_true", help=argparse.SUPPRESS)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.calib:
    parser.error("--calib (or $VR_TELEOP_CALIB) is required")
args.xr = True
simulation_app = AppLauncher(args).app

import numpy as np  # noqa: E402
from isaaclab.sim import SimulationCfg, SimulationContext  # noqa: E402

from vr_teleop.devices.metaquest.quest_openxr_isaaclab import BUTTONS, QuestXR, TrackedPose, request_ar_session  # noqa: E402
from vr_teleop.tools.calibration.mount_calibration import MountCalibrationFile  # noqa: E402
from vr_teleop.tools.calibration.pivot_solver import GOOD_RMS_MM, MIN_ROTATION_SPAN_DEG, solve_pivot  # noqa: E402


HAND = {"right": ("오른", "B", "A"), "left": ("왼", "Y", "X")}


def step(n: int, title: str, *lines: str) -> None:
    print(f"\n[피벗] ===== {n}/4 {title} =====", *(f"[피벗]   {line}" for line in lines), sep="\n", flush=True)


def main() -> None:
    hand, start_btn, end_btn = HAND[args.side]
    sim = SimulationContext(SimulationCfg(dt=1 / 90))
    quest = QuestXR(controllers=True)
    sim.reset()
    request_ar_session(simulation_app)
    tracked = TrackedPose()
    samples: list[np.ndarray] = []
    ready, recording, b_prev, frame, t_start = False, False, False, 0, 0.0
    step(1, "헤드셋 연결", "Quest 브라우저에서 Disconnect → Connect → Enter VR (헤드셋은 쓰지 않아도 됨, 켜서 접속만)",
         f"{hand}쪽 컨트롤러가 잡히면 다음 단계로 넘어갑니다.")
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
            step(2, "준비", f"{hand}손 손목(팔뚝 끝)을 책상 모서리에 꽉 고정하세요.",
                 f"{start_btn}를 한 번 누르면 기록이 시작됩니다.")
        pose, inputs = ctrl
        b_now = inputs[BUTTONS["secondary"]] > 0.5
        if b_now and not b_prev:
            step(3, "기록 중" + (" (다시 시작)" if recording else ""),
                 "손목은 고정한 채 위아래 꺾기 · 좌우 꺾기 · 비틀기를 세 방향 모두 크게 20~30초.",
                 f"{end_btn}: 기록 종료 → 계산 → 저장   /   {start_btn}: 처음부터 다시 기록")
            recording, samples, t_start = True, [], time.monotonic()
        b_prev = b_now
        if recording:
            samples.append(pose.copy())
            if frame % 180 == 0:
                print(f"[피벗]   기록 {time.monotonic() - t_start:4.0f}초, 샘플 {len(samples)}개", flush=True)
        if recording and inputs[BUTTONS["primary"]] > 0.5:
            break

    step(4, "계산 · 저장")
    calib = MountCalibrationFile(args.calib)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    npz = calib.path.parent / "history" / f"pivot_{args.side}_{stamp}.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz, poses=np.asarray(samples).reshape(-1, 7), side=args.side)
    try:
        res = solve_pivot(np.asarray(samples).reshape(-1, 7))
    except ValueError as exc:
        print(f"[피벗]   저장 안 함: {exc}", f"[피벗]   다시 실행해서 {start_btn} → 20~30초 돌리기 → {end_btn}", sep="\n")
        return
    r, span = res["offset_ctrl_m"], res["rotation_span_deg"]
    rms_ok, span_ok = res["rms_mm"] <= GOOD_RMS_MM, min(span) >= MIN_ROTATION_SPAN_DEG
    print(f"[피벗]   손목 오프셋 (컨트롤러 기준): ({r[0]:+.4f}, {r[1]:+.4f}, {r[2]:+.4f}) m, 거리 {res['offset_norm_m'] * 100:.1f} cm",
          f"[피벗]   RMS {res['rms_mm']:.1f} mm (기준 {GOOD_RMS_MM:.0f} mm 이하) → {'OK' if rms_ok else '부족: 손목이 움직임, 더 꽉 고정'}",
          f"[피벗]   회전 범위 {', '.join(f'{v:.0f}' for v in span)}° (가장 작은 값 {MIN_ROTATION_SPAN_DEG:.0f}° 이상) → "
          f"{'OK' if span_ok else '부족: 세 방향 모두 더 크게'}",
          f"[피벗]   샘플 {res['n_used']}/{res['n_valid']}개 사용", sep="\n")
    calib.set_pivot(args.side, res, source=str(npz))
    calib.save()
    print(f"[피벗]   저장: {calib.path}", f"[피벗]   원본 포즈: {npz}",
          f"[피벗]   결과: {'정상' if rms_ok and span_ok else '저장은 했지만 다시 하는 것을 권장'}"
          " — 이어서 회전 캘리브레이션(calibrate_rotation.py)을 하세요.", sep="\n", flush=True)


if __name__ == "__main__":
    main()
    # SimulationApp.close() can hang in an XR session and keep the headset session: exit directly.
    sys.stdout.flush()
    os._exit(0)
