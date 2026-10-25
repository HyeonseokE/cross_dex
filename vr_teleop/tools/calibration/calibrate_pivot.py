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
from vr_teleop.tools.calibration.pivot_solver import solve_pivot  # noqa: E402


def main() -> None:
    sim = SimulationContext(SimulationCfg(dt=1 / 90))
    quest = QuestXR(controllers=True)
    sim.reset()
    request_ar_session(simulation_app)
    tracked = TrackedPose()
    samples: list[np.ndarray] = []
    recording, b_prev, frame = False, False, 0
    print(f"[pivot] {args.side}: B/Y = start recording, rotate the hand about the fixed wrist; A/X = finish.")
    while simulation_app.is_running():
        simulation_app.update()
        frame += 1
        quest.poll()
        ctrl = quest.controller(args.side)
        if ctrl is None or not tracked.update(ctrl[0]):
            if frame % 270 == 0:
                print(f"[pivot] waiting for the {args.side} controller (connect the headset in controller mode)")
            continue
        pose, inputs = ctrl
        b_now = inputs[BUTTONS["secondary"]] > 0.5
        if b_now and not b_prev:
            print(f"[pivot] recording{' restarted' if recording else ''}: rotate the hand; A/X = finish")
            recording, samples = True, []
        b_prev = b_now
        if recording:
            samples.append(pose.copy())
        if frame % 180 == 0:
            print(f"[pivot] {args.side}=({pose[0]:+.3f},{pose[1]:+.3f},{pose[2]:+.3f}) samples={len(samples)}")
        if inputs[BUTTONS["primary"]] > 0.5:
            break

    calib = MountCalibrationFile(args.calib)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    npz = calib.path.parent / "history" / f"pivot_{args.side}_{stamp}.npz"
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz, poses=np.asarray(samples).reshape(-1, 7), side=args.side)
    try:
        res = solve_pivot(np.asarray(samples).reshape(-1, 7))
    except ValueError as exc:
        print(f"[pivot] not saved: {exc}. Press B/Y, then rotate the hand for 20-30 s before A/X.")
        return
    r = res["offset_ctrl_m"]
    print(f"[pivot] {args.side}: offset=({r[0]:+.4f}, {r[1]:+.4f}, {r[2]:+.4f}) m |r|={res['offset_norm_m'] * 100:.1f} cm "
          f"RMS={res['rms_mm']:.1f} mm samples={res['n_used']}/{res['n_valid']} "
          f"span={', '.join(f'{s:.0f}' for s in res['rotation_span_deg'])} deg")
    for w in res["warnings"]:
        print(f"[pivot]   WARNING: {w}")
    calib.set_pivot(args.side, res, source=str(npz))
    calib.save()
    print(f"[pivot] saved to {calib.path} (raw poses {npz})")


if __name__ == "__main__":
    main()
    # SimulationApp.close() can hang in an XR session and keep the headset session: exit directly.
    sys.stdout.flush()
    os._exit(0)
