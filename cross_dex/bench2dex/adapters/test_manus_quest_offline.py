"""Offline check of the Bench2Dex Manus + controller adapter (no headset, no Isaac app).

Needs the fake Manus bridge (``cross_dex/teleop/tools/run_manus_bridge.sh --fake --port 47001``) and must run from the
bench2dex repo (``teleop.xr_hand_source`` on sys.path). A fake QuestXR feeds a slowly moving controller pose; the
source must produce tracked HandSamples whose wrist follows the controller through the calibration offset, and the
controller must retarget them to finger joints for the given hand types.

    cd /workspace/bench2dex && python /workspace/cross_dex/cross_dex/bench2dex/adapters/test_manus_quest_offline.py wuji shadow
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.environ.get("B2D_ROOT", os.getcwd()))  # bench2dex repo: teleop.xr_hand_source / xr_teleop

from cross_dex.bench2dex.adapters.manus_quest_hand_source import ManusQuestHandSource
from cross_dex.bench2dex.adapters.manus_quest_teleop import make_controller_class


class FakeQuest:
    """QuestXR stand-in: both controllers tracked, moving 1 mm per poll along +x, fixed orientation."""

    def __init__(self):
        self.n = 0
        self.callbacks = {}
        self.device = None

    def poll(self):
        self.n += 1

    def controller(self, side):
        x = 0.3 + 0.001 * self.n
        y = -0.2 if side == "right" else 0.2
        pose = np.array([x, y, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return pose, np.zeros(8)

    def add_callback(self, key, fn):
        self.callbacks[key] = fn

    def reset(self):
        pass


def main():
    hands = sys.argv[1:] or ["wuji"]
    src = ManusQuestHandSource(port=47001, quest=FakeQuest(), stream_log_every=0)
    assert src.start()
    samples = {}
    for _ in range(40):  # TrackedPose needs two differing poses, the bridge a few frames
        samples = src.poll()
        time.sleep(0.03)
    assert samples, "no HandSample: is the fake bridge running on 47001?"
    for side, s in samples.items():
        assert s.joints.shape == (26, 7), s.joints.shape
        ctrl = src._quest.controller(side)[0]
        d = np.linalg.norm(s.wrist_pos - ctrl[:3])
        print(f"@@ source {side}: tracked={s.tracked} lost_polls={s.lost_polls} wrist={np.round(s.wrist_pos, 3).tolist()} "
              f"| controller->wrist offset {d * 100:.1f} cm (calibration offset {np.round(src._ctrl.offset[side], 3).tolist()})")
    Controller = make_controller_class()
    for hand_type in hands:
        src2 = ManusQuestHandSource(port=47001, quest=FakeQuest())
        ctrl = Controller(src2, hand_type, enable_right=True, enable_left=True, wrist_smoothing=0.5)
        assert ctrl.start()
        q = None
        for _ in range(40):
            q = ctrl.step() or q
            time.sleep(0.03)
        assert q, f"{hand_type}: no finger joints"
        for side, qs in q.items():
            names = ctrl._bridge(side).joint_names
            print(f"@@ {hand_type} {side}: {len(qs)} joints ({names[:3]}...), wrist_pose ok={ctrl.wrist_pose(side) is not None}, "
                  f"q range [{qs.min():.2f}, {qs.max():.2f}]")
        ctrl.stop()
    print("@@ OFFLINE TEST OK")


if __name__ == "__main__":
    main()
