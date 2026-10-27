"""Glove-mounted controller calibration: file format and mount rotation (numpy only).

Two fixed transforms take a hand-mounted controller's pose to the wrist (teleop/hand_mounted_controller.py):

* pivot offset ``offset_ctrl_m``: the wrist rotation centre in the controller frame (``pivot_solver.py``);
* mount rotation ``mount_rot_ctrl_to_wrist``: controller frame -> OpenXR wrist frame (:func:`mount_rotation`).

Both live per hand side in one JSON (:class:`MountCalibrationFile`), written by ``calibrate_pivot.py`` /
``calibrate_rotation.py`` and loaded by the teleop.
"""

from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path

import numpy as np

def mount_rotation(r_ctrl: np.ndarray, r_hand: np.ndarray) -> np.ndarray:
    """Controller frame -> OpenXR wrist frame, from a controller orientation ``r_ctrl`` taken while the
    hand had the known orientation ``r_hand`` (both 3x3 in the same world)."""
    return r_ctrl.T @ r_hand


def _now() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


class MountCalibrationFile:
    """Per-side controller calibration JSON, read by the teleop and written by the calibration tools::

        {"description": ..., "right": {"offset_ctrl_m": [x, y, z], "rms_mm": ..., ...,
                                       "mount_rot_ctrl_to_wrist": 3x3, "mount_rot_date": ..., ...}, "left": {...}}

    Writing keeps the other side and backs the previous file up into ``history/`` next to it.
    """

    DESCRIPTION = ("Quest controller mounted on the back of the hand. wrist position = t_ctrl + R_ctrl @ "
                   "offset_ctrl_m; wrist frame (OpenXR) = R_ctrl @ mount_rot_ctrl_to_wrist")

    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()  # follow a current.json link: write and back up next to the real file
        self.data: dict = json.loads(self.path.read_text()) if self.path.exists() else {"description": self.DESCRIPTION}

    def offset(self, side: str) -> np.ndarray:
        entry = self.data.get(side)
        return np.zeros(3) if entry is None else np.asarray(entry["offset_ctrl_m"], dtype=np.float64)

    def mount_rot(self, side: str) -> np.ndarray | None:
        entry = self.data.get(side) or {}
        rot = entry.get("mount_rot_ctrl_to_wrist")
        return None if rot is None else np.asarray(rot, dtype=np.float64)

    def set_pivot(self, side: str, result: dict, source: str | None = None) -> None:
        entry = self.data.setdefault(side, {})
        entry.update(result)
        entry["pivot_date"], entry["pivot_source"] = _now(), source

    def set_mount_rot(self, side: str, rot: np.ndarray, method: str) -> None:
        entry = self.data.setdefault(side, {"offset_ctrl_m": [0.0, 0.0, 0.0]})
        entry["mount_rot_ctrl_to_wrist"] = np.asarray(rot).round(6).tolist()
        entry["mount_rot_date"], entry["mount_rot_method"] = _now(), method

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            history = self.path.parent / "history"
            history.mkdir(exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            shutil.copy2(self.path.resolve(), history / f"{self.path.stem}_{stamp}.json")
        self.path.write_text(json.dumps(self.data, indent=2) + "\n")
