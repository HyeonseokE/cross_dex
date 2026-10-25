"""Step-by-step guidance for the calibration apps: printed to the terminal and shown on a panel in the headset.

The panel is a camera-facing XR text widget built like Isaac Lab's ``show_instruction`` (used by DexVerse
record_demos). Korean needs a Korean font in Kit's font atlas: :func:`korean_font_kit_args` returns the Kit
arguments for it (pass them to ``AppLauncher`` as ``kit_args``), with only the Hangul syllables of the given
sources in the atlas. Without the font, or with ``lang="en"``, the panel shows the short English text instead.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

KOREAN_FONT = "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"  # apt-get install fonts-nanum


def korean_font_kit_args(*sources: str) -> str:
    """Kit arguments that add ``KOREAN_FONT`` with the Hangul used in ``sources`` (files); "" without the font."""
    if not os.path.isfile(KOREAN_FONT):
        return ""
    text = "".join(Path(s).read_text(encoding="utf-8") for s in sources)
    glyphs = sorted({c for c in text if "가" <= c <= "힣"} | set("·↔→°"))
    region = Path(tempfile.gettempdir()) / "vr_teleop_ko_glyphs.txt"
    region.write_text("".join(glyphs), encoding="utf-8")
    return f"--/app/font/customFontPath={KOREAN_FONT} --/app/font/customRegionFiles=[{region}]"


class StepGuide:
    """``step()`` starts a step (title + what to do), ``status()`` updates its live line; both print and update
    the headset panel. The panel is rebuilt only per step (placed in front of the headset at that moment);
    ``status()`` changes its text in place. Panel errors only disable the panel."""

    def __init__(self, tag: str, total: int, lang: str, head_pose=None):
        self.tag, self.total = tag, total
        self.korean = lang == "ko" and os.path.isfile(KOREAN_FONT)
        self._head_pose = head_pose  # callable -> headset pose [x, y, z, qw, qx, qy, qz] (zeros before connect)
        self._panel = head_pose is not None
        self._head: list[str] = []
        self._container = None
        self._label = None

    def step(self, n: int, title: str, lines: tuple[str, ...] = (), en: str = "") -> None:
        print(f"\n[{self.tag}] ===== {n}/{self.total} {title} =====", *(f"[{self.tag}]   {line}" for line in lines),
              sep="\n", flush=True)
        self._head = [f"{n}/{self.total}  {title}", *lines] if self.korean else [f"{n}/{self.total}  {en}"]
        self._guard(self._build)

    def show(self, lines: list[str]) -> None:
        """Rebuild the panel (in front of the headset now) with these lines and no terminal output."""
        self._head = list(lines)
        self._guard(self._build)

    def status(self, text: str, en: str = "", echo: bool = True) -> None:
        if echo:
            print(f"[{self.tag}]   {text}", flush=True)
        if self._label is not None:
            self._guard(lambda: self._label.set_label_text(self._wrap([*self._head, text if self.korean else en])))

    def linger(self, app, seconds: float = 8.0) -> None:
        """Keep the app (and the final panel) up for a while, so the result can be read in the headset."""
        import time

        end = time.monotonic() + (seconds if self._container is not None else 0.0)
        while time.monotonic() < end and app.is_running():
            app.update()

    def _guard(self, func) -> None:
        if not self._panel:
            return
        try:
            func()
        except Exception as exc:  # noqa: BLE001 - the panel is a convenience
            self._panel, self._label = False, None
            print(f"[{self.tag}]   (headset panel disabled: {type(exc).__name__}: {exc})", flush=True)

    _FONT, _MAX_W = 0.045, 1.3  # panel font height and text width (m)

    def _width(self, line: str) -> float:  # Hangul is about 1 em wide, Latin about 0.55 em
        return sum(1.0 if ord(c) > 0x2E80 else 0.55 for c in line) * self._FONT

    def _wrap(self, lines: list[str]) -> str:
        out = []
        for line in lines:
            while self._width(line) > self._MAX_W:  # break at the last space that fits
                fits = [i for i in range(1, len(line)) if self._width(line[:i]) <= self._MAX_W]
                cut = max((i for i in fits if line[i] == " "), default=fits[-1])
                out.append(line[:cut].rstrip())
                line = line[cut:].lstrip()
            out.append(line)
        return "\n".join(out)

    def _build(self) -> None:
        """Like Isaac Lab's ``show_instruction`` (which assumes 0.6 em per character, so Korean overflowed it),
        but placed from the headset pose without copying the camera prim (a copy changes the selection)."""
        import numpy as np
        from isaaclab.ui.xr_widgets.instruction_widget import SimpleTextWidget
        from omni.kit.xr.scene_view.utils import UiContainer, WidgetComponent
        from omni.kit.xr.scene_view.utils.spatial_source import SpatialSource
        from pxr import Gf

        from ...common.rotation_utils import rot_from_wxyz

        head = np.asarray(self._head_pose(), dtype=np.float64)
        if not np.any(head[:3]):
            return  # headset not connected yet: nothing to see
        fwd = -rot_from_wxyz(head[3:])[:, 2]
        fwd[2] = 0.0
        fwd /= max(np.linalg.norm(fwd), 1e-6)
        pos = head[:3] + 1.5 * fwd + np.array([0.0, 0.0, 0.32])  # above the work area, so it does not cover the hands
        text = self._wrap([*self._head, ""])  # one spare line for status()
        n_lines = text.count("\n") + 1
        w = max(min(max(self._width(line) for line in text.split("\n")), self._MAX_W) + 2 * self._FONT, 0.6)
        h = (n_lines + 0.6) * 1.25 * self._FONT
        if self._container is not None:
            self._container.root.clear()
        component = WidgetComponent(SimpleTextWidget, width=w, height=h, resolution_scale=300,
                                    widget_args=[text, {"font_size": self._FONT, "color": 0xFFFFFFFF}, w])
        self._container = UiContainer(component, space_stack=[
            SpatialSource.new_translation_source(Gf.Vec3d(*pos.tolist())),
            SpatialSource.new_look_at_camera_source(),
        ])
        self._label = None
        comps = getattr(self._container.manipulator, "_ComposableManipulator__components", [])
        if comps and comps[0].component and comps[0].component.widget:
            self._label = comps[0].component.widget
