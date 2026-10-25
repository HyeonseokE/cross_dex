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
PANEL_PRIM = "/VrTeleopCalibrationPanel"
HEAD_PRIM = "/_xr/stage/xrCamera"


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
    the headset panel. Panel errors only disable the panel."""

    def __init__(self, tag: str, total: int, lang: str):
        self.tag, self.total = tag, total
        self.korean = lang == "ko" and os.path.isfile(KOREAN_FONT)
        self._panel = True
        self._head: list[str] = []
        self._shown = ""
        self._container = None

    def step(self, n: int, title: str, lines: tuple[str, ...] = (), en: str = "") -> None:
        print(f"\n[{self.tag}] ===== {n}/{self.total} {title} =====", *(f"[{self.tag}]   {line}" for line in lines),
              sep="\n", flush=True)
        self._head = [f"{n}/{self.total}  {title}", *lines] if self.korean else [f"{n}/{self.total}  {en}"]
        self._show(self._head)

    def status(self, text: str, en: str = "", echo: bool = True) -> None:
        if echo:
            print(f"[{self.tag}]   {text}", flush=True)
        self._show([*self._head, text if self.korean else en])

    def linger(self, app, seconds: float = 8.0) -> None:
        """Keep the app (and the final panel) up for a while, so the result can be read in the headset."""
        import time

        end = time.monotonic() + (seconds if self._panel and self._shown else 0.0)
        while time.monotonic() < end and app.is_running():
            app.update()

    def _show(self, lines: list[str]) -> None:
        text = "\n".join(line for line in lines if line)
        if not self._panel or text == self._shown:
            return
        try:
            self._render(text)
            self._shown = text
        except Exception as exc:  # noqa: BLE001 - the panel is a convenience
            self._panel = False
            print(f"[{self.tag}]   (headset panel disabled: {type(exc).__name__}: {exc})", flush=True)

    def _render(self, text: str, font: float = 0.045, max_width: float = 1.3) -> None:
        """Like Isaac Lab's ``show_instruction``, but keeps the line breaks and sizes Hangul (about 1 em wide;
        ``show_instruction`` assumes 0.6 em per character, so Korean overflowed its panel)."""
        import omni.kit.commands
        import omni.usd
        from isaaclab.ui.xr_widgets.instruction_widget import SimpleTextWidget
        from omni.kit.xr.scene_view.utils import UiContainer, WidgetComponent
        from omni.kit.xr.scene_view.utils.spatial_source import SpatialSource
        from pxr import Gf

        def width(line: str) -> float:
            return sum(1.0 if ord(c) > 0x2E80 else 0.55 for c in line) * font

        wrapped = []
        for line in text.split("\n"):
            while width(line) > max_width:  # break at the last space that fits
                cut = max((i for i, c in enumerate(line) if c == " " and width(line[:i]) <= max_width), default=0)
                cut = cut or next(i for i in range(len(line), 0, -1) if width(line[:i]) <= max_width)
                wrapped.append(line[:cut].rstrip())
                line = line[cut:].lstrip()
            wrapped.append(line)
        w = max(min(max(width(line) for line in wrapped) + 2 * font, max_width + 2 * font), 0.6)
        h = (len(wrapped) + 0.6) * 1.25 * font
        if self._container is not None:
            self._container.root.clear()
        stage = omni.usd.get_context().get_stage()
        if stage.GetPrimAtPath(PANEL_PRIM).IsValid():
            stage.RemovePrim(PANEL_PRIM)
        omni.kit.commands.execute("CopyPrim", path_from=HEAD_PRIM, path_to=PANEL_PRIM, exclusive_select=False,
                                  copy_to_introducing_layer=False)
        component = WidgetComponent(SimpleTextWidget, width=w, height=h, resolution_scale=300,
                                    widget_args=["\n".join(wrapped), {"font_size": font, "color": 0xFFFFFFFF}, w])
        # Above the work area, so it does not cover the hands; turns to face the headset.
        self._container = UiContainer(component, space_stack=[
            SpatialSource.new_prim_path_source(PANEL_PRIM),
            SpatialSource.new_translation_source(Gf.Vec3d(0.0, 0.32, -1.5)),
            SpatialSource.new_look_at_camera_source(),
        ])
