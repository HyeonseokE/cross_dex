"""LAD finger codecs for the bimanual pi0.5 space (dexverse_xlvla.bimanual, space "lad"): one frozen LAD contrastive action
model per side (HF Dexverse-CSI/shared-action-spaces dexverse/lad/{right,left}-6hand-ft), wrapped by pi05_space.LadSpace
(used unchanged) with the side's hand model (right: hands.HANDS[h], left: h + "_left").

Joint names of the LAD inputs per side, from hands.json next to model.pt:
  right: joint_names_right.
  left : joint_names_left if present; the left 6-hand models' hands.json carries only the right pair-file names, because
         train_cam --left converts the right pairs joint-wise (build_pairs_bench.left_map) and names each column after the
         left model's joint of the same name (train_cam.hand_models: right_ -> left_ where the left model needs it). The same
         renaming is done here; LadSpace then checks that the names cover exactly the left model's DoFs.

  codecs = lad_codecs("allegro", {"right": <dir or model.pt>, "left": <dir or model.pt>}, {"right": m_r, "left": m_l})
  codecs[side].encode("allegro", fn) / .decode("allegro", z) / .dim
"""
from __future__ import annotations

import json
import os
import tempfile

from dexverse_lad.pi05_space import LadSpace


def _paths(p: str) -> tuple[str, str]:
    d = p if os.path.isdir(p) else os.path.dirname(os.path.abspath(p))
    f = f"{d}/model.pt" if os.path.isdir(p) else p
    return f, d


def side_names(folder: str, side: str) -> dict:
    """{hand: LAD input joint names (sim / URDF names the hand model maps)} for this side."""
    hj = json.load(open(f"{folder}/hands.json"))
    if hj.get("side", side) != side:
        raise ValueError(f"{folder}/hands.json is side {hj.get('side')}, wanted {side}")
    out = {}
    for h, v in hj["hands"].items():
        if f"joint_names_{side}" in v:
            out[h] = v[f"joint_names_{side}"]
        elif side == "left" and "joint_names_right" in v:          # as train_cam.hand_models(left=True)
            out[h] = [n.replace("right_", "left_") for n in v["joint_names_right"]]
    return out


def lad_codecs(hand: str, ckpts: dict, models: dict) -> dict:
    """side -> LadSpace holding only `hand` (keyed by the family name, model = that side's CanonicalHandModel)."""
    out = {}
    for side, p in ckpts.items():
        f, d = _paths(p)
        with tempfile.NamedTemporaryFile("w", suffix=f"_lad_{side}_names.json", delete=False) as t:
            json.dump(side_names(d, side), t)
        try:
            out[side] = LadSpace(f, {hand: models[side]}, names=t.name)
        finally:
            os.unlink(t.name)
        if hand not in out[side].perm:
            raise KeyError(f"{f}: no LAD encoder for {hand}")
    return out
