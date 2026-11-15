"""DexJoCo-X hands for the XL-VLA latent (2026-11-15), registered into ``hands.HANDS`` by ``hands.py``.

All seven hands, right and left, are URDFs written from the DexJoCo-X MuJoCo models themselves
(``dexjoco_mjcf2urdf.py`` -> ``dexjoco_urdf/<hand>_<side>.urdf`` + ``.json``): body offsets, hinge axes, joint ranges and the
coupled joints (joint equalities -> mimic) are copied from the compiled model, so every model takes DexJoCo-X data as is
(actuator order, no sign / order conversion) and its link positions equal MuJoCo's (max 0.0003 mm over random poses).
Fingertips = the fingertip geoms of the MuJoCo model.

Why not the DexVerse URDFs of the four shared hands (they were used first): they differ from DexJoCo-X's models by
2-10 mm at the fingertips, DexJoCo-X's left Allegro flips the three abduction joints, its left LinkerHand flips the four
mcp_roll joints, and its left LEAP is not a mirror of the right LEAP under any joint sign map (checks of 2026-11-15).

  right  dj_allegro,dj_inspire,dj_leap,dj_sharpa,dj_wuji,dj_linkerhand,dj_xhand
  left   the same names + _left

Dataset joint names (``data_name`` keys) = the names in the LAD / chunk pair files of the DexJoCo-X benchmark: the
DexVerse / retarget joint names of each hand listed in DexJoCo-X actuator order (``dexjoco_urdf/dataset_joint_names.json``).
"""

from __future__ import annotations

import json
import os

G = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dexjoco_urdf")
FILE = {"allegro": "allegro", "inspire": "inspire", "leap": "leap", "sharpa": "sharpa_wave", "wuji": "wuji",
        "linkerhand": "linkerhand_l20", "xhand": "xhand"}
_F5 = ("thumb", "index", "middle", "ring", "pinky")


def _tips(hand: str, side: str):
    """Fingertip links in the order thumb, index, middle, ring[, little] (index = the finger next to the thumb)."""
    L = "_left" if side == "left" else ""
    return {
        "allegro": tuple(f"{f}_tip{L}_tip" for f in ("th", "ff", "mf", "rf")),
        "inspire": (f"inspire_thumb_distal{L}_tip",) + tuple(f"inspire_{f}_intermediate{L}_tip" for f in _F5[1:]),
        "leap": ("thumb_fingertip_tip", "fingertip_tip", "fingertip_2_tip", "fingertip_3_tip"),
        "sharpa": tuple(f"{side}_{f}_DP_tip" for f in _F5),
        "wuji": ("wuji_IP_thumb_tip",) + tuple(f"wuji_tipmount_{f}_tip" for f in _F5[1:]),
        "linkerhand": tuple(f"l20_{f}_distal_tip" for f in _F5),
        "xhand": (f"xhand_{side}_hand_thumb_rota_tip_tip", f"xhand_{side}_hand_index_rota_tip_tip", f"xhand_{side}_hand_mid_tip_tip",
                  f"xhand_{side}_hand_ring_tip_tip", f"xhand_{side}_hand_pinky_tip_tip"),
    }[hand]


RIGHT = [f"dj_{h}" for h in FILE]
LEFT = [f"dj_{h}_left" for h in FILE]


def specs(HandSpec, cache: str = "", asset_root: str = "") -> dict:
    """HandSpec per DexJoCo-X hand and side (empty when the generated URDFs are missing)."""
    names_file = os.path.join(G, "dataset_joint_names.json")
    if not os.path.exists(names_file):
        return {}
    data_names = json.load(open(names_file))
    out = {}
    for hand, f in FILE.items():
        for side, suf in (("right", ""), ("left", "_left")):
            urdf = os.path.join(G, f"{f}_{side}.urdf")
            if not os.path.exists(urdf):
                continue
            meta = json.load(open(urdf[:-5] + ".json"))
            tips = _tips(hand, side)
            missing = [t for t in tips if t not in meta["tips"]]
            if missing:
                raise RuntimeError(f"dj_{hand}{suf}: fingertips {missing} not in {urdf}")
            out[f"dj_{hand}{suf}"] = HandSpec(urdf, "hand_root", tips, dict(zip(data_names[hand], meta["actuator_joints"])))
    return out
