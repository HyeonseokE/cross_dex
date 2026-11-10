"""Joint-angle finger retargeting: human hand joint angles -> robot joint angles by a per-hand linear map.   [vr_teleop]

The Bench2Dex / dex-retargeting path (``bench2dex_finger_retargeter``) fits the robot joints to the human
*fingertip positions* with an optimiser. That leaves the middle joints under-determined (on LEAP it put all
flexion into PIP / DIP and kept MCP straight), needs per-operator scale / offset tuning and adds optimiser +
filter latency. The Manus glove, however, gives a full hand skeleton, so the joint angles can be read off it
directly: this module computes the human joint angles from the OpenXR hand joints and maps them linearly onto
the robot joints with ``angle_cfgs/<hand>.yml``::

    low_pass_alpha: 0.6                      # new = alpha * target + (1 - alpha) * old   (1 = no filter)
    joints:
      "1": {features: {index_mcp: 1.0}, offset: 0.0, min: -0.3, max: 2.2}
      "13": {features: {thumb_oppose: 1.2, thumb_cmc: 0.3}, offset: -0.2, min: -0.47, max: 2.44}

q_robot = sum(gain * feature) + offset, clipped to [min, max]; a joint with no ``features`` is held at ``offset``.
``dexpilot_joints: ["12", "13", "14", "15"]`` takes those joints from the Bench2Dex DexPilot solution instead (the
teleop retargeter runs both); for joints whose axes have no human counterpart, e.g. the LEAP thumb.
``pinch_blend: {fingers: [index, middle, ring], start_cm: 3.0, full_cm: 1.5}``: the angle map reproduces joint
angles, not fingertip contact (robot fingers are longer / placed differently), so a pinch does not close exactly.
When the human thumb tip comes within ``start_cm`` of a listed fingertip, that finger's joints and the thumb
joints blend linearly toward the DexPilot solution, fully at ``full_cm`` (``blend_weights`` gives the per-joint
weight for the teleop retargeter).
Angles are in radians. ``joint_names`` is the yaml order; the DexVerse action joints are matched by name in
``dexverse_integration`` exactly as for the Bench2Dex path. The yaml is hot-reloaded (gains, offsets, limits,
filter) like the Bench2Dex ``hand_cfgs``.

Human features (all rad, flexion / opposition positive, 0 = straight; spread positive toward the thumb side on
both hands):
  <f>_mcp, <f>_pip, <f>_dip for f in index, middle, ring, little   (angle between the two bones at the joint)
  <f>_spread            abduction of the proximal bone relative to the middle finger's (index, ring, little)
                        or to the index finger's (middle), measured in the palm plane
  thumb_cmc, thumb_mcp, thumb_ip   flexion at the thumb joints (wrist-metacarpal-proximal, ...)
  thumb_oppose          angle between the thumb metacarpal and the index metacarpal (opposition / rotation)
  thumb_abd             angle of the thumb metacarpal out of the palm plane (palmar abduction)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

ANGLE_CFGS_DIR = Path(__file__).resolve().parent / "angle_cfgs"
FINGERS = ("index", "middle", "ring", "little")
_SPREAD_REF = {"index": "middle", "middle": "index", "ring": "middle", "little": "middle"}


def _pos(joints: dict, name: str) -> np.ndarray:
    return np.asarray(joints[name], dtype=np.float64)[:3]


def _flex(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Flexion at b between bones b->a and b->c: 0 straight, pi folded."""
    u, v = a - b, c - b
    cos = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
    return float(np.pi - np.arccos(np.clip(cos, -1.0, 1.0)))


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def human_hand_angles(joints: dict, side: str) -> dict[str, float] | None:
    """Joint-angle features (rad) of one hand from its OpenXR joints ``{name: [x, y, z, qw, qx, qy, qz]}``."""
    try:
        w = _pos(joints, "wrist")
        P = {f: {j: _pos(joints, f"{f}_{j}") for j in ("metacarpal", "proximal", "intermediate", "distal", "tip")}
             for f in FINGERS}
        T = {j: _pos(joints, f"thumb_{j}") for j in ("metacarpal", "proximal", "distal", "tip")}
    except KeyError:
        return None
    out: dict[str, float] = {}
    for f in FINGERS:
        p = P[f]
        out[f"{f}_mcp"] = _flex(p["metacarpal"], p["proximal"], p["intermediate"])
        out[f"{f}_pip"] = _flex(p["proximal"], p["intermediate"], p["distal"])
        out[f"{f}_dip"] = _flex(p["intermediate"], p["distal"], p["tip"])
    # palm plane: wrist, index and little metacarpal heads (proximal joints). Normal points out of the back of
    # the hand for a right hand with this triple order; thumb side = index side.
    n = _unit(np.cross(P["index"]["proximal"] - w, P["little"]["proximal"] - w))
    thumb_dir = _unit(P["index"]["proximal"] - P["little"]["proximal"])  # across the palm, toward the thumb
    for f in FINGERS:
        v = P[f]["intermediate"] - P[f]["proximal"]
        r = P[_SPREAD_REF[f]]["intermediate"] - P[_SPREAD_REF[f]]["proximal"]
        vp, rp = _unit(v - (v @ n) * n), _unit(r - (r @ n) * n)
        ang = float(np.arccos(np.clip(vp @ rp, -1.0, 1.0)))
        sign = 1.0 if (vp - rp) @ thumb_dir > 0 else -1.0  # this finger points more thumb-ward than the reference
        out[f"{f}_spread"] = sign * ang
    out["thumb_cmc"] = _flex(w, T["metacarpal"], T["proximal"])
    out["thumb_mcp"] = _flex(T["metacarpal"], T["proximal"], T["distal"])
    out["thumb_ip"] = _flex(T["proximal"], T["distal"], T["tip"])
    tm = _unit(T["proximal"] - T["metacarpal"])
    im = _unit(P["index"]["proximal"] - P["index"]["metacarpal"])
    out["thumb_oppose"] = float(np.arccos(np.clip(tm @ im, -1.0, 1.0)))
    out["thumb_abd"] = float(np.arcsin(np.clip(abs(tm @ n), 0.0, 1.0)))
    return out


# robot joint name -> human finger it belongs to (for the pinch blend), by the naming of each hand
_NAME_HINTS = (
    ("thumb", ("thumb", "th", "finger1_")), ("index", ("index", "ff", "finger2_")), ("middle", ("middle", "mf", "finger3_")),
    ("ring", ("ring", "rf", "finger4_")), ("little", ("little", "pinky", "lf", "finger5_")),
)
_LEAP_FINGER = {**{str(j): "index" for j in (0, 1, 2, 3)}, **{str(j): "middle" for j in (4, 5, 6, 7)},
                **{str(j): "ring" for j in (8, 9, 10, 11)}, **{str(j): "thumb" for j in (12, 13, 14, 15)}}
_ALLEGRO_FINGER = {**{f"joint_{j}.0": "index" for j in (0, 1, 2, 3)}, **{f"joint_{j}.0": "middle" for j in (4, 5, 6, 7)},
                   **{f"joint_{j}.0": "ring" for j in (8, 9, 10, 11)}, **{f"joint_{j}.0": "thumb" for j in (12, 13, 14, 15)}}


def _finger_of_joint(name: str, hand_type: str) -> str | None:
    if hand_type == "leap":
        return _LEAP_FINGER.get(name)
    if hand_type == "allegro":
        return _ALLEGRO_FINGER.get(name.replace("mult_", ""))
    low = name.lower().replace("right_", "").replace("left_", "")
    if low.startswith("wrj") or low.startswith("lfj5"):
        return None  # wrist / palm arch: never blended
    for finger, hints in _NAME_HINTS:
        if any(low.startswith(h) for h in hints):
            return finger
    return None


class AngleHand:
    """One hand: OpenXR joints -> robot joint angles (``joint_names`` order) by ``angle_cfgs/<hand_type>.yml``."""

    def __init__(self, hand_type: str, side: str, reload_every: int = 120):
        self.hand_type, self.side = hand_type, side
        self.path = ANGLE_CFGS_DIR / f"{hand_type}.yml"
        if not self.path.is_file():
            raise FileNotFoundError(f"no angle retargeting config for '{hand_type}': {self.path}")
        self._reload_every, self._calls, self._mtime = reload_every, 0, 0.0
        self._q: np.ndarray | None = None
        self._load()
        self.bridge = f"AngleHand({hand_type}, {side}, {len(self.joint_names)} joints, {self.path.name})"

    def __repr__(self) -> str:
        return self.bridge

    def _load(self) -> None:
        import yaml

        cfg = yaml.safe_load(self.path.read_text())
        side_cfg = cfg.get(self.side) or {}
        joints = dict(cfg["joints"])
        joints.update(side_cfg.get("joints", {}))  # per-side overrides (e.g. mirrored spread signs)
        self.joint_names = [str(n) for n in joints]
        self._rules = []
        for n in self.joint_names:
            r = joints[n] or {}
            self._rules.append((dict(r.get("features", {})), float(r.get("offset", 0.0)),
                                float(r.get("min", -np.inf)), float(r.get("max", np.inf))))
        self.alpha = float(cfg.get("low_pass_alpha", 1.0))
        self.dexpilot_joints = [str(n) for n in side_cfg.get("dexpilot_joints", cfg.get("dexpilot_joints", []))]
        pb = side_cfg.get("pinch_blend", cfg.get("pinch_blend")) or {}
        self.pinch_fingers = list(pb.get("fingers", []))
        self.pinch_start, self.pinch_full = float(pb.get("start_cm", 3.0)) / 100, float(pb.get("full_cm", 1.5)) / 100
        # which human finger each robot joint belongs to (from its features; "thumb" when a thumb feature or when
        # the joint is in dexpilot_joints and named like a thumb)
        self._finger_of = [_finger_of_joint(n, self.hand_type) for n in self.joint_names]
        self.needs_dexpilot = bool(self.dexpilot_joints or self.pinch_fingers)
        self._mtime = self.path.stat().st_mtime

    def reload_if_changed(self) -> None:
        try:
            if self.path.stat().st_mtime != self._mtime:
                names = list(self.joint_names)
                self._load()
                if names != self.joint_names:
                    self._q = None
                print(f"[angle] reloaded {self.path.name}", flush=True)
        except Exception as exc:  # noqa: BLE001 - keep the old mapping on a bad edit
            print(f"[angle] reload failed ({type(exc).__name__}: {exc}); keeping the previous mapping", flush=True)

    def features(self, joints: dict) -> dict[str, float] | None:
        return human_hand_angles(joints, self.side)

    def blend_weights(self, joints: dict) -> np.ndarray | None:
        """Per-joint weight toward the DexPilot solution for the pinch blend (None when not configured)."""
        if not self.pinch_fingers:
            return None
        try:
            thumb = np.asarray(joints["thumb_tip"], dtype=np.float64)[:3]
            dist = {f: float(np.linalg.norm(np.asarray(joints[f"{f}_tip"], dtype=np.float64)[:3] - thumb))
                    for f in self.pinch_fingers}
        except KeyError:
            return None
        span = max(self.pinch_start - self.pinch_full, 1e-6)
        w_f = {f: float(np.clip((self.pinch_start - d_) / span, 0.0, 1.0)) for f, d_ in dist.items()}
        w_thumb = max(w_f.values()) if w_f else 0.0
        w = np.zeros(len(self.joint_names), dtype=np.float32)
        for i, f in enumerate(self._finger_of):
            if f == "thumb":
                w[i] = w_thumb
            elif f in w_f:
                w[i] = w_f[f]
        return w

    def retarget(self, joints: dict) -> np.ndarray | None:
        self._calls += 1
        if self._reload_every and self._calls % self._reload_every == 0:
            self.reload_if_changed()
        feat = self.features(joints)
        if feat is None:
            return None
        q = np.empty(len(self._rules), dtype=np.float32)
        for i, (gains, off, lo, hi) in enumerate(self._rules):
            v = off + sum(g * feat.get(k, 0.0) for k, g in gains.items())
            q[i] = min(max(v, lo), hi)
        if self._q is None or self.alpha >= 1.0:
            self._q = q
        else:
            self._q = self.alpha * q + (1.0 - self.alpha) * self._q
        return self._q.copy()
