"""Bimanual policy space for the pi0.5 baselines: the one-hand space ([canonical wrist 9 | finger part]) per hand,
right hand first:

  latent (pi05_XL): [R wrist 9 | R z 32 | L wrist 9 | L z 32]                    = 82 dims
  joint  (pi05)   : [R wrist 9 | R fingers padded to 22 | L wrist 9 | L fingers padded to 22] = 62 dims

Which env action dims are which hand's wrist / fingers, their joint names, scale and offset and both wrists' kinematics
come from the simulator (measure_wrist_bimanual.py -> wrist_calib_<task>.json, one entry per hand family), never from
dataset names. Right hand = the one-hand model (hands.HANDS[h], VAE of the right 6-hand latent), left hand = h + "_left"
(hands.LEFT_HANDS, left 6-hand latent). Finger values are angles_to_normalized(joints).clamp(-1, 1), as one-hand.

Finger joint names: bimanual sims prefix every joint with its side (right_/rh_, left_/lh_). The left models map those
names already (LEFT_HANDS data_name); for the right model the names are tried as they are, then mirrored to the left name
and mapped by the left alias (left and right URDFs share joint names, check_left_hands.py), then with the prefix removed.
The mapping that is used is checked against the simulator's fingertip positions (verify_bimanual.py).
"""
from __future__ import annotations

import json

import numpy as np
import torch

from dexverse_xlvla import hands as H
from dexverse_xlvla.wrist import Wrist

SIDES = ("right", "left")
PREFIX = {"right": ("right_", "rh_"), "left": ("left_", "lh_")}
MIRROR = {"right_": "left_", "rh_": "lh_"}
PAD = 22


def model_name(hand: str, side: str) -> str:
    return hand if side == "right" else f"{hand}_left"


def _strip(n: str, side: str) -> str:
    for p in PREFIX[side]:
        if n.startswith(p):
            return n[len(p):]
    return n


def finger_index(model, names: list, side: str) -> tuple[list, str]:
    """Positions in `names` (the side's finger joints, sim names) of the model DoFs, and the rule that matched."""
    cands = [("as is", list(names))]
    if side == "right":
        left = H.HANDS.get(model.hand_name + "_left")
        mir = []
        for n in names:
            for a, b in MIRROR.items():
                if n.startswith(a):
                    n = b + n[len(a):]
                    break
            mir.append(left.data_name.get(n, n) if left else n)
        cands.append(("mirrored left alias", mir))
    cands.append(("prefix removed", [_strip(n, side) for n in names]))
    for rule, c in cands:
        try:
            return model.data_index(c), rule
        except KeyError:
            continue
    raise KeyError(f"{model.hand_name}: no name rule maps {names} to {model.dof_joints}")


def load_calib(path: str, hand: str) -> dict:
    c = json.load(open(path))
    if hand not in c:
        raise KeyError(f"{path}: no calibration for {hand} (has {sorted(c)})")
    return c[hand]


class BimanualSpace:
    """env action / state (action-term order, as the common evaluator sends joint_pos) <-> policy vector."""

    def __init__(self, calib: dict, hand: str, models: dict, codecs: dict | None = None, pad: int = PAD):
        """models: side -> CanonicalHandModel; codecs: side -> trainer with the side model's VAE (latent) or None (joint)."""
        self.hand, self.models, self.codecs, self.pad = hand, models, codecs, pad
        self.action_joints = calib["action_joints"]
        self.state_joints = calib.get("state_order_joints")       # evaluator state order (action joints first)
        self.n_act = len(self.action_joints)
        self.side = {}
        for s in SIDES:
            c, m = calib["sides"][s], models[s]
            if m.hand_name != model_name(hand, s):
                raise ValueError(f"{s}: model {m.hand_name} != {model_name(hand, s)}")
            idx, rule = finger_index(m, c["finger_joints"], s)
            fd = np.asarray(c["finger_dims"])
            sc, of = np.asarray(c["finger_scale"], np.float64), np.asarray(c["finger_offset"], np.float64)
            uncovered = sorted(set(range(len(fd))) - set(idx))
            if uncovered:
                raise ValueError(f"{hand} {s}: action dims without a model DoF: {[c['finger_joints'][i] for i in uncovered]}")
            if codecs is None and m.dof_count() > pad:
                raise ValueError(f"{hand} {s}: {m.dof_count()} finger joints > pad {pad}")
            self.side[s] = {"wrist": Wrist(c, m.canon.numpy()), "wdims": list(c["wrist_dims"]),
                            "cols": fd[idx].tolist(), "scale": sc[idx], "offset": of[idx], "rule": rule}
        self.k = codecs[SIDES[0]].config.latent_dim_hand if codecs else pad
        self.dim = 2 * (9 + self.k)

    def names(self) -> list:
        f = "z" if self.codecs else "finger"
        w = ("x", "y", "z", "r6d_0", "r6d_1", "r6d_2", "r6d_3", "r6d_4", "r6d_5")
        return [n for s in SIDES for n in [f"{s}_wrist_{x}" for x in w] + [f"{s}_{f}_{i}" for i in range(self.k)]]

    def _fingers(self, s: str, fn: torch.Tensor) -> np.ndarray:
        if self.codecs:
            from dexverse_xlvla.eval_latent import enc
            with torch.no_grad():
                return enc(self.codecs[s], model_name(self.hand, s), fn).numpy()
        out = np.zeros((len(fn), self.pad), np.float32)
        out[:, : fn.shape[1]] = fn.numpy()
        return out

    def to_policy(self, x: np.ndarray, kind: str) -> np.ndarray:
        """(T, >= n_act) env actions (kind "action") or joint positions in action order (kind "state") -> (T, dim)."""
        x = np.atleast_2d(np.asarray(x, np.float64))
        parts = []
        for s in SIDES:
            d, m = self.side[s], self.models[s]
            w = x[:, d["wdims"]]
            wrist = d["wrist"].vec_from_actions(w) if kind == "action" else d["wrist"].vec_from_joints(w)
            q = x[:, d["cols"]]
            if kind == "action":
                q = d["offset"] + d["scale"] * q
            fn = m.angles_to_normalized(torch.tensor(q, dtype=torch.float32)).clamp(-1, 1)
            parts += [wrist, self._fingers(s, fn)]
        return np.concatenate(parts, 1).astype(np.float32)

    def to_action(self, vec: np.ndarray, q_wrist_init: dict) -> np.ndarray:
        """(T, dim) policy vectors -> (T, n_act) env actions. q_wrist_init: side -> wrist joints (action order) that
        warm-start the IK (evaluation: the observed wrist joints; the solution is carried along the T steps)."""
        vec = np.atleast_2d(np.asarray(vec, np.float64))
        out = np.zeros((len(vec), self.n_act))
        for i, s in enumerate(SIDES):
            d, m, o = self.side[s], self.models[s], i * (9 + self.k)
            out[:, d["wdims"]] = d["wrist"].actions_from_vec(vec[:, o:o + 9], q_init=q_wrist_init[s])
            f = torch.tensor(vec[:, o + 9:o + 9 + self.k], dtype=torch.float32)
            if self.codecs:
                from dexverse_xlvla.eval_latent import dec
                with torch.no_grad():
                    qn = dec(self.codecs[s], model_name(self.hand, s), f).clamp(-1, 1)
            else:
                qn = f[:, : m.dof_count()].clamp(-1, 1)
            lo, hi = m._lower, m._upper
            q = ((qn + 1) * 0.5 * (hi - lo) + lo).numpy().astype(np.float64)
            out[:, d["cols"]] = (q - d["offset"]) / d["scale"]
        return out


def build(calib_json: str, hand: str, space: str, ckpt: str | None = None, ckpt_left: str | None = None,
          pad: int = PAD) -> BimanualSpace:
    """space "latent": VAEs from ckpt (right 6-hand) / ckpt_left (left 6-hand); "joint": hand models only."""
    calib = load_calib(calib_json, hand)
    if space == "latent":
        from dexverse_xlvla.eval_latent import load
        codecs = {"right": load(ckpt), "left": load(ckpt_left)}
        for tr in codecs.values():
            for a in tr.autoencoders.values():
                a.eval()
        models = {s: codecs[s].hand_models[model_name(hand, s)] for s in SIDES}
    elif space == "joint":
        codecs, models = None, {s: H.CanonicalHandModel(model_name(hand, s)) for s in SIDES}
    else:
        raise ValueError(space)
    return BimanualSpace(calib, hand, models, codecs, pad)
