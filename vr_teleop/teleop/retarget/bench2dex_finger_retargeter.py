"""Finger retargeting with Bench2Dex's Manus-tuned ``RetargetBridge`` (simulator-independent).

``bench2dex/`` is the Bench2Dex teleop code (``bench2dex_teleop/teleop``, commit 511cbf3), kept unmodified:
``retarget_bridge.py`` maps 21 MediaPipe keypoints in a canonical wrist frame to robot joint angles with
DexPilot, using per-hand YAMLs (``bench2dex/hand_cfgs``) whose scale / offset / joint locks were tuned
on the Manus skeleton, and the Bench2Dex URDF-Zoo URDFs (directory in ``$URDF_ZOO_DIR``).

:class:`Bench2DexHand` feeds it the OpenXR 26-joint hand (``common/openxr_hand.py``) every device produces.
"""

from __future__ import annotations

import contextlib

import numpy as np

from ...common.rotation_utils import rot_from_wxyz

# OpenXR joint names in MediaPipe 21-keypoint order (wrist, thumb CMC/MCP/IP/TIP, then MCP/PIP/DIP/TIP
# per finger; the finger metacarpals are skipped, as in Bench2Dex).
MEDIAPIPE_FROM_OPENXR = (
    "wrist",
    "thumb_metacarpal", "thumb_proximal", "thumb_distal", "thumb_tip",
    "index_proximal", "index_intermediate", "index_distal", "index_tip",
    "middle_proximal", "middle_intermediate", "middle_distal", "middle_tip",
    "ring_proximal", "ring_intermediate", "ring_distal", "ring_tip",
    "little_proximal", "little_intermediate", "little_distal", "little_tip",
)

# Columns = RetargetBridge canonical wrist axes expressed in the OpenXR wrist joint frame: (+Y, -X, +Z),
# the joint frame turned +90 deg about Z (Bench2Dex xr_hand_source.OPENXR_JOINT_TO_CANONICAL, verified
# against the URDF-Zoo hands' zero-pose fingertips).
OPENXR_JOINT_TO_CANONICAL = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

# Bench2Dex hand configs (hand_cfgs/<name>.yml).
HAND_TYPES = ("shadow", "allegro", "leap", "sharpa", "wuji", "rh56dfx", "rh5dg2", "schunk_svh", "orca",
              "ability", "dexhand021", "revo2")


class Bench2DexHand:
    """One hand: OpenXR joints -> robot joint angles (``joint_names`` order)."""

    def __init__(self, hand_type: str, side: str, input_device: str = "manus", reload_every: int = 120):
        """``input_device="manus"`` uses the Manus-tuned scale / offset, ``"xr"`` the OpenXR hand-tracking ones.
        Edits to ``hand_cfgs/<hand_type>.yml`` scale / offset are reloaded every ``reload_every`` calls."""
        from .bench2dex.retarget_bridge import RetargetBridge

        self.bridge = RetargetBridge(hand_type, side, input_device=input_device)
        self.joint_names = [str(n) for n in self.bridge._retargeting.optimizer.robot.dof_joint_names]
        self._reload_every, self._calls = reload_every, 0
        self._prior = None  # (AngleHand theta source, lambda, pinch_scale) when angle_cfgs/<hand>.yml has `prior:`
        self._install_prior(hand_type, side)
        self._pinned = 0  # consecutive solves with most target joints on a limit (warm-start lock-in guard)

    def seed(self, q_by_name: dict) -> None:
        """Warm-start the next solve from these joint angles (robot joint name -> rad); unknown names are ignored.

        Used by the angle backend: DexPilot then only refines the angle-map posture (fingertip contact), instead of
        carrying its own solution from frame to frame, which can lock into limit-pinned local minima."""
        seq = self.bridge._retargeting
        q = seq.get_qpos() if hasattr(seq, "get_qpos") else np.zeros(seq.optimizer.robot.dof)
        q = np.array(q, dtype=np.float32)
        for i, n in enumerate(self.joint_names):
            v = q_by_name.get(n)
            if v is not None:
                q[i] = float(v)
        seq.set_qpos(q)

    def retarget(self, joints: dict) -> np.ndarray | None:
        """``joints``: ``{name: [x, y, z, qw, qx, qy, qz]}`` (OpenXR). Returns joint angles or None."""
        try:
            keypoints = np.stack([np.asarray(joints[n], dtype=np.float32)[:3] for n in MEDIAPIPE_FROM_OPENXR])
        except KeyError:
            return None
        wrist_frame = (rot_from_wxyz(np.asarray(joints["wrist"], dtype=np.float64)[3:]) @ OPENXR_JOINT_TO_CANONICAL)
        self._calls += 1
        if self._reload_every and self._calls % self._reload_every == 0:
            self.bridge.reload_tunables_if_changed()
        if self._prior is not None:
            self._update_prior(joints)
        with _grad_enabled():  # dex-retargeting optimisers use torch autograd
            qpos = self.bridge.retarget_keypoints(keypoints, wrist_frame.astype(np.float32))
        if qpos is not None:
            self._lock_in_guard()
        return None if qpos is None else np.asarray(qpos, dtype=np.float32)

    # ----------------------------------------------------------------- human joint-angle prior
    def _install_prior(self, hand_type: str, side: str) -> None:
        """Wrap the optimiser objective with lambda * sum w_j (q_j - theta_j)^2 when angle_cfgs/<name>.yml asks for it
        (``prior: {lambda: ...}``). theta comes from that file's joint map (human angles -> this hand's joints)."""
        from .angle_retargeter import ANGLE_CFGS_DIR, AngleHand
        from .dexverse_integration_names import angle_cfg_name  # hand_type -> angle cfg file name

        name = angle_cfg_name(hand_type)
        if name is None or not (ANGLE_CFGS_DIR / f"{name}.yml").is_file():
            return
        import yaml

        cfg = yaml.safe_load((ANGLE_CFGS_DIR / f"{name}.yml").read_text()).get("prior") or {}
        lam = float(cfg.get("lambda", 0.0))
        if lam <= 0:
            return
        theta_src = AngleHand(name, side)
        theta_src.alpha = 1.0  # raw angles; DexPilot's own filter smooths the result
        opt = self.bridge._retargeting.optimizer
        target = [str(n) for n in opt.target_joint_names]
        theta_idx = [(ti, theta_src.joint_names.index(n)) for ti, n in enumerate(target) if n in theta_src.joint_names]
        jw = {str(k): float(v) for k, v in (cfg.get("joint_weights") or {}).items()}  # per-joint multipliers of lambda
        self._prior = {"src": theta_src, "lam": lam, "jw": jw, "pinch_scale": float(cfg.get("pinch_scale", 0.1)),
                       "start": float(cfg.get("start_cm", 3.0)) / 100, "full": float(cfg.get("full_cm", 1.5)) / 100,
                       "theta_idx": theta_idx, "theta": None, "w": None, "n": len(target)}
        base_get = opt.get_objective_function
        prior = self._prior

        def get_objective_function(ref_value, fixed_qpos, last_qpos):
            base = base_get(ref_value, fixed_qpos, last_qpos)
            theta, w = prior["theta"], prior["w"]
            if theta is None:
                return base

            def objective(x, grad):
                val = base(x, grad)
                d = x - theta
                if grad.size > 0:
                    grad[:] += 2.0 * w * d
                return val + float(np.sum(w * d * d))

            return objective

        opt.get_objective_function = get_objective_function
        print(f"[vr_teleop] {hand_type} ({side}): DexPilot + joint-angle prior lambda={lam} on {len(theta_idx)}/{len(target)} "
              f"joints (theta from angle_cfgs/{name}.yml, pinch gate x{prior['pinch_scale']} under {prior['start']*100:.0f} cm)")

    def _update_prior(self, joints: dict) -> None:
        p = self._prior
        q_theta = p["src"].retarget(joints)
        if q_theta is None:
            p["theta"] = None
            return
        theta = np.zeros(p["n"], dtype=np.float64)
        w = np.zeros(p["n"], dtype=np.float64)
        for ti, ai in p["theta_idx"]:
            theta[ti] = q_theta[ai]
            w[ti] = p["lam"] * p["jw"].get(p["src"].joint_names[ai], 1.0)
        # pinch gate: thumb + fingers whose tip is near the thumb tip get a small weight (DexPilot decides the contact)
        bw = p["src"].blend_weights(joints) if p["src"].pinch_fingers else None
        if bw is not None:
            for ti, ai in p["theta_idx"]:
                if bw[ai] > 0:
                    w[ti] *= (1.0 - bw[ai]) + bw[ai] * p["pinch_scale"]
        p["theta"], p["w"] = theta, w

    def _lock_in_guard(self, frac: float = 0.5, frames: int = 90) -> None:
        """DexPilot warm-starts from its previous solution and can lock into a limit-pinned local minimum (seen on the
        live glove 2026-10-28: Shadow MCP at 90 deg, LEAP MCP at -18 deg for minutes while the hand was open). When
        more than ``frac`` of the optimised joints sit on a limit for ``frames`` consecutive solves (~1.5-3 s), the
        warm start is reset to the open hand (lower bounds) so the next solve starts over."""
        seq = self.bridge._retargeting
        q, lim = np.asarray(seq.last_qpos), np.asarray(seq.joint_limits)
        free = (lim[:, 1] - lim[:, 0]) > 1e-3  # ignore locked joints
        if not free.any():
            return
        on_limit = np.mean((np.abs(q[free] - lim[free, 0]) < 0.02) | (np.abs(q[free] - lim[free, 1]) < 0.02))
        self._pinned = self._pinned + 1 if on_limit > frac else 0
        if self._pinned >= frames:
            seq.last_qpos = lim[:, 0].astype(np.float32)
            if getattr(seq, "filter", None) is not None:
                seq.filter.reset()
            self._pinned = 0
            print(f"[vr_teleop] {self.bridge._hand_type} ({self.bridge._side}): DexPilot stuck on joint limits "
                  f"({on_limit:.0%} of joints) - warm start reset to the open hand", flush=True)


@contextlib.contextmanager
def _grad_enabled():
    try:
        import torch
    except ImportError:
        yield
        return
    with torch.enable_grad(), torch.inference_mode(False):  # teleop loops often run under inference_mode
        yield
