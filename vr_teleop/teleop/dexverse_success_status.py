"""Live success-condition status for DexVerse recording: which gate of the task's success term holds.   [DexVerse only]

record_demos only says "saved" once the success term has held ``--num_success_steps`` frames in a row; this
breaks the term into its gates so the operator sees what is missing, e.g.

    lift 0.12/0.30 m NO | tilt 45>=100 deg NO | goal 4.1<=2.0 cm NO | grip OK | hold 0/10

Supported success terms (the current target tasks): ``lift_and_tilt_with_contact_zones`` (Grasp* / Functional*
pour-style), ``hammer_strike_success``, ``object_upright_and_lifted``, ``object_lifted``,
``active_objects_lifted`` (GraspTwoItems). Any other term shows only whether the whole condition holds.
Plain ASCII, so the headset panel's default font can draw it.
"""

from __future__ import annotations

import math

import torch


def _flag(ok) -> str:
    return "OK" if bool(ok) else "NO"


class SuccessStatus:
    def __init__(self, env, success_term, resolve):
        """``resolve(env, term)`` returns the callable record_demos evaluates (a class term's instance)."""
        self.env, self.term = env, success_term
        self.func = resolve(env, success_term) if success_term is not None else None
        name = getattr(self.func, "__name__", None) or type(self.func).__name__
        self.name = name
        self._parts = {
            "lift_and_tilt_with_contact_zones": self._lift_and_tilt,
            "hammer_strike_success": self._hammer,
            "object_upright_and_lifted": self._upright_lifted,
            "object_lifted": self._lifted,
            "active_objects_lifted": self._two_items,
        }.get(name)

    def text(self, hold: int, need: int) -> str:
        if self.func is None:
            return "no success condition"
        try:
            parts = self._parts() if self._parts is not None else [f"success {_flag(self.func(self.env, **self.term.params)[0])}"]
        except Exception as exc:  # noqa: BLE001 - a display helper must not stop the recording
            parts = [f"status n/a ({type(exc).__name__})"]
            self._parts = None
        return " | ".join([*parts, f"hold {hold}/{need}"])

    # ------------------------------------------------------------------ helpers
    def _obj(self, cfg):
        return self.env.scene[cfg.name]

    def _height(self, cfg) -> float:
        from dexverse.tasks.mdp.utils import root_height_delta

        return float(root_height_delta(self._obj(cfg))[0])

    def _tilt_deg(self, cfg, axis_local, world_axis) -> float:
        from dexverse.tasks.mdp.utils import axis_tilt_angle

        return math.degrees(float(axis_tilt_angle(self._obj(cfg).data.root_quat_w, axis_local=axis_local,
                                                  world_axis=world_axis)[0]))

    def _zones(self, prefix: str, asset_cfg, object_cfg, coverage: bool):
        """Forbidden-zone violation (or contact-zone coverage) from the term instance's zone tensors."""
        from dexverse.tasks.mdp import terminations as T

        f = self.func
        names = ("sphere_centers", "sphere_r2", "box_centers", "box_half", "cylinder_centers",
                 "cylinder_radius2_height", "cylinder_quats")
        tensors = [getattr(f, f"_{prefix}{n}", None) for n in names]
        if tensors[0] is None or all(t.shape[0] == 0 for t in (tensors[0], tensors[2], tensors[4])):
            return None
        fn = T._eval_zone_coverage if coverage else T._eval_zone_violation
        return bool(fn(self.env, asset_cfg, object_cfg, *tensors)[0])

    # ------------------------------------------------------------------ per success term
    def _lift_and_tilt(self):
        p = self.term.params
        out = []
        if p["min_height"] > 0:
            h = self._height(p["object_cfg"])
            out.append(f"lift {h:.2f}/{p['min_height']:.2f} m {_flag(h >= p['min_height'])}")
        ang = self._tilt_deg(p["object_cfg"], p.get("axis_local", (0, 0, 1)), p.get("world_axis", (0, 0, 1)))
        thr = math.degrees(p["threshold_rad"])
        ge = p.get("tilt_ge", True)
        out.append(f"tilt {ang:.0f}{'>=' if ge else '<='}{thr:.0f} deg {_flag(ang >= thr if ge else ang <= thr)}")
        if p.get("goal_asset_cfg") is not None and p.get("goal_xy_threshold") is not None:
            from isaaclab.utils.math import quat_apply

            obj, goal = self._obj(p["object_cfg"]), self._obj(p["goal_asset_cfg"])
            pos = obj.data.root_pos_w
            off = p.get("goal_object_local_offset")
            if off is not None:
                pos = pos + quat_apply(obj.data.root_quat_w, torch.tensor([off], device=pos.device, dtype=pos.dtype))
            d = float((pos[0, :2] - goal.data.root_pos_w[0, :2]).norm()) * 100
            thr_cm = p["goal_xy_threshold"] * 100
            out.append(f"goal {d:.1f}<={thr_cm:.1f} cm {_flag(d <= thr_cm)}")
        bad = self._zones("forbidden_", p["asset_cfg"], p["object_cfg"], coverage=False)
        if bad is not None:
            out.append(f"grip zone {_flag(not bad)}")
        cov = self._zones("contact_", p.get("contact_asset_cfg") or p["asset_cfg"],
                          p.get("contact_object_cfg") or p["object_cfg"], coverage=True)
        if cov is not None:
            out.append(f"contact {_flag(cov)}")
        return out

    def _hammer(self):
        from dexverse.tasks.mdp.utils import get_init_joint_pos, resolve_joint_ids

        p = self.term.params
        out = []
        bad = self._zones("forbidden_", p["asset_cfg"], p["object_cfg"], coverage=False)
        if bad is not None:
            out.append(f"grip zone {_flag(not bad)}")
        tbad = self._zones("target_forbidden_", p["asset_cfg"], p["target_asset_cfg"], coverage=False)
        if tbad is not None:
            out.append(f"no finger on nail {_flag(not tbad)}")
        target = self._obj(p["target_joint_cfg"])
        ids = resolve_joint_ids(self.env, p["target_joint_cfg"])
        press = float((target.data.joint_pos[:, ids] - get_init_joint_pos(target, ids)).abs().max()) * 100
        thr = p["press_threshold_m"] * 100
        out.append(f"press {press:.1f}/{thr:.1f} cm {_flag(press >= thr)}")
        return out

    def _upright_lifted(self):
        p = self.term.params
        cfg = p["object_cfg"]
        h, ang = self._height(cfg), self._tilt_deg(cfg, (0, 0, 1), (0, 0, 1))
        thr = math.degrees(p["max_tilt_rad"])
        return [f"lift {h:.2f}/{p['min_height']:.2f} m {_flag(h >= p['min_height'])}",
                f"tilt {ang:.0f}<={thr:.0f} deg {_flag(ang <= thr)}"]

    def _lifted(self):
        p = self.term.params
        cfg = p.get("object_cfg") or p["asset_cfg"]
        h = self._height(cfg)
        return [f"lift {h:.2f}/{p['min_height']:.2f} m {_flag(h >= p['min_height'])}"]

    def _two_items(self):
        from dexverse.tasks.config.grasping import grasp_two_items_cfg as G

        env = self.env
        active = G._ensure_active_object_state(env)
        table = float(env.scene.env_origins[0, 2]) + G.LIFT_SUCCESS_Z - G.LIFT_SUCCESS_HEIGHT_M  # table top
        out = []
        for i, name in enumerate(G.OBJECT_NAMES):
            if bool(active[0, i]):
                h = float(env.scene[name].data.root_pos_w[0, 2]) - table
                out.append(f"{name} {h:.2f}/{G.LIFT_SUCCESS_HEIGHT_M:.2f} m {_flag(h >= G.LIFT_SUCCESS_HEIGHT_M)}")
        return out
