"""Lift-height marker for DexVerse-GraspTwoItems: a sphere above each active object at the success height.   [DexVerse only]

GraspTwoItems succeeds when both active objects are >= table top + 0.20 m (``active_objects_lifted``); unlike the
pour-style tasks it has no goal position, so nothing in the scene shows the operator how high to lift. This adds a
``scene_vis`` observation term (the same mechanism as ``mdp.pour_progress_marker_vis`` / ``show_goal_marker``):
one sphere per *active* object, hovering at the success height above the object's current xy, red -> green with
the object's lift progress. Inactive (parked) objects get no marker. Visual only (hidden from the recorded cameras).
"""

from __future__ import annotations

import torch
from isaaclab.managers import ManagerTermBase


class lift_height_marker_vis(ManagerTermBase):  # noqa: N801 - DexVerse mdp terms are lower-case classes
    """Params: ``object_names``, ``height_m`` (above the table top), ``radius``, ``opacity``, ``num_color_steps``,
    ``prim_path_prefix``."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        import isaaclab.sim as sim_utils
        from dexverse.visual_purpose import hide_marker_from_cameras
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        self._names = tuple(cfg.params["object_names"])
        self._height = float(cfg.params.get("height_m", 0.20))
        self._num_steps = max(int(cfg.params.get("num_color_steps", 11)), 2)
        radius = float(cfg.params.get("radius", 0.025))
        opacity = float(cfg.params.get("opacity", 0.6))
        prefix = cfg.params.get("prim_path_prefix", "/Visuals/LiftHeightMarker")
        markers = {}
        for i in range(self._num_steps):
            t = i / (self._num_steps - 1)
            markers[f"step_{i}"] = sim_utils.SphereCfg(
                radius=radius, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0 - t, t, 0.0), opacity=opacity))
        self._visualizer = VisualizationMarkers(VisualizationMarkersCfg(prim_path=prefix, markers=markers))
        hide_marker_from_cameras(self._visualizer)
        self._visualizer.set_visibility(True)

    def __call__(self, env, object_names=None, height_m=None, radius=None, opacity=None, num_color_steps=None,
                 prim_path_prefix=None) -> torch.Tensor:  # the manager checks the names against cfg.params
        from dexverse.tasks import dexverse_base_env_cfg as base
        from dexverse.tasks.config.grasping import grasp_two_items_cfg as G

        active = G._ensure_active_object_state(env)  # (num_envs, num_objects) bool
        table_z = env.scene.env_origins[:, 2] + base.DEFAULT_TABLE_TOP_HEIGHT
        pos, idx = [], []
        for i, name in enumerate(self._names):
            p = env.scene[name].data.root_pos_w
            on = active[:, i]
            if not bool(on.any()):
                continue
            progress = torch.clamp((p[:, 2] - table_z) / self._height, 0.0, 1.0)
            marker = torch.stack([p[:, 0], p[:, 1], table_z + self._height], dim=1)
            pos.append(marker[on])
            idx.append((progress[on] * (self._num_steps - 1)).round().to(torch.int32))
        if pos:
            self._visualizer.visualize(translations=torch.cat(pos), marker_indices=torch.cat(idx))
        return torch.zeros(env.num_envs, 0, device=env.device)
