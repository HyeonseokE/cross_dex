# Copyright (c) 2025-2026, The DexVerse Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Headless check without a headset: drive a DexVerse floating hand from the Manus gloves (fingers only).   [DexVerse]

The wrist stays fixed (``wrist_mode="fixed"``); records the camera videos and the retargeted finger
targets per step. Needs the Manus bridge (``run_manus_bridge.sh``; ``--fake`` = synthetic hand)::

    python vr_teleop/tools/check_fingers_headless.py --task Dexverse-PushT-v1 \\
        --robot_type floating_shadow_right --steps 600 --headless --enable_cameras

Writes ``<out>/manus_<camera>.mp4`` and ``<out>/finger_joints.csv`` (retargeted finger targets per step).
"""

from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Manus fingers -> DexVerse floating hand check.")
parser.add_argument("--task", default="Dexverse-PushT-v1")
parser.add_argument("--robot_type", default=None, help="floating_{hand}_{side} override")
parser.add_argument("--steps", type=int, default=600)
parser.add_argument("--wrist_mode", default="fixed", choices=("fixed", "glove"))
parser.add_argument("--retargeting_scheme", default="dexpilot", choices=("dexpilot", "vector"))
parser.add_argument("--fingers", default="bench2dex", choices=("bench2dex", "dexverse"),
                    help="finger retargeting: Bench2Dex Manus-tuned RetargetBridge or the DexVerse YAMLs")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, default=47000)
parser.add_argument("--video_every", type=int, default=2, help="record a frame every N steps (0: off)")
parser.add_argument("--out", default="/workspace/logs/manus_check")
parser.add_argument("--enable_pinocchio", action="store_true", default=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.enable_pinocchio:
    import pinocchio  # noqa: F401  (must load before the Kit app for dex-retargeting)
app = AppLauncher(args).app

import csv  # noqa: E402
import os  # noqa: E402

import dexverse.tasks  # noqa: E402,F401
import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
import numpy as np  # noqa: E402
import torch  # noqa: E402
from dexverse.tasks.config.floating_teleop import apply_teleop_retargeting_scheme  # noqa: E402
from dexverse.tasks.utils import parse_env_cfg  # noqa: E402
from isaaclab.devices.teleop_device_factory import create_teleop_device  # noqa: E402

from vr_teleop.teleop.dexverse_integration import add_manus_device  # noqa: E402


def main() -> None:
    os.makedirs(args.out, exist_ok=True)
    cfg = parse_env_cfg(args.task, device=args.device, num_envs=1)
    if args.robot_type:
        cfg = type(cfg)(robot_type=args.robot_type)
        cfg.sim.device = args.device or cfg.sim.device
        cfg.scene.num_envs = 1
    cfg.terminations.time_out = None
    apply_teleop_retargeting_scheme(cfg.teleop_devices, args.retargeting_scheme)
    add_manus_device(cfg.teleop_devices, fingers=args.fingers, host=args.host, port=args.port, wrist_mode=args.wrist_mode)

    env = gym.make(args.task, cfg=cfg).unwrapped
    active = {"on": False}

    def on_start():
        active["on"] = True
        for r in device._retargeters:
            if hasattr(r, "calibrate_wrist_pose"):
                r.calibrate_wrist_pose()
        print("[check] glove data arrived: teleop active, wrist calibrated")

    device = create_teleop_device("manus", cfg.teleop_devices.devices, {"START": on_start})
    print(f"[check] {device}  robot_type={getattr(cfg, 'robot_type', None)}  action_dim={env.action_space.shape}")
    env.reset()
    device.reset()

    cams = {n: s for n, s in env.scene.sensors.items() if hasattr(s, "data") and "rgb" in getattr(s.data, "output", {})}
    frames = {n: [] for n in cams}
    rows = []
    zero = torch.zeros(env.action_space.shape, device=env.device)
    t0 = time.time()
    with torch.inference_mode():
        for step in range(args.steps):
            action = device.advance()
            act = action.reshape(1, -1).to(env.device) if active["on"] else zero
            env.step(act)
            rows.append(act[0].detach().cpu().numpy())
            if args.video_every and step % args.video_every == 0:
                for n, s in cams.items():
                    rgb = s.data.output["rgb"][0, ..., :3].detach().cpu().numpy()
                    frames[n].append(rgb if rgb.dtype == np.uint8 else (np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    dt = time.time() - t0
    print(f"[check] {args.steps} steps in {dt:.1f}s ({args.steps / dt:.1f} step/s), active={active['on']}, {device.status()}")

    acts = np.stack(rows)
    with open(f"{args.out}/finger_joints.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + [f"a{i}" for i in range(acts.shape[1])])
        for i, a in enumerate(acts):
            w.writerow([i] + [f"{v:.4f}" for v in a])
    span = acts.max(axis=0) - acts.min(axis=0)
    print(f"[check] action range per dim (max-min): {np.round(span, 3).tolist()}")
    fps = 1.0 / (env.step_dt * max(args.video_every, 1))
    for n, fr in frames.items():
        if fr:
            imageio.mimwrite(f"{args.out}/manus_{n}.mp4", fr, fps=fps, codec="libx264", quality=8, macro_block_size=1)
            print(f"[check] video {args.out}/manus_{n}.mp4 ({len(fr)} frames)")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
