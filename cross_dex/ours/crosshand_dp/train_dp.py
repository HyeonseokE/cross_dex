#!/usr/bin/env python
"""Train LeRobot's Diffusion Policy on a multi-hand set (lerobot env), with a fixed step count and snapshot interval.

Same settings as the common teleop DP trainer (DexVerse baselines/lerobot_dp/train_lerobot.py: one RGB encoder per
camera, 240x320 resize in the data loader + crop 216x288, batch 64, LeRobot defaults otherwise), whose helper modules
are imported unchanged; only the schedule differs: --steps (200k) and --save_freq (50k) instead of epochs.

  /workspace/envs/dexverse/lerobot/bin/python train_dp.py --dataset_root <set> --output_dir <dir> [--resume]
"""
import argparse
import json
import os
import sys
from pathlib import Path

COMMON = os.environ.get("COMMON_DP", os.environ.get("WORKSPACE", "/workspace") + "/codes/dexverse/DexVerse/baselines/lerobot_dp")
sys.path.insert(0, COMMON)
sys.path.insert(0, os.environ.get("EXPERIMENTS_ROOT", os.environ.get("WORKSPACE", "/workspace") + "/experiments"))
from seed_control import DEFAULT_SEED, TRAIN_SEEDS, seed_everything  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--dataset_root", required=True)
p.add_argument("--output_dir", required=True)
p.add_argument("--seed", type=int, default=DEFAULT_SEED, choices=TRAIN_SEEDS)
p.add_argument("--steps", type=int, default=200000)
p.add_argument("--save_freq", type=int, default=50000)
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--num_workers", type=int, default=12)
p.add_argument("--resume", action="store_true")
args = p.parse_args()

from image_resize import RESIZED_CROP, install_dataset_resize, install_resize  # noqa: E402

info = json.loads((Path(args.dataset_root) / "meta/info.json").read_text())
import torch  # noqa: E402

if not torch.cuda.is_available():
    sys.exit("[train] CUDA is not available (GPU lost or MPS stuck): refusing to train on CPU")
seed_everything(args.seed)
os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", ACCELERATE_MIXED_PRECISION="no")
sys.argv = [sys.argv[0],
            f"--dataset.repo_id=local/{Path(args.dataset_root).name}", f"--dataset.root={args.dataset_root}",
            "--policy.type=diffusion", "--policy.device=cuda", "--policy.push_to_hub=false",
            f"--policy.crop_shape=[{RESIZED_CROP[0]},{RESIZED_CROP[1]}]", "--policy.use_separate_rgb_encoder_per_camera=true",
            f"--batch_size={args.batch_size}", f"--num_workers={args.num_workers}", f"--steps={args.steps}",
            f"--save_freq={args.save_freq}", "--eval_freq=0", "--log_freq=200", "--wandb.enable=false",
            f"--seed={args.seed}", f"--output_dir={args.output_dir}", f"--job_name={Path(args.output_dir).name}"]
if args.resume:
    config = Path(args.output_dir) / "checkpoints/last/pretrained_model/train_config.json"
    sys.argv = [sys.argv[0], "--resume=true", f"--config_path={config}", f"--steps={args.steps}"]
print(f"[train] {info['total_episodes']} episodes, {info['total_frames']} frames, {args.steps} steps "
      f"(~{args.steps * args.batch_size / info['total_frames']:.0f} epochs), save every {args.save_freq}, seed {args.seed}",
      flush=True)

from data_keys import install_needed_keys, install_worker_settings  # noqa: E402
from lerobot.scripts.lerobot_train import main  # noqa: E402

install_needed_keys()
install_worker_settings()
install_dataset_resize()
install_resize()
main()
