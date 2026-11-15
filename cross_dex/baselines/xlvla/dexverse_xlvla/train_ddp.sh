#!/bin/bash
# Multi-GPU (DDP) run of the common LeRobot DP training script (teleop_dp_image/train_lerobot.py), unchanged, through
# accelerate: the LeRobot 0.4.2 training loop is accelerate-based, so only the launcher differs from a single-GPU run.
# Same schedule as one GPU with batch 64: per-GPU batch = BATCH/N (effective BATCH), and since train_lerobot.py derives
# steps/epoch from --batch_size, it gets EPOCHS/N and SNAP/N -> same total steps, a snapshot every SNAP data epochs.
#   GPUS=0,1 bash train_ddp.sh --dataset_root <dataset> --output_dir <dir> [--resize] [--resume] [train_lerobot.py args]
# --resume continues with the saved config (per-GPU batch), so resume with the same GPUS.
set -eu
GPUS=${GPUS:-0,1}; N=$(echo "$GPUS" | tr , '\n' | wc -l)
BATCH=${BATCH:-64}; EPOCHS=${EPOCHS:-300}; SNAP=${SNAP:-50}
for v in BATCH EPOCHS SNAP; do (( ${!v} % N == 0 )) || { echo "[train_ddp] $v=${!v} is not divisible by $N GPUs"; exit 1; }; done
SERVER=${SERVER:-$(cd /workspace/experiments && ls -d */ | tr -d / | grep -v '^[_.]' | head -1)}   # the container's one <server> folder
COMMON=${XLVLA_COMMON_DP:-/workspace/experiments/$SERVER/20261031_teleop_single/teleop_dp_image}
LP=${LEROBOT_PYTHON:-/workspace/envs/dexverse/lerobot/bin/python}
PORT=${PORT:-$((29500 + RANDOM % 1000))}
cd "$COMMON"
echo "[train_ddp] GPUs $GPUS: batch $((BATCH / N)) x $N, epochs $((EPOCHS / N)), snapshot_epochs $((SNAP / N)) (of the per-GPU batch)"
CUDA_VISIBLE_DEVICES=$GPUS exec "$LP" -m accelerate.commands.launch --multi_gpu --num_processes "$N" --num_machines 1 \
  --mixed_precision no --dynamo_backend no --main_process_port "$PORT" \
  train_lerobot.py --batch_size $((BATCH / N)) --epochs $((EPOCHS / N)) --snapshot_epochs $((SNAP / N)) "$@"
