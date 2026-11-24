#!/bin/bash
# ours: hand zero-shot pi0.5 (shadow held out) on our latent spaces, same recipe as the XL-VLA zero-shot baseline
# (pi05-hand-zeroshot: one task, 5 hands x 50 episodes, 5,000 steps, batch 32, warmup 250, no EMA, save every 1000).
# Datasets from make_zs.sh. Waits until the GPU is free of other jobs (as train_mt.sh), then trains one after another
# (finished skipped, interrupted resumed); SHARE=1: start when FREE_GB is free even if other jobs use the GPU. Workspace root from WORKSPACE (default /workspace).
#   WORKSPACE=<ws> GPU=3 [TAGS="ea eh e2c"] [TASKS="GraspBleach GraspKettle GraspPan"] bash train_zs.sh
set -u
WS=${WORKSPACE:-/workspace}; GPU=${GPU:-0}; CHECK=${CHECK:-900}; FREE_GB=${FREE_GB:-150}; SEED=${SEED:-42}
TAGS=${TAGS:-ea eh e2c}; TASKS=${TASKS:-GraspBleach GraspKettle GraspPan}
SERVER=${SERVER:-$(find $WS/experiments -mindepth 1 -maxdepth 1 -type d ! -name '_*' ! -name '.*' -printf '%f\n' | head -1)}
OP=$WS/envs/dexverse/openpi/bin/python; L=$WS/logs/pi05_ours/zs; mkdir -p $L
B=$WS/experiments/$SERVER/RSS2027/ours/dexverse-ours-pi05/pi05-space-zeroshot
export OPENPI_DATA_HOME=$WS/.cache/openpi XLVLA_PI05_OUT=$WS/checkpoints/dexverse/xlvla/pi05 OPENPI_ROOT=$WS/codes/dexverse/openpi \
  XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HOME=$WS/.cache/hf
[ -d $WS/envs/tools/ffmpeg/lib ] && export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$WS/envs/tools/ffmpeg/lib
cd $WS/codes/dexverse/DexLatent
now() { date -u -d '+9 hours' "$@"; }
log() { echo "$(now '+%m-%d %H:%M') $*" | tee -a $L/main.log; }
declare -A SPACE=([ea]="E-a xlvla-local-pair-allegrofix" [eh]="E-h xlvla-local-pair-inspire60-hn" [e2c]="E2c xlvla-local-pair-staged-cont60k")
gpu_free() {
  local free; free=$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  [ "$free" -ge $((FREE_GB * 1024)) ] && { [ -n "${SHARE:-}" ] || [ -z "$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader)" ]; }
}

for t in $TAGS; do for task in $TASKS; do
  set -- ${SPACE[$t]}; name=$1; hf=$2; k=${task,,}
  root=$WS/datasets/dexverse/ours_latent/pi05-$k-no-shadow-$t/all; run=ours-pi05-$k-no-shadow-$t; repo=pi05-$k-no-shadow-$t; exp=$t-5hand-5k
  ck=$XLVLA_PI05_OUT/checkpoints/$run/$exp; A="--root $root --repo_id $repo --run $run --action_dim 41"; id="$t $task"
  if [ -d $ck/4999 ]; then log "$id: already trained"; continue; fi
  [ -f $root/meta/info.json ] || { log "$id: dataset missing ($root)"; exit 1; }
  [ -f $XLVLA_PI05_OUT/assets/$run/$repo/norm_stats.json ] || { log "$id: norm"; $OP -m dexverse_xlvla.pi05.openpi_train norm $A > $L/norm_${k}_$t.log 2>&1 || { log "$id: norm FAILED"; exit 1; }; }
  until gpu_free; do log "$id: GPU$GPU busy ($(nvidia-smi -i $GPU --query-gpu=memory.used,utilization.gpu --format=csv,noheader)), next check in $((CHECK / 60)) min"; sleep $CHECK; done
  slug=pi05-$k-zs-$t; R=$(ls -d $B/*_${SERVER}_$slug 2>/dev/null | tail -1)
  if [ -z "$R" ]; then
    R=$B/$(now +%Y%m%d-%H%M%S)_${SERVER}_$slug; mkdir -p $R/logs $R/artifacts
    cat > $R/meta.json <<JSON
{"run_id": "$(basename $R)", "project": "pi05-space-zeroshot", "server": "$SERVER", "agent": "claude-code",
 "git_repo": "HyeonseokE/cross_dex (baselines/xlvla/dexverse_xlvla) + openpi", "git_commit": "$(git -C $WS/codes/cross_dex rev-parse --short HEAD 2>/dev/null || echo 기록 없음)", "git_dirty": null,
 "docker_image": null, "command": "GPU=$GPU bash cross_dex/ours/pi05_space/train_zs.sh ($t $task) -> openpi_train train $A --exp $exp --steps 5000 --accum 1 --warmup 250 --batch 32 --ema 0 --workers 8 --save_interval 1000 --seed $SEED",
 "seed": $SEED, "latent_space": "ours: $name (HF Dexverse-CSI/shared-action-spaces ablation/dexverse/$hf/right-6hand/model.pt)",
 "zeroshot": "Dexverse-$task-v0, train hands inspire,sharpa,wuji,allegro,leap, held out shadow",
 "start_time": "$(now +%Y-%m-%dT%H:%M:%S+09:00)", "end_time": null, "status": "running"}
JSON
  fi
  mode=--overwrite; ls -d $ck/[0-9]* >/dev/null 2>&1 && mode=--resume
  log "$id: train on GPU$GPU ($mode) -> $R"
  CUDA_VISIBLE_DEVICES=$GPU $OP -m dexverse_xlvla.pi05.openpi_train train $A --exp $exp --steps 5000 --accum 1 --warmup 250 \
    --batch 32 --ema 0 --workers 8 --log_interval 100 --save_interval 1000 --seed $SEED $mode >> $L/train_${k}_$t.log 2>&1
  rc=$?; cp $L/train_${k}_$t.log $R/logs/train.log
  st=failed; [ -d $ck/4999 ] && st=done
  python3 - $R/meta.json $st <<'PY'
import json, sys, subprocess
p, st = sys.argv[1:3]; m = json.load(open(p)); m["status"] = st
m["end_time"] = subprocess.run(["date", "-u", "-d", "+9 hours", "+%Y-%m-%dT%H:%M:%S+09:00"], capture_output=True, text=True).stdout.strip()
json.dump(m, open(p, "w"), ensure_ascii=False, indent=1)
PY
  log "$id: train exit $rc ($st)"
  [ $st = done ] || exit 1
done; done
log "ALL DONE"
