#!/bin/bash
# ours: one-arm pi0.5 multitask on our latent spaces (E-a, E-h, E2c), same recipe as the XL-VLA / joint pi0.5 baselines
# (dexverse_xlvla.pi05.openpi_train: 41 dims = wrist 9 + finger latent 32, 20k steps, batch 32, warmup 1000, no EMA).
# Waits until the given GPU is free of other users' jobs (checked every CHECK seconds), then trains the spaces one after
# another (finished ones skipped, an interrupted one resumes). Workspace root from WORKSPACE (default /workspace).
#   WORKSPACE=<ws> GPU=3 [SEED=43] bash train_mt.sh [ea eh e2c]
# SEED (default 42; repeats 43, 44 per EXPERIMENT_RULES 4) -> names get a -s<seed> suffix except for 42.
# DIAG=1 (2026-11-24): composition generalization = the multitask set without the 4 diagonal cells of the XL-VLA pi05
#   diagonal baseline (GB:shadow, GP:inspire, GK:sharpa, FH:wuji); dataset ours_latent/pi05-6hand-mt-goal-diag-<tag>/all
#   (dexverse_xlvla.pi05.subset_v3), names get -diag. SHARE=1: start when FREE_GB is free even if other jobs use the GPU.
set -u
WS=${WORKSPACE:-/workspace}; GPU=${GPU:-0}; CHECK=${CHECK:-900}; FREE_GB=${FREE_GB:-150}
TAGS=${*:-ea eh e2c}
SEED=${SEED:-42}; SFX=; [ $SEED = 42 ] || SFX=-s$SEED
V=; [ -n "${DIAG:-}" ] && V=-diag
SERVER=${SERVER:-$(find $WS/experiments -mindepth 1 -maxdepth 1 -type d ! -name '_*' ! -name '.*' -printf '%f\n' | head -1)}
OP=$WS/envs/dexverse/openpi/bin/python; L=$WS/logs/pi05_ours; mkdir -p $L
B=$WS/experiments/$SERVER/RSS2027/ours/dexverse-ours-pi05/pi05-space-mt
export OPENPI_DATA_HOME=$WS/.cache/openpi XLVLA_PI05_OUT=$WS/checkpoints/dexverse/xlvla/pi05 OPENPI_ROOT=$WS/codes/dexverse/openpi \
  XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 HF_HOME=$WS/.cache/hf
# video decoding (torchcodec) needs FFmpeg 4-7 shared libs: apt ffmpeg in containers, conda-forge ffmpeg in
# $WS/envs/tools/ffmpeg on servers without root (appended, so system libraries keep priority)
[ -d $WS/envs/tools/ffmpeg/lib ] && export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$WS/envs/tools/ffmpeg/lib
cd $WS/codes/dexverse/DexLatent
log() { echo "$(date -u -d '+9 hours' '+%m-%d %H:%M') $*" | tee -a $L/main.log; }
declare -A SPACE=([ea]="E-a xlvla-local-pair-allegrofix" [eh]="E-h xlvla-local-pair-inspire60-hn" [e2c]="E2c xlvla-local-pair-staged-cont60k")

gpu_free() {  # MiB used on $GPU by processes, total MiB free
  local free; free=$(nvidia-smi -i $GPU --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
  [ "$free" -ge $((FREE_GB * 1024)) ] && { [ -n "${SHARE:-}" ] || [ -z "$(nvidia-smi -i $GPU --query-compute-apps=pid --format=csv,noheader)" ]; }
}

for t in $TAGS; do
  set -- ${SPACE[$t]}; name=$1; hf=$2
  root=$WS/datasets/dexverse/ours_latent/pi05-6hand-mt-goal$V-$t/all; run=ours-pi05-6hand-mt-goal$V-$t; repo=pi05-6hand-mt-goal$V-$t; exp=$t-mt$V-bs32$SFX
  ck=$XLVLA_PI05_OUT/checkpoints/$run/$exp; A="--root $root --repo_id $repo --run $run --action_dim 41"
  if [ -d $ck/19999 ]; then log "$t$V$SFX: already trained"; continue; fi
  mkdir -p $L/$t
  [ -f $XLVLA_PI05_OUT/assets/$run/$repo/norm_stats.json ] || { log "$t$V$SFX: norm"; $OP -m dexverse_xlvla.pi05.openpi_train norm $A > $L/$t/norm$V$SFX.log 2>&1 || { log "$t$V$SFX: norm FAILED"; exit 1; }; }
  until gpu_free; do log "$t$V$SFX: GPU$GPU busy ($(nvidia-smi -i $GPU --query-gpu=memory.used,utilization.gpu --format=csv,noheader)), next check in $((CHECK / 60)) min"; sleep $CHECK; done
  R=$(ls -d $B/*_${SERVER}_pi05-$t-mt$V-bs32$SFX 2>/dev/null | tail -1)
  if [ -z "$R" ]; then
    R=$B/$(date -u -d '+9 hours' +%Y%m%d-%H%M%S)_${SERVER}_pi05-$t-mt$V-bs32$SFX; mkdir -p $R/logs $R/artifacts
    cat > $R/meta.json <<JSON
{"run_id": "$(basename $R)", "project": "pi05-space-mt", "server": "$SERVER", "agent": "claude-code",
 "git_repo": "HyeonseokE/cross_dex (baselines/xlvla/dexverse_xlvla) + openpi", "git_commit": "$(git -C $WS/codes/cross_dex rev-parse --short HEAD 2>/dev/null || echo 기록 없음)", "git_dirty": null,
 "docker_image": null, "command": "GPU=$GPU SEED=$SEED DIAG=${DIAG:-} bash cross_dex/ours/pi05_space/train_mt.sh $t -> openpi_train train $A --exp $exp --steps 20000 --accum 1 --warmup 1000 --batch 32 --ema 0 --workers 8 --save_interval 2000 --seed $SEED",
 "seed": $SEED, "latent_space": "ours: $name (HF Dexverse-CSI/shared-action-spaces ablation/dexverse/$hf/right-6hand/model.pt)",
 "start_time": "$(date -u -d '+9 hours' +%Y-%m-%dT%H:%M:%S+09:00)", "end_time": null, "status": "running"}
JSON
  fi
  mode=--overwrite; ls -d $ck/[0-9]* >/dev/null 2>&1 && mode=--resume
  log "$t$V$SFX: train on GPU$GPU ($mode) -> $R"
  CUDA_VISIBLE_DEVICES=$GPU $OP -m dexverse_xlvla.pi05.openpi_train train $A --exp $exp --steps 20000 --accum 1 --warmup 1000 \
    --batch 32 --ema 0 --workers 8 --log_interval 100 --save_interval 2000 --seed $SEED $mode >> $L/$t/train$V$SFX.log 2>&1
  rc=$?; cp $L/$t/train$V$SFX.log $R/logs/train.log
  st=failed; [ -d $ck/19999 ] && st=done
  python3 - $R/meta.json $st <<'PY'
import json, sys, subprocess
p, st = sys.argv[1:3]; m = json.load(open(p)); m["status"] = st
m["end_time"] = subprocess.run(["date", "-Iseconds"], capture_output=True, text=True, env={"TZ": "Asia/Seoul"}).stdout.strip()
json.dump(m, open(p, "w"), ensure_ascii=False, indent=1)
PY
  log "$t$V$SFX: train exit $rc ($st)"
  [ $st = done ] || exit 1
done
log "ALL DONE"
