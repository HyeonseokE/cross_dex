#!/bin/bash
# Multi-task latent DP pipeline after a latent training run: wait for the latent checkpoint -> per run (NAME:GPU) build the
# latent dataset (make_dataset --task_onehot [--exclude]) -> train the common LeRobot DP on its own GPU -> final evaluation
# (eval_mt.sh, SAMPLER e.g. DDIM:10, WORKERS per GPU). Each step skips work already done -> relaunch with the same variables resumes.
#   P=<project dir> STAMP=<run id stamp> LATENT=<latent .pt> WRIST_DIR=<dir> RUNS="mt-all:0 mt-diag:1" BATCH=192 \
#     SAMPLER=DDIM:10 WORKERS=4 MAX_STEPS=1500 bash dexverse_xlvla/mt_pipeline.sh
# mt-all = all 16 cells, mt-diag = without the diagonal (hand i x task i), st-<Task> (e.g. st-GraspPan-v0) = one task, all hands.
# NO_EVAL=1: stop after training. DP_HANDS: hands of the dataset (default the 4 five-finger hands; mt-diag always leaves out the 4 five-finger diagonal cells). Same data passes as the batch-64 runs (SAMPLES).
set -u
: "${P:?}" "${STAMP:?}" "${LATENT:?}" "${WRIST_DIR:?}"
RUNS=${RUNS:-"mt-all:0 mt-diag:1"}; BATCH=${BATCH:-192}; SAMPLES=${SAMPLES:-12880000}
NW=${NUM_WORKERS:-14}    # data workers per run; 2 runs x 24 at batch 192 hit the 259 GB container memory limit (OOM kill, vast 11-09)
SNAPS=${SNAPS:-2}        # checkpoints per run (3.2 GB each with optimizer state)
SAMPLER=${SAMPLER:-DDIM:10}; WORKERS=${WORKERS:-4}; MAX_STEPS=${MAX_STEPS:-1500}
SERVER=${SERVER:-$(cd /workspace/experiments && ls -d */ | tr -d / | grep -v '^[_.]' | head -1)}
TASKS=Dexverse-GraspBleach-v0,Dexverse-GraspPan-v0,Dexverse-GraspKettle-v0,Dexverse-FunctionalHammerStrike-v0
DIAG=Dexverse-GraspBleach-v0:shadow,Dexverse-GraspPan-v0:inspire,Dexverse-GraspKettle-v0:sharpa,Dexverse-FunctionalHammerStrike-v0:wuji
PY=/workspace/envs/dexverse/dexlatent/bin/python; LP=/workspace/envs/dexverse/lerobot/bin/python
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); WT=${DEXVERSE_WORKTREE:-DexVerse}
DP=/workspace/codes/dexverse/$WT/baselines/lerobot_dp
ulimit -n 65536
bash /workspace/codes/_setup/gpu_watch.sh ensure >/dev/null 2>&1
now() { date -u -d '+9 hours' +%Y-%m-%dT%H:%M:%S+09:00; }

until [ -s "$LATENT" ]; do sleep 120; done   # latent training still running
sleep 30

lane() {  # name gpu
  local n=$1 g=$2 id=${STAMP}_${SERVER}_latent-dp-$1 R D CK ex frames spe ep
  R=$P/$id; D=/workspace/datasets/dexverse/xlvla_latent/$id; CK=/workspace/checkpoints/dexverse/xlvla/$id
  mkdir -p $R/logs
  log() { echo "[mt $(date -u -d '+9 hours' +%m-%d\ %H:%M)] $id: $*" >> $R/logs/main.log; }
  ex=; [ $n = mt-diag ] && ex="--exclude $DIAG"
  local tasks=$TASKS; [[ $n == st-* ]] && tasks=Dexverse-${n#st-}   # st-<Task>: single-task run, e.g. st-GraspPan-v0
  [ -s $R/meta.json ] || cat > $R/meta.json <<EOF
{"run_id": "$id", "project": "$(basename $P)", "server": "$SERVER", "agent": "claude-code",
 "git_repo": "https://github.com/EmptyBlueBox/DexLatent (branch dexverse)", "git_commit": "$(cat $CODE/.git_commit 2>/dev/null || echo see sync)", "git_dirty": false,
 "docker_image": null, "command": "LATENT=$LATENT RUNS='$RUNS' BATCH=$BATCH SAMPLER=$SAMPLER WORKERS=$WORKERS MAX_STEPS=$MAX_STEPS bash dexverse_xlvla/mt_pipeline.sh",
 "seed": 42, "latent_ckpt": "$LATENT", "wrist_dir": "$WRIST_DIR", "excluded": "${ex:-none}", "batch_size": $BATCH, "eval_sampler": "$SAMPLER",
 "start_time": "$(now)", "end_time": null, "status": "running"}
EOF
  if [ ! -s $D/all/meta/info.json ]; then
    log "dataset"
    (cd $CODE && $PY -m dexverse_xlvla.make_dataset --ckpt $LATENT --wrist $WRIST_DIR --out $D --task $tasks \
      --hands ${DP_HANDS:-shadow,inspire,sharpa,wuji} --task_onehot $ex) >> $R/logs/make_dataset.log 2>&1 || { log "dataset FAILED"; return 1; }
  fi
  frames=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['total_frames'])" $D/all/meta/info.json)
  spe=$(( (frames + BATCH - 1) / BATCH )); ep=$(( (SAMPLES + frames - 1) / frames ))
  if [ ! -d $CK/checkpoints/last ] || ! grep -q "End of training" $R/logs/train.log 2>/dev/null; then
    log "train: $frames frames, batch $BATCH -> $spe steps/epoch, $ep epochs = $((ep * spe)) steps, GPU $g"
    local res=; [ -d $CK/checkpoints/last ] && res=--resume
    (cd $DP && CUDA_VISIBLE_DEVICES=$g $LP train_lerobot.py --dataset_root $D/all --output_dir $CK --resize --batch_size $BATCH \
      --num_workers $NW --epochs $ep --snapshot_epochs $(( (ep + SNAPS - 1) / SNAPS )) $res) >> $R/logs/train.log 2>&1
    [ -d $CK/checkpoints/last ] && grep -q "End of training" $R/logs/train.log || { log "train FAILED"; return 1; }
    log "trained"
  fi
  [ "${NO_EVAL:-0}" = 1 ] && { log "eval skipped (NO_EVAL=1)"; return 0; }
  log "eval: sampler $SAMPLER, $WORKERS workers, max_steps $MAX_STEPS"
  R=$R CK=$CK LATENT=$LATENT WRIST_DIR=$WRIST_DIR GPU=$g WORKERS=$WORKERS MAX_STEPS=$MAX_STEPS SAMPLER=$SAMPLER \
    bash $CODE/dexverse_xlvla/eval_mt.sh
  log "evaluated"
}

for r in $RUNS; do lane ${r%%:*} ${r##*:} & sleep 5; done
wait
