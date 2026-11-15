#!/bin/bash
# Single-task, single-hand latent DP with the joint-DP baseline's exact recipe (DexVerse baselines/lerobot_dp/run_sets.sh):
# only the action space differs (VAE latent [canonical wrist 9 | z 32] instead of joints). Per Task:hand, one after another on one GPU:
#   dataset (make_dataset --task T --hands h, no task one-hot) -> train_lerobot.py batch 64, EPOCHS 300, snapshot every 50 epochs,
#   --resize, seed 42 -> snapshot selection in the background like run_sets.sh: every snapshot except the final one rolled out on
#   pickle episodes 0-9 (max 3000 steps, trained sampler), best = most successes, ties to the later snapshot -> <ck>/best.
#   P=<project dir> STAMP=<stamp> LATENT=<latent.pt> WRIST_DIR=<dir> bash dexverse_xlvla/st_hand.sh <gpu> GraspPan-v0:shadow ...
# Relaunch with the same variables resumes (dataset / training / evaluated snapshots are skipped).
set -u
: "${P:?}" "${STAMP:?}" "${LATENT:?}" "${WRIST_DIR:?}"
GPU=$1; shift
EPOCHS=${EPOCHS:-300}; SNAP=${SNAPSHOT_EPOCHS:-50}; NW=${NUM_WORKERS:-24}; MAX_STEPS=${MAX_STEPS:-3000}
SERVER=${SERVER:-$(cd /workspace/experiments && ls -d */ | tr -d / | grep -v '^[_.]' | head -1)}
PY=/workspace/envs/dexverse/dexlatent/bin/python; LP=/workspace/envs/dexverse/lerobot/bin/python
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); WT=${DEXVERSE_WORKTREE:-DexVerse}
DP=/workspace/codes/dexverse/$WT/baselines/lerobot_dp
PKL_ROOT=${XLVLA_PKL_ROOT:-/workspace/datasets/dexverse/teleop_pkl_bs1}
export XLVLA_COMMON_DP=${XLVLA_COMMON_DP:-$DP} PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
ulimit -n 65536
bash /workspace/codes/_setup/gpu_watch.sh ensure >/dev/null 2>&1
n_ep() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['num_episodes'])" $1/metrics.json 2>/dev/null || echo 0; }
n_ok() { python3 -c "import json,sys; print(sum(e['success'] for e in json.load(open(sys.argv[1]))['episodes']))" $1/metrics.json; }

select_snapshots() {  # r c final t h : like run_sets.sh select_snapshots, with the latent policy server
  local r=$1 c=$2/checkpoints final=$3 t=$4 h=$5 s o ok best pk
  pk=$(ls $PKL_ROOT/*/Dexverse-$t/Dexverse-${t}_*_floating_${h}_right.pkl | head -1)
  mkdir -p $r/eval/select
  while true; do
    for s in $(ls $c 2>/dev/null | grep -E '^[0-9]+$' | sort); do
      [ $((10#$s)) = $final ] && continue
      [ -f $c/$s/training_state/training_step.json ] || continue
      o=$r/eval/select/$s
      if [ "$(n_ep $o/eval)" != 10 ]; then
        rm -rf $o; mkdir -p $o
        echo "{\"mode\": \"latent_dp\", \"hand\": \"$h\", \"policy_ckpt\": \"$c/$s/pretrained_model\", \"latent_ckpt\": \"$LATENT\", \"wrist\": \"$WRIST_DIR/wrist_calib_Dexverse-$t.json\", \"task\": \"Dexverse-$t\", \"log_dir\": null}" > $o/server.json
        ( source /workspace/codes/_setup/dexverse_env.sh $WT >/dev/null; cd $DEXVERSE_ROOT; unset DEXVERSE_FINGER_DRIVE DEXVERSE_DRIVE_SPECS
          timeout 4h env CUDA_VISIBLE_DEVICES=$GPU python $CODE/dexverse_xlvla/run_eval.py --demos $pk --ckpt $o --output_dir $o/eval \
            --episodes $(seq -s, 0 9) --max_steps $MAX_STEPS --no-video --headless ) > $o/eval.log 2>&1
        [ "$(n_ep $o/eval)" = 10 ] || { echo "[st] $(basename $r) $s: SELECT FAILED" >> $r/logs/main.log; continue; }
      fi
      if ! grep -q "^$s " $r/logs/select.log 2>/dev/null; then
        ok=$(n_ok $o/eval); best=$(cut -d' ' -f2 $r/eval/select/best.txt 2>/dev/null)
        echo "$s $ok/10" >> $r/logs/select.log
        if [ -z "$best" ] || [ $ok -ge ${best%/10} ]; then   # ties go to the later snapshot
          rm -rf $2/best.tmp; cp -r $c/$s/pretrained_model $2/best.tmp; rm -rf $2/best; mv $2/best.tmp $2/best; echo "$s $ok/10" > $r/eval/select/best.txt
        fi
      fi
      [ "$(readlink $c/last)" = $s ] || rm -rf $c/$s
    done
    [ -f $r/logs/trained ] && [ -z "$(for s in $(ls $c | grep -E '^[0-9]+$'); do [ $((10#$s)) = $final ] || [ "$(readlink $c/last)" = $s ] || echo x; done)" ] && { echo "[st] $(basename $r): selected $(cat $r/eval/select/best.txt)" >> $r/logs/main.log; return 0; }
    sleep 120
  done
}

for item in "$@"; do
  t=${item%%:*}; h=${item##*:}; id=${STAMP}_${SERVER}_latent-dp-sh-${t%-v0}-$h
  R=$P/$id; D=/workspace/datasets/dexverse/xlvla_latent/$id; CK=/workspace/checkpoints/dexverse/xlvla/$id; mkdir -p $R/logs
  log() { echo "[st $(date -u -d '+9 hours' +%m-%d\ %H:%M)] $*" >> $R/logs/main.log; }
  [ -s $R/meta.json ] || cat > $R/meta.json <<EOF
{"run_id": "$id", "project": "$(basename $P)", "server": "$SERVER", "agent": "claude-code",
 "git_repo": "https://github.com/EmptyBlueBox/DexLatent (branch dexverse)", "git_commit": "see sync", "git_dirty": false, "docker_image": null,
 "command": "P=$P STAMP=$STAMP LATENT=$LATENT WRIST_DIR=$WRIST_DIR bash dexverse_xlvla/st_hand.sh $GPU $item (recipe = baselines/lerobot_dp/run_sets.sh: EPOCHS $EPOCHS, snapshot $SNAP, batch 64)",
 "seed": 42, "latent_ckpt": "$LATENT", "task": "Dexverse-$t", "hand": "$h", "start_time": "$(date -u -d '+9 hours' +%Y-%m-%dT%H:%M:%S)+09:00", "end_time": null, "status": "running"}
EOF
  if [ ! -s $D/all/meta/info.json ]; then
    log "dataset"
    (cd $CODE && $PY -m dexverse_xlvla.make_dataset --ckpt $LATENT --wrist $WRIST_DIR --out $D --task Dexverse-$t --hands $h) >> $R/logs/make_dataset.log 2>&1 || { log "dataset FAILED"; continue; }
  fi
  frames=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['total_frames'])" $D/all/meta/info.json)
  final=$(( EPOCHS * ( (frames + 63) / 64 ) ))
  select_snapshots $R $CK $final $t $h & sel=$!
  while [ ! -f $R/logs/trained ]; do
    res=; [ -e $CK/checkpoints/last ] && res=--resume
    log "train: $frames frames, batch 64, $EPOCHS epochs = $final steps, GPU $GPU $res"
    (cd $DP && CUDA_VISIBLE_DEVICES=$GPU $LP train_lerobot.py --dataset_root $D/all --output_dir $CK --epochs $EPOCHS --snapshot_epochs $SNAP \
       --num_workers $NW --resize $res) >> $R/logs/train.log 2>&1
    [ "$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['step'])" $CK/checkpoints/last/training_state/training_step.json 2>/dev/null)" = $final ] \
      && touch $R/logs/trained || { log "train FAILED, retry in 5 min"; sleep 300; }
  done
  log "trained; snapshot selection continues in the background"
done
wait
