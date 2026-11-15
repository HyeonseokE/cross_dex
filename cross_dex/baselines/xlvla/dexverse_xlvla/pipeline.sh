#!/bin/bash
# XL-VLA reproduction on DexVerse, steps 3-6, after the step-2 latent training of run $LATENT_RUN (code-based project).
# Runs in the container (nohup); every step writes into its own run folder of the project and skips work already done
# (relaunch with the same variables, incl. STAMP, to resume).
#   LATENT_RUN=<run_id> GPU=1 bash pipeline.sh                                   (a4, the server of step 2)
#   SERVER=vast-isaac LATENT_RUN=<run_id> LATENT_CKPT=<pt> WRIST=<json> GPU_EVAL=1 GPU_TRAIN=0 \
#     XLVLA_TELEOP_ROOT=<a4 teleop copy> XLVLA_PKL_ROOT=<a4 pkl copy> JOINT_CK=<a4 joint DP copy> bash pipeline.sh
# Two lanes: EVAL GPU = step 4 replay -> step 6 joint-DP evaluation -> step 5 snapshot selection (as snapshots appear);
#            TRAIN GPU = step 5 dataset + training; final evaluation is split over both GPUs.
# Steps 4 and 6-joint can be skipped when their results exist from another run (e.g. the server running them went away):
#   SKIP_REPLAY=1 SKIP_JOINT=1 JOINT_ACTIONS=<actions_joint dir of that run> -> only step 5 and the step-6 smoothness run here.
# Common code is only called: evaluator (teleop_dp_image/eval_lerobot.py via run_eval.py), LeRobot training
# (teleop_dp_image/train_lerobot.py), DexVerse (main env). Baseline code: dexverse_xlvla/.
set -u
SERVER=${SERVER:-$(cd /workspace/experiments && ls -d */ | tr -d / | grep -v '^[_.]' | head -1)}; GPU=${GPU:-1}; GPU_EVAL=${GPU_EVAL:-$GPU}; GPU_TRAIN=${GPU_TRAIN:-$GPU}
TASK=Dexverse-GraspKettle-v0; HANDS="shadow allegro inspire leap sharpa wuji"
P=${PROJECT:-/workspace/experiments/$SERVER/RSS2027/baseline-reproduce/dexverse-xl-vla-reproduce/code-based}
R2=$P/$LATENT_RUN; CK2=/workspace/checkpoints/dexverse/xlvla/$LATENT_RUN
CKPT=${LATENT_CKPT:-$CK2/checkpoint_epoch_1000.pt}; WRIST=${WRIST:-$R2/artifacts/wrist_calib.json}
PY=/workspace/envs/dexverse/dexlatent/bin/python; LP=/workspace/envs/dexverse/lerobot/bin/python
export XLVLA_COMMON_DP=${XLVLA_COMMON_DP:-/workspace/experiments/$SERVER/20261031_teleop_single/teleop_dp_image}
COMMON=$XLVLA_COMMON_DP; PKL_ROOT=${XLVLA_PKL_ROOT:-/workspace/datasets/dexverse/teleop_pkl_bs1}
JOINT_CK=${JOINT_CK:-/workspace/checkpoints/20261031_teleop_single}
JOINT_RUN=${JOINT_RUN:-20261101-130737_a4_graspkettle}   # joint-DP comparison runs (trained on a4): <JOINT_RUN>-<hand>/best
CODE=/workspace/codes/dexverse/DexLatent
SKIP_REPLAY=${SKIP_REPLAY:-0}; SKIP_JOINT=${SKIP_JOINT:-0}
LSTAMP=${LATENT_RUN%%_*}; STAMP=${STAMP:-$LSTAMP}       # run ids of steps 4-6 (default: the step-2 time stamp)
R4=$P/${STAMP}_${SERVER}_latent-replay; R5=$P/${STAMP}_${SERVER}_latent-dp; R6=$P/${STAMP}_${SERVER}_smoothness
JOINT_ACTIONS=${JOINT_ACTIONS:-$R6/actions_joint}
[ $SKIP_JOINT = 1 ] && [ "$JOINT_ACTIONS" = "$R6/actions_joint" ] && { echo "SKIP_JOINT=1 needs JOINT_ACTIONS"; exit 1; }
LANE_LOG=$R4/logs/pipeline.log; [ $SKIP_REPLAY = 1 ] && LANE_LOG=$R5/logs/eval_lane.log
CK5=/workspace/checkpoints/dexverse/xlvla/${STAMP}_${SERVER}_latent-dp; DATA=/workspace/datasets/dexverse/xlvla_latent/$LSTAMP
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
log() { echo "[pipeline $(date -u -d '+9 hours' +%m-%d\ %H:%M)] $*"; }
pkl() { ls $PKL_ROOT/*/$TASK/${TASK}_*_floating_$1_right.pkl; }
n_done() { $PY -c "import json,sys; print(json.load(open(sys.argv[1]))['num_episodes'])" $1/metrics.json 2>/dev/null || echo 0; }
n_succ() { $PY -c "import json,sys; print(sum(e['success'] for e in json.load(open(sys.argv[1]))['episodes']))" $1/metrics.json 2>/dev/null || echo 0; }
evaluate() {  # demos policy_dir out episodes(comma) [--no-video]   (main env, Isaac Sim; MAX_STEPS default 3000; GPU = $EGPU)
  local want; want=$(echo $4 | tr , '\n' | wc -l)
  [ "$(n_done $3)" = "$want" ] && return 0
  rm -rf $3; mkdir -p $3
  ( source /workspace/codes/_setup/dexverse_env.sh DexVerse-develop >/dev/null; cd $DEXVERSE_ROOT
    unset DEXVERSE_FINGER_DRIVE DEXVERSE_DRIVE_SPECS
    CUDA_VISIBLE_DEVICES=$EGPU python $CODE/dexverse_xlvla/run_eval.py --demos $1 --ckpt $2 --output_dir $3 \
      --episodes $4 --max_steps ${MAX_STEPS:-3000} ${5:-} --headless ) > $3.log 2>&1
  [ "$(n_done $3)" = "$want" ]
}
eps() { seq -s, $1 $2; }
rmax() { $PY -c "import json,sys; print(json.load(open(sys.argv[1]))['max_steps'])" $1/replay.json; }
dp_json() {  # hand snapshot log_dir
  echo "{\"mode\": \"latent_dp\", \"hand\": \"$1\", \"policy_ckpt\": \"$CK5/checkpoints/$2/pretrained_model\", \"latent_ckpt\": \"$CKPT\", \"wrist\": \"$WRIST\", \"task\": \"$TASK\", \"log_dir\": $3}"
}
bash /workspace/codes/_setup/gpu_watch.sh ensure >/dev/null 2>&1

# ---- GPU assignment (GPU_WAIT=1): TRAIN = the GPU whose foreign training ends first, EVAL = the other one, used once
# its foreign training ends too. "foreign training" = a train_lerobot.py process with CUDA_VISIBLE_DEVICES=<g> that is not
# this pipeline's (output_dir != $CK5). Evaluations of other sessions do not block (they share the GPU).
foreign_train() {  # gpu -> number of foreign training processes on it
  local n=0 p
  for p in $(pgrep -f "train_lerobot.py --dataset_root"); do
    tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q -- "--output_dir $CK5" && continue
    tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -qx "CUDA_VISIBLE_DEVICES=$1" && n=$((n + 1))
  done
  echo $n
}
wait_free() { until [ "$(foreign_train $1)" = 0 ]; do sleep 60; done; }
if [ "${GPU_WAIT:-0}" = 1 ]; then
  log "waiting for a GPU without foreign training (0: $(foreign_train 0), 1: $(foreign_train 1))"
  while true; do
    [ "$(foreign_train 0)" = 0 ] && { GPU_TRAIN=0; GPU_EVAL=1; break; }
    [ "$(foreign_train 1)" = 0 ] && { GPU_TRAIN=1; GPU_EVAL=0; break; }
    sleep 60
  done
  log "GPU_TRAIN=$GPU_TRAIN GPU_EVAL=$GPU_EVAL (eval lane starts when GPU $GPU_EVAL has no foreign training)"
fi

# ---- step 2 (this server only) / step 3 -------------------------------------------------------------------------
if [ -d $R2 ]; then
  until [ -f $CKPT ]; do sleep 60; done
  mkdir -p $R2/artifacts; cp -n /workspace/.tmp/xlvla/wrist_calib.json $WRIST
  log "step 2 checkpoint ready: $CKPT"
  if [ ! -f $R2/artifacts/latent_eval.json ]; then
    $PY -m dexverse_xlvla.eval_latent --run_dir $R2 --ckpt $CKPT > $R2/logs/eval_latent.log 2>&1 && log "step 3 done" || log "step 3 FAILED"
  fi
fi
[ -f $CKPT ] && [ -f $WRIST ] || { log "missing latent checkpoint / wrist calibration"; exit 1; }

# ---- step 4: latent replay ---------------------------------------------------------------------------------------
lane_replay() {
  local EGPU=$GPU_EVAL
  mkdir -p $R4/logs $R4/replay
  for h in $HANDS; do                                    # self: A -> latent -> A, all 50 episodes
    o=$R4/replay/self_$h
    [ -f $o/policy/server.json ] || $PY -m dexverse_xlvla.replay --src $h --dst $h --mode latent --ckpt $CKPT \
      --wrist $WRIST --out $o >> $R4/logs/build.log 2>&1
    MAX_STEPS=$(rmax $o) evaluate $(pkl $h) $o/policy $o/eval $(eps 0 49) --no-video \
      && log "step 4 self $h: $(n_succ $o/eval)/$(n_done $o/eval)" || log "step 4 self $h FAILED"
  done
  for a in $HANDS; do for b in $HANDS; do [ $a = $b ] && continue   # cross: A -> latent -> B, episodes 0-9
    o=$R4/replay/${a}_to_$b
    [ -f $o/policy/server.json ] || $PY -m dexverse_xlvla.replay --src $a --dst $b --mode latent --ckpt $CKPT \
      --wrist $WRIST --out $o --episodes 0-9 >> $R4/logs/build.log 2>&1
    MAX_STEPS=$(rmax $o) evaluate $o/demos.pkl $o/policy $o/eval $(eps 0 9) \
      && log "step 4 $a->$b: $(n_succ $o/eval)/$(n_done $o/eval)" || log "step 4 $a->$b FAILED"
  done; done
  touch $R4/logs/replay_done
}

# ---- step 6 evaluation: joint-space DP (best checkpoints of teleop_dp_image) with action logs ---------------------
lane_joint() {
  local EGPU=$GPU_EVAL
  mkdir -p $R6/logs $R6/artifacts
  for h in $HANDS; do
    c=$JOINT_CK/$JOINT_RUN-$h/best
    [ -d $c ] || { log "step 6 joint DP $h: no best checkpoint"; continue; }
    o=$R6/joint_dp/$h; mkdir -p $o
    echo "{\"mode\": \"joint_dp\", \"policy_ckpt\": \"$c\", \"log_dir\": \"$R6/actions_joint/$h\"}" > $o/server.json
    evaluate $(pkl $h) $o $o/eval $(eps 0 49) && log "step 6 joint DP $h: $(n_succ $o/eval)/50" || log "step 6 joint DP $h FAILED"
  done
  touch $R6/logs/joint_done
}

# ---- step 5 snapshot selection: 10 episodes (0-9) per hand, as snapshots appear; best = most successes (ties -> later)
snap_ready() { [ -f $CK5/checkpoints/$1/pretrained_model/model.safetensors ] && [ -d $CK5/checkpoints/$1/training_state ]; }
lane_select() {
  local EGPU=$GPU_EVAL
  mkdir -p $R5/eval/select
  while true; do
    local fin=0; [ -f $R5/logs/trained ] || [ -f $R5/logs/train_failed ] && fin=1
    for s in $(ls $CK5/checkpoints 2>/dev/null | grep -E '^[0-9]+$' | sort); do
      grep -q "^$s " $R5/logs/select.log 2>/dev/null && continue
      snap_ready $s || continue
      sleep 60                                            # let the snapshot writer finish
      local n=0
      for h in $HANDS; do
        o=$R5/eval/select/$s/$h; mkdir -p $o; dp_json $h $s null > $o/server.json
        evaluate $(pkl $h) $o $o/eval $(eps 0 9) --no-video || log "step 5 select $s $h FAILED"
        n=$((n + $(n_succ $o/eval)))
      done
      echo "$s $n/60" >> $R5/logs/select.log; log "step 5 snapshot $s: $n/60"
    done
    [ $fin = 1 ] && break
    sleep 300
  done
}

# ---- step 5 dataset + training -----------------------------------------------------------------------------------
lane_train() {
  mkdir -p $R5/logs $R5/eval
  if [ ! -f $DATA/all/meta/info.json ]; then
    $PY -m dexverse_xlvla.make_dataset --ckpt $CKPT --wrist $WRIST --out $DATA > $R5/logs/make_dataset.log 2>&1 \
      || { log "step 5 dataset FAILED"; touch $R5/logs/train_failed; return 1; }
    log "step 5 dataset ready: $DATA/all"
  fi
  if [ ! -f $R5/logs/trained ]; then                      # common training script, unchanged (DP schedule of the joint DP)
    ( cd $COMMON && CUDA_VISIBLE_DEVICES=$GPU_TRAIN $LP train_lerobot.py --dataset_root $DATA/all --output_dir $CK5 --resize \
        $( [ -d $CK5/checkpoints/last ] && echo --resume ) ) >> $R5/logs/train.log 2>&1 \
      && touch $R5/logs/trained || { log "step 5 training FAILED"; touch $R5/logs/train_failed; return 1; }
    log "step 5 trained"
  fi
}

final_hands() {  # GPU hands...   final: 50 episodes per hand with action logs (step 6)
  local EGPU=$1; shift
  for h in "$@"; do
    o=$R5/eval/final/$h; mkdir -p $o; dp_json $h $BEST "\"$R5/eval/actions/$h\"" > $o/server.json
    evaluate $(pkl $h) $o $o/eval $(eps 0 49) && log "step 5 final $h: $(n_succ $o/eval)/50" || log "step 5 final $h FAILED"
  done
}

# lerobot into the baseline env before the lanes use it (installing later would change the env under running servers)
mkdir -p $R5/logs $R6/logs; [ $SKIP_REPLAY = 1 ] || mkdir -p $R4/logs
bash /workspace/codes/_setup/envs/dexverse/dexlatent.sh > $R5/logs/env.log 2>&1 || { log "dexlatent env FAILED"; exit 1; }
( [ "${GPU_WAIT:-0}" = 1 ] && { wait_free $GPU_EVAL; log "eval lane start on GPU $GPU_EVAL"; }
  [ $SKIP_REPLAY = 1 ] || lane_replay; [ $SKIP_JOINT = 1 ] || lane_joint; lane_select ) > $LANE_LOG 2>&1 &
A=$!
lane_train > $R5/logs/pipeline.log 2>&1
wait $A
[ -f $R5/logs/trained ] || { log "step 5 not trained, stop"; exit 1; }
BEST=$(awk '{split($2,a,"/"); if (a[1] + 0 >= b) {b = a[1] + 0; s = $1}} END {print s}' $R5/logs/select.log)  # ties -> later
[ -n "$BEST" ] || { log "step 5 no snapshot evaluated"; exit 1; }
grep "^$BEST " $R5/logs/select.log > $R5/eval/best.txt
log "step 5 best snapshot: $(cat $R5/eval/best.txt)"
final_hands $GPU_EVAL shadow inspire sharpa >> $R5/logs/pipeline.log 2>&1 &
B=$!
final_hands $GPU_TRAIN allegro leap wuji >> $R5/logs/pipeline.log 2>&1
wait $B
touch $R5/logs/dp_done

# ---- step 6: smoothness, latent DP vs joint-space DP -------------------------------------------------------------
$PY -m dexverse_xlvla.smoothness --log latent=$R5/eval/actions --log joint=$JOINT_ACTIONS \
  --out $R6/artifacts/smoothness.csv > $R6/logs/smoothness.log 2>&1 && log "step 6 done" || log "step 6 FAILED"
log "ALL DONE"
