#!/bin/bash
# Step 6 (smoothness) data: roll out the latent DP and the joint-space DP on the same cells with action logging
# (dexverse_xlvla.policy log_dir -> <R>/actions/<policy>/<task>/<hand>/epNNN.npz), then dexverse_xlvla.smoothness per task.
#   R=<run dir> TASKS=<t1,t2> HANDS=<h1,h2> LATENT_POLICY=<multi-task latent DP pretrained_model> LATENT=<latent.pt>
#   WRIST_DIR=<dir> JOINT_ROOT=<dir with <task>/<hand>/pretrained model (config.json ...)> SLOTS="0 0 0 1 1 1 1" \
#   EPS=0,1,2,3,4 SAMPLER=DDIM:10 bash dexverse_xlvla/eval_smooth.sh
# ONEHOT_TASKS=none: latent policy without task one-hot (single-task). Jobs whose metrics.json already has all EPS are skipped
# (e.g. joint rollouts copied from an earlier run).
# One job = (policy, task, hand); SLOTS = one GPU id per parallel job (7.1 N per GPU), started 90 s apart. Finished jobs
# (metrics.json with all EPS) are skipped, so a rerun resumes.
set -u
: "${R:?}" "${TASKS:?}" "${HANDS:?}" "${LATENT_POLICY:?}" "${LATENT:?}" "${WRIST_DIR:?}" "${JOINT_ROOT:?}"
SLOTS=${SLOTS:-0}; EPS=${EPS:-0,1,2,3,4}; MAX_STEPS=${MAX_STEPS:-1500}; SAMPLER=${SAMPLER:-DDIM:10}; WT=${DEXVERSE_WORKTREE:-DexVerse}
ONEHOT_TASKS=${ONEHOT_TASKS:-Dexverse-GraspBleach-v0,Dexverse-GraspPan-v0,Dexverse-GraspKettle-v0,Dexverse-FunctionalHammerStrike-v0}
PKL_ROOT=${XLVLA_PKL_ROOT:-/workspace/datasets/dexverse/teleop_pkl_bs1}
export XLVLA_COMMON_DP=${XLVLA_COMMON_DP:-/workspace/codes/dexverse/$WT/baselines/lerobot_dp}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); PY=/workspace/envs/dexverse/dexlatent/bin/python
WANT=$(echo $EPS | tr , '\n' | wc -l)
ONEHOT=$(python3 -c "import json,sys; print(json.dumps(sys.argv[1].split(',')))" $ONEHOT_TASKS)
[ "$ONEHOT_TASKS" = none ] && ONEHOT=null   # single-task latent DP trained without a task one-hot
SJ=; [ -n "$SAMPLER" ] && SJ="\"noise_scheduler_type\": \"${SAMPLER%%:*}\", \"num_inference_steps\": ${SAMPLER##*:}, "
mkdir -p $R/logs $R/artifacts
log() { echo "[smooth $(date -u -d '+9 hours' +%m-%d\ %H:%M)] $*" >> $R/logs/eval_smooth.log; }
n_ep() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['num_episodes'])" $1/metrics.json 2>/dev/null || echo 0; }
bash /workspace/codes/_setup/gpu_watch.sh ensure >/dev/null 2>&1

job() {  # gpu policy task hand
  local g=$1 p=$2 t=$3 h=$4 o=$R/eval/$2/$3/$4 pk
  [ "$(n_ep $o/eval)" -ge $WANT ] && return 0
  pk=$(ls $PKL_ROOT/*/$t/${t}_*_floating_${h}_right.pkl | head -1); mkdir -p $o; rm -rf $o/eval $R/actions/$p/$t/$h
  if [ $p = latent ]; then
    echo "{\"mode\": \"latent_dp\", \"hand\": \"$h\", \"policy_ckpt\": \"$LATENT_POLICY\", \"latent_ckpt\": \"$LATENT\", \"wrist\": \"$WRIST_DIR/wrist_calib_$t.json\", \"task\": \"$t\", \"task_onehot\": $ONEHOT, $SJ\"log_dir\": \"$R/actions/$p/$t/$h\"}" > $o/server.json
  else
    until [ -s $JOINT_ROOT/$t/$h/model.safetensors ]; do sleep 60; done   # joint model still downloading
    echo "{\"mode\": \"joint_dp\", \"policy_ckpt\": \"$JOINT_ROOT/$t/$h\", $SJ\"log_dir\": \"$R/actions/$p/$t/$h\"}" > $o/server.json
  fi
  ( source /workspace/codes/_setup/dexverse_env.sh $WT >/dev/null; cd $DEXVERSE_ROOT; unset DEXVERSE_FINGER_DRIVE DEXVERSE_DRIVE_SPECS
    timeout 3h env CUDA_VISIBLE_DEVICES=$g python $CODE/dexverse_xlvla/run_eval.py --demos $pk --ckpt $o --output_dir $o/eval \
      --episodes $EPS --max_steps $MAX_STEPS --headless ) > $o/eval.log 2>&1
  log "$p $t $h: $(n_ep $o/eval)/$WANT episodes, $(ls $R/actions/$p/$t/$h/ep*.npz 2>/dev/null | wc -l) action logs"
}

QUEUE=$R/logs/.queue; : > $QUEUE
for p in latent joint; do for t in ${TASKS//,/ }; do for h in ${HANDS//,/ }; do echo "$p $t $h" >> $QUEUE; done; done; done   # latent first: joint models may still be arriving
slot() {  # gpu: take queue lines one by one (flock)
  local g=$1 line
  while true; do
    line=$(flock $QUEUE.lock bash -c "head -1 $QUEUE; sed -i 1d $QUEUE")
    [ -z "$line" ] && break
    job $g $line
  done
}
log "start: tasks $TASKS, hands $HANDS, slots $SLOTS, EPS $EPS, sampler $SAMPLER"
for g in $SLOTS; do slot $g & sleep 90; done
wait
for t in ${TASKS//,/ }; do
  (cd $CODE && $PY -m dexverse_xlvla.smoothness --task $t --log latent=$R/actions/latent/$t --log joint=$R/actions/joint/$t \
     --out $R/artifacts/smoothness_$t.csv) > $R/logs/smoothness_$t.log 2>&1 || log "smoothness $t FAILED"
done
log "ALL DONE"
