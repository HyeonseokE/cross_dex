#!/bin/bash
# Final evaluation of a multi-task latent DP run (make_dataset --task ... --task_onehot): every (task, hand) cell of TASKS x HANDS,
# 50 rollouts on pickle episodes 0-49 (common evaluator via run_eval.py, max_steps 3000 like the joint DP), WORKERS cells at
# a time on one GPU (MPS). Cells are claimed with mkdir, finished cells (metrics.json with 50 episodes) are skipped -> rerun resumes.
#   R=<run dir> CK=<training output dir> LATENT=<latent.pt> WRIST_DIR=<dir with wrist_calib_<task>.json> GPU=0 \
#     bash dexverse_xlvla/eval_mt.sh
# Per-cell target: echo <k> > $R/eval/final/<task>/<hand>/want stops that cell after k episodes (the rest runs elsewhere with EPS=k..);
#   read every 20 s, so it can be set while the cell runs.
# env: SNAP (checkpoint folder, default last), SAMPLER (e.g. DDIM:10), WORKERS (3), EPS (0-49), MAX_STEPS (3000), DEXVERSE_WORKTREE (DexVerse),
#      STOP_ON_KILL (1 = shared GPU: a cell that ends early is not retried and stops the whole run; rm <R>/eval/final/STOPPED to run again),
#      KEEP_CLAIMS (1 = add workers to a run whose manager is already running: its claims are kept),
#      TASKS / HANDS (comma lists; TASKS order = the one-hot order of make_dataset --task), ONLY_TASKS (evaluate only these tasks)
# pi0.5 (dexverse_xlvla/pi05/eval_policy.py): MODE=pi05_latent|pi05_joint|pi05_lad (LAD_CKPT=<lad_cam.pt>, PI05_ADIM=32), CK=<openpi checkpoint step dir>, PI05_RUN=<TrainConfig name>,
#      PI05_REPO=<norm-stats asset id>, PI05_ADIM=<41|32>, REPLAN (10); SNAP/SAMPLER unused
# bimanual pi0.5 (pi05/eval_bimanual.py): BIMANUAL=1 MODE=pi05_latent|pi05_joint, PI05_ADIM=82|62, PI05_MAXTOK=360|280,
#      LATENT_LEFT=<left 6-hand latent> (latent), WRIST_DIR=<folder of bimanual wrist_calib_<task>.json>,
#      TASKS=<the 4 bimanual tasks>, HANDS=<families>; the pickle of a cell = the one in its lerobot_conversions.jsonl
set -u
: "${R:?}" "${CK:?}" "${LATENT:?}" "${WRIST_DIR:?}"
GPU=${GPU:-0}; SNAP=${SNAP:-last}; WORKERS=${WORKERS:-3}; MAX_STEPS=${MAX_STEPS:-3000}; EPS=${EPS:-$(seq -s, 0 49)}
TASKS=${TASKS:-Dexverse-GraspBleach-v0,Dexverse-GraspPan-v0,Dexverse-GraspKettle-v0,Dexverse-FunctionalHammerStrike-v0}
HANDS=${HANDS:-shadow,inspire,sharpa,wuji}; WT=${DEXVERSE_WORKTREE:-DexVerse}
PKL_ROOT=${XLVLA_PKL_ROOT:-/workspace/datasets/dexverse/teleop_pkl_bs1}
export XLVLA_COMMON_DP=${XLVLA_COMMON_DP:-/workspace/codes/dexverse/$WT/baselines/lerobot_dp}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MODE=${MODE:-latent_dp}
if [[ $MODE == pi05_* ]]; then POLICY=$(readlink -f $CK); else POLICY=$(readlink -f $CK/checkpoints/$SNAP)/pretrained_model; fi
WANT=$(echo $EPS | tr , '\n' | wc -l)
ONEHOT=$(python3 -c "import json,sys; print(json.dumps(sys.argv[1].split(',')))" $TASKS)
SAMPLER=${SAMPLER:-}   # <type>:<steps>, e.g. DDIM:10 (default: the trained sampler, DDPM 100)
SAMPLER_JSON=; [ -n "$SAMPLER" ] && SAMPLER_JSON="\"noise_scheduler_type\": \"${SAMPLER%%:*}\", \"num_inference_steps\": ${SAMPLER##*:}, "
mkdir -p $R/logs $R/eval/final
log() { echo "[eval-mt $(date -u -d '+9 hours' +%m-%d\ %H:%M)] $*" >> $R/logs/eval_mt.log; }
n_ep() { python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(m['num_episodes'], sum(e['success'] for e in m['episodes']))" $1/metrics.json 2>/dev/null || echo "0 0"; }
bash /workspace/codes/_setup/gpu_watch.sh ensure >/dev/null 2>&1

want() { cat $1/want 2>/dev/null || echo $WANT; }   # per-cell episode target: <cell>/want (written to hand the rest of a cell to another GPU)

cell() {  # task hand
  local t=$1 h=$2 o=$R/eval/final/$1/$2 pk lg pid
  if [ -n "${BIMANUAL:-}" ]; then   # the pickle the bimanual LeRobot cell was converted from (a task can have several)
    pk=$(python3 -c "import json,sys; print(json.loads(open(sys.argv[1]).readlines()[-1])['pickle'])" \
         ${XLVLA_TELEOP_ROOT:-/workspace/datasets/dexverse/teleop_lerobot}/$t/${h}_bimanual/lerobot_conversions.jsonl)
  else pk=$(ls $PKL_ROOT/*/$t/${t}_*_floating_${h}_right.pkl | head -1); fi
  mkdir -p $o; lg=$o/eval.log
  pid=$(pgrep -f "run_eval.py .*--output_dir $o/eval " | head -1)   # adopt a running evaluation (manager restarted, e.g. fewer EPS)
  if [ -n "$pid" ]; then log "$t $h: adopted running evaluation (pid $pid)"
  else
    if [ -n "${BIMANUAL:-}" ]; then
      echo "{\"mode\": \"$MODE\", \"bimanual\": true, \"hand\": \"$h\", \"task\": \"$t\", \"policy_ckpt\": \"$POLICY\", \"run\": \"${PI05_RUN:?}\", \"repo_id\": \"${PI05_REPO:?}\", \"action_dim\": ${PI05_ADIM:?}, \"max_token_len\": ${PI05_MAXTOK:?}, \"latent_ckpt\": \"$LATENT\", \"latent_ckpt_left\": \"${LATENT_LEFT:-}\", \"wrist\": \"$WRIST_DIR/wrist_calib_$t.json\", \"replan\": ${REPLAN:-10}, \"log_dir\": null}" > $o/server.json
    elif [[ $MODE == pi05_* ]]; then
      echo "{\"mode\": \"$MODE\", \"hand\": \"$h\", \"task\": \"$t\", \"policy_ckpt\": \"$POLICY\", \"run\": \"${PI05_RUN:?}\", \"repo_id\": \"${PI05_REPO:?}\", \"action_dim\": ${PI05_ADIM:?}, \"latent_ckpt\": \"$LATENT\", \"lad_ckpt\": \"${LAD_CKPT:-}\", \"wrist\": \"$WRIST_DIR/wrist_calib_$t.json\", \"replan\": ${REPLAN:-10}, \"log_dir\": null}" > $o/server.json
    else
    echo "{\"mode\": \"latent_dp\", \"hand\": \"$h\", \"policy_ckpt\": \"$POLICY\", \"latent_ckpt\": \"$LATENT\", \"wrist\": \"$WRIST_DIR/wrist_calib_$t.json\", \"task\": \"$t\", \"task_onehot\": $ONEHOT, $SAMPLER_JSON\"log_dir\": null}" > $o/server.json
    fi
    rm -rf $o/eval
    ( source /workspace/codes/_setup/dexverse_env.sh $WT >/dev/null; cd $DEXVERSE_ROOT; unset DEXVERSE_FINGER_DRIVE DEXVERSE_DRIVE_SPECS
      exec env CUDA_VISIBLE_DEVICES=$GPU python $CODE/dexverse_xlvla/run_eval.py --demos $pk --ckpt $o --output_dir $o/eval \
        --episodes $EPS --max_steps $MAX_STEPS --headless ) > $lg 2>&1 & pid=$!
  fi
  while kill -0 $pid 2>/dev/null; do   # Isaac Sim can hang at start-up (silent log) or at exit (metrics written): kill both
    sleep 20
    if [ -e $R/eval/final/STOPPED ]; then   # another cell was stopped from outside: the whole run ends here
      log "$t $h: STOPPED file present, ending this cell"; { pkill -P $pid; kill $pid; sleep 5; pkill -9 -P $pid; kill -9 $pid; } 2>/dev/null
    elif [ "$(n_ep $o/eval | cut -d' ' -f1)" -ge $(want $o) ]; then   # all wanted episodes written (-ge: an adopted run may want more): stop it
      { pkill -P $pid; kill $pid; sleep 5; pkill -9 -P $pid; kill -9 $pid; } 2>/dev/null
    elif [ $(( $(date +%s) - $(stat -c %Y $lg) )) -gt 3600 ]; then
      log "$t $h: stalled, killed"; { pkill -P $pid; kill $pid; sleep 5; pkill -9 -P $pid; kill -9 $pid; } 2>/dev/null
    fi
  done
  wait $pid 2>/dev/null
  set -- $(n_ep $o/eval); log "$t $h: $2/$1 (want $(want $o))"
  if [ "$1" -lt $(want $o) ] && [ -n "${STOP_ON_KILL:-}" ]; then
    # shared GPU (STOP_ON_KILL=1): an evaluation that ended early without us stopping it was stopped from outside (the
    # server owner) or crashed -> do not start anything again; every worker stops before its next cell
    log "$t $h: ended early ($2/$1) -> STOP (not restarted, STOP_ON_KILL)"; touch $R/eval/final/STOPPED
  fi
  [ "$1" -ge $(want $o) ]
}

worker() {
  local t h k
  for t in ${TASKS//,/ }; do for h in ${HANDS//,/ }; do
    [ -n "${ONLY_TASKS:-}" ] && [[ ",$ONLY_TASKS," != *",$t,"* ]] && continue   # subset of cells (TASKS keeps the one-hot order)
    k=$R/eval/final/.claim_${t}_$h
    [ "$(n_ep $R/eval/final/$t/$h/eval | cut -d' ' -f1)" -ge $(want $R/eval/final/$t/$h) ] && continue
    [ -e $R/eval/final/STOPPED ] && { log "worker: STOPPED file present, leaving"; return; }
    mkdir $k 2>/dev/null || continue
    if [ -n "${STOP_ON_KILL:-}" ]; then cell $t $h || log "$t $h: FAILED (no retry)"
    else cell $t $h || cell $t $h || log "$t $h: FAILED"; fi   # two attempts
  done; done
}

[ -n "${KEEP_CLAIMS:-}" ] || rm -rf $R/eval/final/.claim_*   # KEEP_CLAIMS=1: extra workers joining a running manager of the same run
log "start: GPU $GPU, $WORKERS workers, sampler ${SAMPLER:-trained}, max_steps $MAX_STEPS, policy $POLICY"
for i in $(seq $WORKERS); do worker & sleep 90; done   # staggered Isaac Sim start-up
wait
log "ALL DONE"
