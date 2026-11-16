#!/bin/bash
# Final evaluation of a pi0.5 multi-task run with ONE openpi model per GPU shared by WORKERS Isaac sessions (ours).
# Same cells, claims, per-cell outputs and resume rules as the baseline manager dexverse_xlvla/eval_mt.sh (not changed,
# not called): every (task, hand) cell of TASKS x HANDS, 50 rollouts on pickle episodes 0-49 through the common evaluator,
#   <R>/eval/final/<task>/<hand>/{server.json, eval.log, eval/{metrics.json, videos/}}   + timing.jsonl (ours)
# claims <R>/eval/final/.claim_<task>_<hand> (mkdir), finished cells skipped, <cell>/want, STOP_ON_KILL / STOPPED, KEEP_CLAIMS.
# Difference: the manager starts openpi_server.py on its GPU (log <R>/logs/shared_gpu<GPU>.{log,jsonl}) and every cell runs
# run_eval_shared.py --mode shared against it (no openpi process per cell). The server is restarted if it dies.
#   R=<run dir> CK=<openpi step dir> LATENT=<latent.pt> WRIST_DIR=<dir> MODE=pi05_latent|pi05_joint|pi05_lad \
#     PI05_RUN=<TrainConfig> PI05_REPO=<norm-stats id> PI05_ADIM=<41|32|...> GPU=0 WORKERS=4 bash eval_shared.sh
# env as eval_mt.sh (pi0.5 part): LAD_CKPT, BIMANUAL=1 (+ PI05_MAXTOK, LATENT_LEFT, LAD_CKPT_LEFT), REPLAN (10), EPS, MAX_STEPS,
#   TASKS, HANDS, ONLY_TASKS, DEXVERSE_WORKTREE, STOP_ON_KILL, KEEP_CLAIMS. Ours:
#   SHARED (1; 0 = each cell runs the BASELINE run_eval.py with its own openpi process = old path, for comparisons)
#   CELLS (task:hand,... instead of TASKS x HANDS), GATHER_MS (5 = greedy, measured best 2026-11-16; 1000 = lockstep), MAX_BATCH (=WORKERS), NOISE (process | episode, NOISE_SEED 0), PRECOMPILE (=MAX_BATCH)
#   LOG_ACTIONS=1 (server.json log_dir = <cell>/actions: policy_out / action per episode, for old-vs-shared comparisons)
#   STAGGER (90 s between session starts), DRY_RUN=1 (print the server command, each cell's server.json and command; no launch)
#   UNIQUE_LAYOUTS (1): per cell, EPS with repeated initial layouts replaced by the next distinct ones (unique_eps.py, cached
#   in <cell>/episodes; unchanged for pickles without repeats); 0 = EPS as given
set -u
: "${R:?}" "${CK:?}" "${LATENT:?}" "${WRIST_DIR:?}" "${MODE:?}"
[[ $MODE == pi05_* ]] || { echo "eval_shared.sh: pi0.5 modes only (MODE=$MODE)"; exit 2; }
GPU=${GPU:-0}; WORKERS=${WORKERS:-4}; MAX_STEPS=${MAX_STEPS:-3000}; EPS=${EPS:-$(seq -s, 0 49)}
TASKS=${TASKS:-Dexverse-GraspBleach-v0,Dexverse-GraspPan-v0,Dexverse-GraspKettle-v0,Dexverse-FunctionalHammerStrike-v0}
HANDS=${HANDS:-shadow,inspire,sharpa,wuji}; WT=${DEXVERSE_WORKTREE:-DexVerse}
PKL_ROOT=${XLVLA_PKL_ROOT:-/workspace/datasets/dexverse/teleop_pkl_bs1}
SHARED=${SHARED:-1}; GATHER_MS=${GATHER_MS:-5}; MAX_BATCH=${MAX_BATCH:-$WORKERS}; NOISE=${NOISE:-process}
PRECOMPILE=${PRECOMPILE:-$MAX_BATCH}; STAGGER=${STAGGER:-90}; DRY=${DRY_RUN:-}; UNIQUE=${UNIQUE_LAYOUTS:-1}
export XLVLA_COMMON_DP=${XLVLA_COMMON_DP:-/workspace/codes/dexverse/$WT/baselines/lerobot_dp}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
XLVLA=${XLVLA_CODE:-/workspace/codes/dexverse/DexLatent}            # DexLatent root (dexverse_xlvla link inside)
OPENPI_PY=${XLVLA_OPENPI_PYTHON:-/workspace/envs/dexverse/openpi/bin/python}
POLICY=$(readlink -f $CK)
WANT=$(echo $EPS | tr , '\n' | wc -l)
[ -n "$DRY" ] || mkdir -p $R/logs $R/eval/final
log() { [ -n "$DRY" ] && { echo "[log] $*"; return; }; echo "[eval-shared $(date -u -d '+9 hours' +%m-%d\ %H:%M)] $*" >> $R/logs/eval_shared.log; }
n_ep() { python3 -c "import json,sys; m=json.load(open(sys.argv[1])); print(m['num_episodes'], sum(e['success'] for e in m['episodes']))" $1/metrics.json 2>/dev/null || echo "0 0"; }
want() { cat $1/want 2>/dev/null || echo $WANT; }
[ -n "$DRY" ] || bash /workspace/codes/_setup/gpu_watch.sh ensure >/dev/null 2>&1

# ---------------------------------------------------------------- the GPU's shared openpi server
PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
export SHARED_EVAL_AUTHKEY=${SHARED_EVAL_AUTHKEY:-$(python3 -c 'import secrets; print(secrets.token_hex(16))')}
SERVER_CMD=(env -u LD_PRELOAD -u CONDA_PREFIX PYTHONPATH=$XLVLA XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$GPU
  $OPENPI_PY $HERE/openpi_server.py --ckpt $POLICY --run ${PI05_RUN:?} --repo_id ${PI05_REPO:?} --action_dim ${PI05_ADIM:?}
  ${PI05_MAXTOK:+--max_token_len $PI05_MAXTOK} --noise $NOISE --seed ${NOISE_SEED:-0} --precompile $PRECOMPILE
  --gather_ms $GATHER_MS --max_batch $MAX_BATCH --port $PORT --log $R/logs/shared_gpu$GPU.jsonl --exit_with_pid $$)
start_server() {
  ( cd $XLVLA && exec "${SERVER_CMD[@]}" ) < /dev/null >> $R/logs/shared_gpu$GPU.log 2>&1 &
  echo $! > $R/logs/shared_gpu$GPU.pid
  log "openpi server pid $! on GPU $GPU port $PORT (gather $GATHER_MS ms, max batch $MAX_BATCH, noise $NOISE)"
}

server_json() {  # task hand -> the baseline's server.json (same keys as eval_mt.sh), optional log_dir
  local t=$1 h=$2 o=$R/eval/final/$1/$2 ld=null
  [ -n "${LOG_ACTIONS:-}" ] && ld="\"$o/actions\""
  if [ -n "${BIMANUAL:-}" ]; then
    echo "{\"mode\": \"$MODE\", \"bimanual\": true, \"hand\": \"$h\", \"task\": \"$t\", \"policy_ckpt\": \"$POLICY\", \"run\": \"${PI05_RUN:?}\", \"repo_id\": \"${PI05_REPO:?}\", \"action_dim\": ${PI05_ADIM:?}, \"max_token_len\": ${PI05_MAXTOK:?}, \"latent_ckpt\": \"$LATENT\", \"latent_ckpt_left\": \"${LATENT_LEFT:-}\", \"lad_ckpt\": \"${LAD_CKPT:-}\", \"lad_ckpt_left\": \"${LAD_CKPT_LEFT:-}\", \"wrist\": \"$WRIST_DIR/wrist_calib_$t.json\", \"replan\": ${REPLAN:-10}, \"log_dir\": $ld}"
  else
    echo "{\"mode\": \"$MODE\", \"hand\": \"$h\", \"task\": \"$t\", \"policy_ckpt\": \"$POLICY\", \"run\": \"${PI05_RUN:?}\", \"repo_id\": \"${PI05_REPO:?}\", \"action_dim\": ${PI05_ADIM:?}, \"latent_ckpt\": \"$LATENT\", \"lad_ckpt\": \"${LAD_CKPT:-}\", \"wrist\": \"$WRIST_DIR/wrist_calib_$t.json\", \"replan\": ${REPLAN:-10}, \"log_dir\": $ld}"
  fi
}

pickle_of() {  # task hand -> pickle (as eval_mt.sh)
  if [ -n "${BIMANUAL:-}" ]; then
    python3 -c "import json,sys; print(json.loads(open(sys.argv[1]).readlines()[-1])['pickle'])" \
      ${XLVLA_TELEOP_ROOT:-/workspace/datasets/dexverse/teleop_lerobot}/$1/${2}_bimanual/lerobot_conversions.jsonl 2>/dev/null || echo MISSING_PICKLE
  else ls $PKL_ROOT/*/$1/${1}_*_floating_${2}_right.pkl 2>/dev/null | head -1 | grep . || echo MISSING_PICKLE; fi
}

eps_of() {  # task hand pickle -> this cell's episodes (EPS, or with distinct initial layouts; cached in <cell>/episodes)
  local o=$R/eval/final/$1/$2 e
  [ "$UNIQUE" = 1 ] || { echo $EPS; return; }
  [ -s $o/episodes ] && [ "$(head -1 $o/episodes)" = "$3 $EPS" ] && { tail -1 $o/episodes; return; }
  e=$( source /workspace/codes/_setup/dexverse_env.sh $WT >/dev/null 2>&1; python $HERE/unique_eps.py $3 $EPS ) || { log "$1 $2: unique_eps failed, EPS as given" >&2; echo $EPS; return; }
  [ "$e" = "$EPS" ] || log "$1 $2: repeated initial layouts in $(basename $3) -> episodes $e" >&2
  [ -n "$DRY" ] || { mkdir -p $o; printf '%s\n%s\n' "$3 $EPS" "$e" > $o/episodes; }
  echo $e
}

eval_cmd() {  # task hand pickle episodes -> the evaluator command (inside the main env, cwd DEXVERSE_ROOT)
  local o=$R/eval/final/$1/$2
  if [ "$SHARED" = 1 ]; then
    echo env CUDA_VISIBLE_DEVICES=$GPU python $HERE/run_eval_shared.py --mode shared --shared_port $PORT --timing $o/timing.jsonl \
      --demos $3 --ckpt $o --output_dir $o/eval --episodes $4 --max_steps $MAX_STEPS --headless
  else   # old path: the baseline wrapper and server, unchanged
    echo env CUDA_VISIBLE_DEVICES=$GPU python $XLVLA/dexverse_xlvla/run_eval.py \
      --demos $3 --ckpt $o --output_dir $o/eval --episodes $4 --max_steps $MAX_STEPS --headless
  fi
}

cell() {  # task hand
  local t=$1 h=$2 o=$R/eval/final/$1/$2 pk lg pid ep
  pk=$(pickle_of $t $h); mkdir -p $o; lg=$o/eval.log; ep=$(eps_of $t $h $pk)
  pid=$(pgrep -f "run_eval(_shared)?.py .*--output_dir $o/eval " | head -1)
  if [ -n "$pid" ] && [ "$SHARED" = 1 ]; then   # its openpi server (previous manager, new port) is gone: start the cell again
    log "$t $h: stale evaluation (pid $pid) of a previous manager, killed"; { pkill -P $pid; kill $pid; sleep 5; pkill -9 -P $pid; kill -9 $pid; } 2>/dev/null; pid=
  fi
  if [ -n "$pid" ]; then log "$t $h: adopted running evaluation (pid $pid)"
  else
    server_json $t $h > $o/server.json
    rm -rf $o/eval $o/actions $o/timing.jsonl
    ( source /workspace/codes/_setup/dexverse_env.sh $WT >/dev/null; cd $DEXVERSE_ROOT; unset DEXVERSE_FINGER_DRIVE DEXVERSE_DRIVE_SPECS
      exec $(eval_cmd $t $h $pk $ep) ) < /dev/null > $lg 2>&1 & pid=$!
  fi
  while kill -0 $pid 2>/dev/null; do   # as eval_mt.sh: STOPPED, wanted episodes reached, silent log for 1 h
    sleep 20
    if [ -e $R/eval/final/STOPPED ]; then
      log "$t $h: STOPPED file present, ending this cell"; { pkill -P $pid; kill $pid; sleep 5; pkill -9 -P $pid; kill -9 $pid; } 2>/dev/null
    elif [ "$(n_ep $o/eval | cut -d' ' -f1)" -ge $(want $o) ]; then
      { pkill -P $pid; kill $pid; sleep 5; pkill -9 -P $pid; kill -9 $pid; } 2>/dev/null
    elif [ $(( $(date +%s) - $(stat -c %Y $lg) )) -gt 3600 ]; then
      log "$t $h: stalled, killed"; { pkill -P $pid; kill $pid; sleep 5; pkill -9 -P $pid; kill -9 $pid; } 2>/dev/null
    fi
  done
  wait $pid 2>/dev/null
  set -- $(n_ep $o/eval); log "$t $h: $2/$1 (want $(want $o))"
  if [ "$1" -lt $(want $o) ] && [ -n "${STOP_ON_KILL:-}" ]; then
    log "$t $h: ended early ($2/$1) -> STOP (not restarted, STOP_ON_KILL)"; touch $R/eval/final/STOPPED
  fi
  [ "$1" -ge $(want $o) ]
}

cells() {
  local t h c
  if [ -n "${CELLS:-}" ]; then                     # explicit cells "task:hand,task:hand" (e.g. diagonal hold-out cells)
    for c in ${CELLS//,/ }; do echo "${c%%:*} ${c#*:}"; done; return
  fi
  for t in ${TASKS//,/ }; do for h in ${HANDS//,/ }; do
    [ -n "${ONLY_TASKS:-}" ] && [[ ",$ONLY_TASKS," != *",$t,"* ]] && continue
    echo "$t $h"
  done; done
}

worker() {
  local t h k
  while read -r t h; do
    k=$R/eval/final/.claim_${t}_$h
    [ "$(n_ep $R/eval/final/$t/$h/eval | cut -d' ' -f1)" -ge $(want $R/eval/final/$t/$h) ] && continue
    [ -e $R/eval/final/STOPPED ] && { log "worker: STOPPED file present, leaving"; return; }
    mkdir $k 2>/dev/null || continue
    if [ -n "${STOP_ON_KILL:-}" ]; then cell $t $h || log "$t $h: FAILED (no retry)"
    else cell $t $h || cell $t $h || log "$t $h: FAILED"; fi
  done < <(cells)
}

if [ -n "$DRY" ]; then
  [ "$SHARED" = 1 ] && echo "[server] cd $XLVLA && ${SERVER_CMD[*]}"
  while read -r t h; do
    pk=$(pickle_of $t $h)
    echo "[cell $t $h] server.json: $(server_json $t $h)"
    echo "[cell $t $h] cd \$DEXVERSE_ROOT && $(eval_cmd $t $h $pk $(eps_of $t $h $pk))"
  done < <(cells)
  exit 0
fi

[ -n "${KEEP_CLAIMS:-}" ] || rm -rf $R/eval/final/.claim_*
log "start: GPU $GPU, $WORKERS sessions, shared $SHARED, max_steps $MAX_STEPS, policy $POLICY"
if [ "$SHARED" = 1 ]; then
  start_server
  ( while sleep 30; do   # the server died (crash / GPU fault): start it again; sessions fail and their cells are retried
      kill -0 $(cat $R/logs/shared_gpu$GPU.pid) 2>/dev/null || { log "openpi server gone, restarting"; start_server; }
    done ) & KEEPER=$!
fi
PIDS=()
for i in $(seq $WORKERS); do worker & PIDS+=($!); [ $i -lt $WORKERS ] && sleep $STAGGER; done   # staggered Isaac Sim start-up
wait "${PIDS[@]}"
if [ "$SHARED" = 1 ]; then kill $KEEPER 2>/dev/null; kill $(cat $R/logs/shared_gpu$GPU.pid) 2>/dev/null; fi
log "ALL DONE"
