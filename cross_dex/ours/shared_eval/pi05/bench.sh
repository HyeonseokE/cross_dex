#!/bin/bash
# N evaluation sessions of ONE pi0.5 cell (same pickle, same episodes), old or shared path (inside a server container).
#   PKL=<pickle> SERVER_JSON=<a cell's server.json> GPU=0 bash bench.sh <out_dir> <dedicated|shared> <N> <episodes>
# dedicated = every session has its own openpi process (the baseline path, via session_server --mode dedicated)
# shared    = one openpi_server.py (model, CK/run/repo/action_dim/max_token_len read from SERVER_JSON), N sessions on it
# Every session writes the same cell into <out>/s<i>/ (server.json copy with log_dir = s<i>/actions, so actions can be
# compared between runs: summarize.py --compare). Sessions start STAGGER s apart and wait at a barrier (first reset), so the
# episodes run concurrently. Session 1 saves its first SAVE_OBS infer inputs (<out>/s1/obs.npz, for check_batch.py).
# env: STAGGER (90), MAX_STEPS (400), VIDEO (0 = --no-video), GATHER_MS (1000), NOISE (process), MAX_BATCH (=N),
#      PRECOMPILE (=MAX_BATCH), SAVE_OBS (8), DEXVERSE_WORKTREE (DexVerse)
# Writes <out>/{s<i>/{server.json,eval/,actions/,timing.jsonl,eval.log}, server.{log,jsonl}, gpu.csv, DONE}.
set -u
OUT=$1 MODE=$2 N=$3 EPS=$4
: "${PKL:?}" "${SERVER_JSON:?}"
GPU=${GPU:-0}; STAGGER=${STAGGER:-90}; MAX_STEPS=${MAX_STEPS:-400}; WT=${DEXVERSE_WORKTREE:-DexVerse}
GATHER_MS=${GATHER_MS:-1000}; NOISE=${NOISE:-process}; MAX_BATCH=${MAX_BATCH:-$N}; PRECOMPILE=${PRECOMPILE:-$MAX_BATCH}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
XLVLA=${XLVLA_CODE:-/workspace/codes/dexverse/DexLatent}
OPENPI_PY=${XLVLA_OPENPI_PYTHON:-/workspace/envs/dexverse/openpi/bin/python}
export XLVLA_COMMON_DP=${XLVLA_COMMON_DP:-/workspace/codes/dexverse/$WT/baselines/lerobot_dp}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
VID=--no-video; [ "${VIDEO:-0}" = 1 ] && VID=--video
bash /workspace/codes/_setup/gpu_watch.sh ensure >/dev/null 2>&1
mkdir -p "$OUT"; rm -rf "$OUT/barrier"
J() { python3 -c "import json,sys; v=json.load(open(sys.argv[1])).get(sys.argv[2]); print('' if v is None else v)" "$SERVER_JSON" "$1"; }

(while true; do
   echo "$(date +%s),$(nvidia-smi -i $GPU --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits),$(free -m | awk '/Mem:/{print $3}'),$(cut -d' ' -f1 /proc/loadavg)"
   sleep 5
 done) > "$OUT/gpu.csv" &
SAMPLER=$!

EXTRA=()
if [ "$MODE" = shared ]; then
  PORT=$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
  export SHARED_EVAL_AUTHKEY=$(python3 -c 'import secrets; print(secrets.token_hex(16))')
  MAXTOK=$(J max_token_len)
  ( cd $XLVLA && exec env -u LD_PRELOAD -u CONDA_PREFIX PYTHONPATH=$XLVLA XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=$GPU \
      $OPENPI_PY $HERE/openpi_server.py --ckpt "$(J policy_ckpt)" --run "$(J run)" --repo_id "$(J repo_id)" \
      --action_dim "$(J action_dim)" ${MAXTOK:+--max_token_len $MAXTOK} --noise $NOISE --precompile $PRECOMPILE \
      --gather_ms $GATHER_MS --max_batch $MAX_BATCH --sessions $N --port $PORT --log $OUT/server.jsonl ) \
    < /dev/null > "$OUT/server.log" 2>&1 &
  SERVER=$!
  EXTRA=(--shared_port $PORT)
fi

PIDS=()
for i in $(seq 1 "$N"); do
  S=$OUT/s$i; rm -rf $S; mkdir -p $S
  python3 -c "import json,sys; c=json.load(open(sys.argv[1])); c['log_dir']=sys.argv[2]; json.dump(c, open(sys.argv[3],'w'))" \
    "$SERVER_JSON" $S/actions $S/server.json
  OBS=(); [ "$i" = 1 ] && OBS=(--save_obs ${SAVE_OBS:-8})
  ( source /workspace/codes/_setup/dexverse_env.sh $WT >/dev/null; cd $DEXVERSE_ROOT; unset DEXVERSE_FINGER_DRIVE DEXVERSE_DRIVE_SPECS
    exec env CUDA_VISIBLE_DEVICES=$GPU python $HERE/run_eval_shared.py --mode $MODE "${EXTRA[@]}" "${OBS[@]}" \
      --barrier $OUT/barrier:$N --timing $S/timing.jsonl --demos $PKL --ckpt $S --output_dir $S/eval \
      --episodes $EPS --max_steps $MAX_STEPS --headless $VID ) < /dev/null > $S/eval.log 2>&1 &
  PIDS+=($!)
  [ "$i" -lt "$N" ] && sleep "$STAGGER"
done
wait "${PIDS[@]}"
[ "$MODE" = shared ] && { wait $SERVER || true; }
kill $SAMPLER
date +%s > "$OUT/DONE"
