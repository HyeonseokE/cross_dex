#!/bin/bash
# N evaluation sessions of one DP checkpoint, one of three policy-server modes (inside a server container).
#   PKL=<demos.pkl> CKPT=<pretrained_model> bash bench.sh <out_dir> <own|dedicated|shared> <N> <episodes> [run_eval_dp args]
# own       = each session starts DexVerse's policy_server.py (today: one model per session)
# dedicated = each session starts shared_server.py for itself only (seeded noise, no sharing)
# shared    = one shared_server.py, N sessions connect to it
# Sessions start STAGGER s apart (Isaac Sim start-up VRAM peak) and wait at a barrier, so the episodes run concurrently.
# Writes <out_dir>/{s<i>.log, s<i>/timing/ep*.npz, server.jsonl, gpu.csv, DONE}.
set -u
OUT=$1 MODE=$2 N=$3 EPS=$4
shift 4
: "${PKL:?}" "${CKPT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
STAGGER=${STAGGER:-90}
MAX_STEPS=${MAX_STEPS:-400}
LEROBOT_PYTHON=${LEROBOT_PYTHON:-/workspace/envs/dexverse/lerobot/bin/python}
bash /workspace/codes/_setup/gpu_watch.sh ensure
source /workspace/codes/_setup/dexverse_env.sh DexVerse
mkdir -p "$OUT"
rm -rf "$OUT/barrier"

(while true; do
   echo "$(date +%s),$(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits | head -1),$(free -m | awk '/Mem:/{print $3}'),$(cut -d' ' -f1 /proc/loadavg)"
   sleep 5
 done) > "$OUT/gpu.csv" &
SAMPLER=$!

EXTRA=()
if [ "$MODE" = shared ]; then
  PORT=$(python -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
  export DP_SHARED_AUTHKEY=$(python -c 'import secrets; print(secrets.token_hex(16))')
  env -u LD_PRELOAD -u PYTHONPATH -u CONDA_PREFIX "$LEROBOT_PYTHON" "$HERE/shared_server.py" --ckpt "$CKPT" \
    --port "$PORT" --authkey "$DP_SHARED_AUTHKEY" --sessions "$N" --log "$OUT/server.jsonl" \
    --dp_dir "$DEXVERSE_ROOT/baselines/lerobot_dp" > "$OUT/server.log" 2>&1 &
  SERVER=$!
  EXTRA=(--address "127.0.0.1:$PORT")
fi

PIDS=()
for i in $(seq 1 "$N"); do
  LOGARGS=()
  [ "$MODE" = dedicated ] && LOGARGS=(--server_log "$OUT/server_s$i.jsonl")
  python "$HERE/run_eval_dp.py" --mode "$MODE" "${EXTRA[@]}" "${LOGARGS[@]}" --barrier "$OUT/barrier:$N" \
    --timing_dir "$OUT/s$i/timing" --demos "$PKL" --ckpt "$CKPT" --output_dir "$OUT/s$i" --headless \
    --episodes "$EPS" --max_steps "$MAX_STEPS" --no-video "$@" > "$OUT/s$i.log" 2>&1 &
  PIDS+=($!)
  [ "$i" -lt "$N" ] && sleep "$STAGGER"
done
wait "${PIDS[@]}"
[ "$MODE" = shared ] && wait "$SERVER"
kill "$SAMPLER"
date +%s > "$OUT/DONE"
