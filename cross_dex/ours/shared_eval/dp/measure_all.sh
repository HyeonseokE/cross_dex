#!/bin/bash
# The shared-model measurement set (2026-11-16, amp7): today's path, seeded dedicated runs, delay injection, shared N=2/4/5.
#   PKL=... CKPT=... bash measure_all.sh <root>     (each run in <root>/<name>/, skipped if <name>/DONE exists)
set -u
ROOT=$1
B=$(cd "$(dirname "$0")" && pwd)/bench.sh
EPS=${EPS:-0,2}
# (greedy shared_n5 and a cuda dedicated reference were dropped on 2026-11-16 to keep GPU time down)
run() {  # name mode N [args]
  local name=$1; shift
  [ -f "$ROOT/$name/DONE" ] && return
  echo "[measure] $(date -u +%FT%TZ) start $name"
  bash "$B" "$ROOT/$name" "$@"
  echo "[measure] $(date -u +%FT%TZ) done $name"
}
run own_n1        own       1 "$EPS"
run ded_n1        dedicated 1 "$EPS" --seed 0
run ded_n1_rep    dedicated 1 "$EPS" --seed 0
run ded_n1_delay  dedicated 1 "$EPS" --seed 0 --inject_delay_ms 500
run shared_n2     shared    2 "$EPS" --seed 0
run shared_n4     shared    4 "$EPS" --seed 0
run own_n4        own       4 "$EPS"
# shared_server noise generators: cpu up to ded_n1_delay, cuda from shared_n2 on (default changed in between).
# Lockstep batching from here (a chunk request waits up to 1 s for every active session; N=1 unaffected).
export GATHER_MS=1000
run shared_n4_sync shared 4 "$EPS" --seed 0
run shared_n5_sync shared 5 "$EPS" --seed 0
run shared_n1     shared    1 "$EPS" --seed 0
echo "[measure] $(date -u +%FT%TZ) ALL DONE"
