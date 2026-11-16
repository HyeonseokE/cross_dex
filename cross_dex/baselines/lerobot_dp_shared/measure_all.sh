#!/bin/bash
# The shared-model measurement set (2026-11-16, amp7): today's path, seeded dedicated runs, delay injection, shared N=2/4/5.
#   PKL=... CKPT=... bash measure_all.sh <root>     (each run in <root>/<name>/, skipped if <name>/DONE exists)
set -u
ROOT=$1
B=$(cd "$(dirname "$0")" && pwd)/bench.sh
EPS=${EPS:-0,2}
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
run shared_n5     shared    5 "$EPS" --seed 0
echo "[measure] $(date -u +%FT%TZ) ALL DONE"
