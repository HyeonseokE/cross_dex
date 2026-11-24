#!/bin/bash
# Action ceilings for every arm x hand (replay.sh), P cells at a time on one GPU, starts 90 s apart (Isaac start-up).
# Finished cells (result.txt with "exit") are skipped, so a relaunch resumes. Summary -> <out>/summary.tsv
#   bash replay_all.sh <out> <gpu> [P=4] [episodes=0-9] [arms="raw joint xlvla ea eh e2c"]
O=$1 G=$2 P=${3:-4} EPS=${4:-0-9} ARMS=${5:-raw joint xlvla ea eh e2c}
HERE=$(cd $(dirname $0) && pwd); mkdir -p $O
cells=(); for a in $ARMS; do for h in shadow inspire sharpa wuji allegro leap; do cells+=("$h $a"); done; done
i=0
for c in "${cells[@]}"; do
  set -- $c
  grep -qs exit $O/${1}_$2/result.txt && continue
  while [ $(jobs -rp | wc -l) -ge $P ]; do sleep 10; done
  echo "[replay_all $(date -u -d '+9 hours' '+%m-%d %H:%M') KST] start $1 $2"
  bash $HERE/replay.sh $1 $2 $EPS $O/${1}_$2 $G > $O/${1}_$2.log 2>&1 &
  i=$((i + 1)); sleep 90
done
wait
for c in "${cells[@]}"; do set -- $c; printf "%s\t%s\t%s\n" $2 $1 "$(head -1 $O/${1}_$2/result.txt 2>/dev/null)"; done > $O/summary.tsv
echo "[replay_all] ALL DONE"
