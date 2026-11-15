#!/bin/bash
# Snapshot-selection rollouts of a single-hand latent DP (st_hand.sh recipe = run_sets.sh: pickle episodes 0-9, max 3000 steps,
# trained sampler, no video), on any server that holds a copy of the snapshot. One call = one snapshot; GPU slots are taken with
# flock on /workspace/.tmp/sel_slot.g<gpu>.<n> (SLOTS = one GPU id per slot, 7.1 N per GPU), so many calls can be queued at once.
#   SLOTS="0 0 0 0 1 1 1 1" LATENT=<latent.pt> WRIST_DIR=<dir> bash dexverse_xlvla/sel_eval.sh <out_dir> <pretrained_model> <Task-v0> <hand>
set -u
: "${LATENT:?}" "${WRIST_DIR:?}"
O=$1; M=$2; T=$3; H=$4; SLOTS=${SLOTS:-0}; WT=${DEXVERSE_WORKTREE:-DexVerse}
PKL_ROOT=${XLVLA_PKL_ROOT:-/workspace/datasets/dexverse/teleop_pkl_bs1}
export XLVLA_COMMON_DP=${XLVLA_COMMON_DP:-/workspace/codes/dexverse/$WT/baselines/lerobot_dp} PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
CODE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
n_ep() { python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['num_episodes'])" $1/metrics.json 2>/dev/null || echo 0; }
[ "$(n_ep $O/eval)" = 10 ] && exit 0
pk=$(ls $PKL_ROOT/*/Dexverse-$T/Dexverse-${T}_*_floating_${H}_right.pkl | head -1)
i=0
while true; do   # take a free slot
  declare -A k=()   # lock per (gpu, n-th slot of that gpu) so callers with different SLOTS lists share slots correctly
  for g in $SLOTS; do k[$g]=$(( ${k[$g]:-0} + 1 )); exec 9>/workspace/.tmp/sel_slot.g$g.${k[$g]}; if flock -n 9; then break 2; fi; exec 9>&-; done
  sleep 60
done
mkdir -p $O; rm -rf $O/eval
echo "{\"mode\": \"latent_dp\", \"hand\": \"$H\", \"policy_ckpt\": \"$M\", \"latent_ckpt\": \"$LATENT\", \"wrist\": \"$WRIST_DIR/wrist_calib_Dexverse-$T.json\", \"task\": \"Dexverse-$T\", \"log_dir\": null}" > $O/server.json
( source /workspace/codes/_setup/dexverse_env.sh $WT >/dev/null; cd $DEXVERSE_ROOT; unset DEXVERSE_FINGER_DRIVE DEXVERSE_DRIVE_SPECS
  timeout 4h env CUDA_VISIBLE_DEVICES=$g python $CODE/dexverse_xlvla/run_eval.py --demos $pk --ckpt $O --output_dir $O/eval \
    --episodes $(seq -s, 0 9) --max_steps 3000 --no-video --headless ) > $O/eval.log 2>&1
[ "$(n_ep $O/eval)" = 10 ]
