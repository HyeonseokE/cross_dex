#!/bin/bash
# Shared-part vs residual attribution (RSS2027 problem 2): replay a hand's own demos open loop in the simulator from the
# demo initial states, either unchanged (raw: q*) or with the fingers replaced by their shared part s_h = D_h(E_h(q*))
# (latent self replay; wrist unchanged). The success gap raw - shared is the task effect of the residual r_h = q* - s_h.
# Same builder / evaluator as dexverse_xlvla/replay_monitor.sh (dexverse_xlvla.replay + run_eval.py), one cell per call.
#   bash self_replay.sh <task> <hand> <raw|shared> <episodes, e.g. 0-9> <out dir> [latent ckpt] [gpu]
set -euo pipefail
T=$1 HAND=$2 MODE=$3 EPS=$4 O=$5 CKPT=${6:-/workspace/checkpoints/dexverse/xlvla/hf6/latent/checkpoint_epoch_25000.pt} G=${7:-${CUDA_VISIBLE_DEVICES:-0}}
PY=/workspace/envs/dexverse/dexlatent/bin/python; CODE=/workspace/codes/dexverse/DexLatent; WT=${DEXVERSE_WORKTREE:-DexVerse}
export XLVLA_COMMON_DP=${XLVLA_COMMON_DP:-/workspace/codes/dexverse/$WT/baselines/lerobot_dp}
export XLVLA_PKL_ROOT=${XLVLA_PKL_ROOT:-/workspace/datasets/dexverse/teleop_pkl_bs1}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
bash /workspace/codes/_setup/gpu_watch.sh ensure >/dev/null 2>&1 || true
mkdir -p $O; cd $CODE
if [ $MODE = raw ]; then
  $PY -m dexverse_xlvla.replay --task $T --src $HAND --dst $HAND --mode raw --out $O --episodes $EPS > $O/build.log 2>&1
else
  $PY -m dexverse_xlvla.replay --task $T --src $HAND --dst $HAND --mode latent --ckpt $CKPT --out $O --episodes $EPS > $O/build.log 2>&1
fi
DEMOS=$(ls $XLVLA_PKL_ROOT/*/$T/${T}_*_floating_${HAND}_right.pkl)
MS=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['max_steps'])" $O/replay.json)
E=$(seq -s, ${EPS%-*} ${EPS#*-})
( source /workspace/codes/_setup/dexverse_env.sh $WT >/dev/null; cd $DEXVERSE_ROOT
  unset DEXVERSE_FINGER_DRIVE DEXVERSE_DRIVE_SPECS
  CUDA_VISIBLE_DEVICES=$G python $CODE/dexverse_xlvla/run_eval.py --demos $DEMOS --ckpt $O/policy --output_dir $O/eval \
    --episodes $E --max_steps $MS --no-video --headless ) > $O/eval.log 2>&1
$PY -c "import json,sys; m=json.load(open(sys.argv[1])); print(sum(e['success'] for e in m['episodes']), m['num_episodes'])" $O/eval/metrics.json | tee $O/result.txt
