#!/bin/bash
# Shared-part vs residual attribution (RSS2027 problem 2) through the conversion replay path: the same command that
# reproduces the teleop demos (scripts/data_tools/pickle_to_lerobot.py -> record_demos.py --replay_demos, cameras on;
# 2026-11-19 leap GraspPan 50/50, while the policy evaluator eval_lerobot replays only 1/10). Open loop from the recorded
# initial states; success = the task's own success term held as in recording.
#   raw     recorded actions q*            shared  fingers replaced by s_h = D_h(E_h(q*)) (wrist unchanged)
#   fixed   fingers D_h(E_h(q*)) + f_h(E_h(q*)) ([1-b] decoder correction, XLVLA_FIX=<decoder_fix.py checkpoint>)
#   bash pkl_replay.sh <task> <hand> <raw|shared> <episodes, e.g. 0-9> <out dir> [latent ckpt]
# The rendered LeRobot output goes to a temporary folder under /workspace/.tmp and is deleted at the end.
set -euo pipefail
T=$1 HAND=$2 MODE=$3 EPS=$4 O=$5 CKPT=${6:-/workspace/checkpoints/dexverse/xlvla/hf6/latent/checkpoint_epoch_25000.pt}
PY=/workspace/envs/dexverse/dexlatent/bin/python; MAIN=/workspace/envs/dexverse/main/bin/python
CODE=/workspace/codes/dexverse/DexLatent; WT=${DEXVERSE_WORKTREE:-DexVerse}; HERE=$(cd $(dirname $0) && pwd)
export XLVLA_PKL_ROOT=${XLVLA_PKL_ROOT:-/workspace/datasets/dexverse/teleop_pkl_bs1}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
bash /workspace/codes/_setup/gpu_watch.sh ensure >/dev/null 2>&1 || true
mkdir -p $O; O=$(cd $O && pwd)
SRC=$(ls -S $XLVLA_PKL_ROOT/*/$T/${T}_*_floating_${HAND}_right.pkl | head -1)   # top-up pickles next to the 50-ep one: largest
TMP=/workspace/.tmp/pkl_replay_$(basename $(dirname $O))_$(basename $O)_$$; rm -rf $TMP; mkdir -p $TMP/in
trap 'rm -rf $TMP' EXIT
if [ $MODE = shared ]; then
  ( cd $CODE && $PY -m dexverse_xlvla.replay --task $T --src $HAND --dst $HAND --mode latent --ckpt $CKPT --out $O/build \
      --episodes $EPS ) > $O/build.log 2>&1
  $MAIN $HERE/make_replay_pkl.py $SRC shared $EPS $TMP/in/$(basename $SRC) $O/build/actions.npz >> $O/build.log 2>&1
elif [ $MODE = fixed ]; then   # [1-b] corrected decoder D_h + f_h (decoder_fix.py, checkpoint in XLVLA_FIX)
  ( cd $CODE && PYTHONPATH=/workspace/codes/cross_dex $PY -m cross_dex.ours.diag.decoder_fix build --fix $XLVLA_FIX --task $T --hand $HAND --episodes $EPS \
      --out $O/build ) > $O/build.log 2>&1
  $MAIN $HERE/make_replay_pkl.py $SRC fixed $EPS $TMP/in/$(basename $SRC) $O/build/actions.npz >> $O/build.log 2>&1
else
  $MAIN $HERE/make_replay_pkl.py $SRC raw $EPS $TMP/in/$(basename $SRC) > $O/build.log 2>&1
fi
( source /workspace/codes/_setup/dexverse_env.sh $WT >/dev/null; cd $DEXVERSE_ROOT
  unset DEXVERSE_FINGER_DRIVE DEXVERSE_DRIVE_SPECS; export PYTHONUNBUFFERED=1
  python scripts/data_tools/pickle_to_lerobot.py $TMP/in/$(basename $SRC) --datasets_dir $TMP/out \
    --lerobot_python /workspace/envs/dexverse/lerobot/bin/python ) > $O/convert.log 2>&1 || true
cp $TMP/out/trajectories_replayed/*_replay.log $O/replay.log 2>/dev/null || true
grep -oE "replay exit [0-9]+: [0-9]+/[0-9]+ episode\(s\) reproduced(, not reproduced: [0-9, ]+)?" $O/convert.log \
  | sed -E 's/replay exit ([0-9]+): ([0-9]+)\/([0-9]+).*/\2 \3 exit\1/' | tee $O/result.txt
grep -oE "not reproduced: [0-9, ]+" $O/convert.log >> $O/result.txt || true
