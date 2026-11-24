#!/bin/bash
# Action ceiling of one arm on one hand (GraspKettle): replay_build.py actions -> pickle -> the conversion replay path
# (pickle_to_lerobot.py -> record_demos.py --replay_demos, cameras on; the path that reproduces the teleop demos, while
# the policy evaluator eval_lerobot replays only 1/10, 2026-11-19). Open loop from the recorded initial states.
#   bash replay.sh <hand> <raw|joint|xlvla|ea|eh|e2c> <episodes, e.g. 0-9> <out dir> [gpu]
# result.txt = "<reproduced> <episodes> exit<rc>"; the rendered LeRobot output is deleted at the end.
set -euo pipefail
HAND=$1 ARM=$2 EPS=$3 O=$4 G=${5:-${CUDA_VISIBLE_DEVICES:-0}}
SP=${SPACES:-/workspace/checkpoints/dexverse/crosshand_dp/spaces}
WRIST=${WRIST:-/workspace/checkpoints/dexverse/xlvla/hf6/latent/wrist/wrist_calib_Dexverse-GraspKettle-v0.json}
PY=/workspace/envs/dexverse/dexlatent/bin/python; MAIN=/workspace/envs/dexverse/main/bin/python
CODE=/workspace/codes/dexverse/DexLatent; WT=${DEXVERSE_WORKTREE:-DexVerse}; HERE=$(cd $(dirname $0) && pwd)
export XLVLA_PKL_ROOT=${XLVLA_PKL_ROOT:-/workspace/datasets/dexverse/teleop_pkl_bs1}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=/workspace/codes/cross_dex
bash /workspace/codes/_setup/gpu_watch.sh ensure >/dev/null 2>&1 || true
mkdir -p $O; O=$(cd $O && pwd)
[ -f $O/result.txt ] && grep -q exit $O/result.txt && { cat $O/result.txt; exit 0; }
SRC=$(ls $XLVLA_PKL_ROOT/*/Dexverse-GraspKettle-v0/Dexverse-GraspKettle-v0_*_floating_${HAND}_right.pkl)
CK=; [ $ARM != raw ] && CK="--ckpt $SP/$([ $ARM = joint ] && echo xlvla || echo $ARM)/model.pt"
TMP=/workspace/.tmp/xhdp_replay_${HAND}_${ARM}_$$; rm -rf $TMP; mkdir -p $TMP/in
trap 'rm -rf $TMP' EXIT
( cd $CODE && $PY -m cross_dex.ours.crosshand_dp.replay_build actions --hand $HAND --arm $ARM $CK --wrist $WRIST \
    --eps $EPS --out $O/actions.npz ) > $O/build.log 2>&1
$MAIN $HERE/replay_build.py pickle $SRC $O/actions.npz $TMP/in/$(basename $SRC) >> $O/build.log 2>&1
( source /workspace/codes/_setup/dexverse_env.sh $WT >/dev/null; cd $DEXVERSE_ROOT
  unset DEXVERSE_FINGER_DRIVE DEXVERSE_DRIVE_SPECS; export PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=$G
  timeout ${REPLAY_TIMEOUT:-3600} python scripts/data_tools/pickle_to_lerobot.py $TMP/in/$(basename $SRC) --datasets_dir $TMP/out \
    --lerobot_python /workspace/envs/dexverse/lerobot/bin/python ) > $O/convert.log 2>&1 || true
cp $TMP/out/trajectories_replayed/*_replay.log $O/replay.log 2>/dev/null || true
grep -oE "replay exit [0-9]+: [0-9]+/[0-9]+ episode\(s\) reproduced(, not reproduced: [0-9, ]+)?" $O/convert.log \
  | sed -E 's/replay exit ([0-9]+): ([0-9]+)\/([0-9]+).*/\2 \3 exit\1/' > $O/result.txt || echo "0 0 failed" > $O/result.txt
grep -oE "not reproduced: [0-9, ]+" $O/convert.log >> $O/result.txt || true
cat $O/result.txt
