#!/bin/bash
# Space-vs-ceiling evaluation (RSS2027): hand s's teleop episodes replayed on hand t through the conversion replay path
# (pickle_to_lerobot -> record_demos --replay_demos, same as pkl_replay.sh), scene = s's episode, wrist converted
# through the canonical frame. Fingers: dexpilot = DexPilot(s->t) (ceiling) / latent = D_t(E_s(q_s)) of <ckpt> / contact = world fingertip match (oracle intent).
#   bash cross_pkl_replay.sh <task> <src> <dst> <dexpilot|latent|contact|oracle_fix> <episodes> <out dir> [ckpt]
# DIAG=1: render every episode (failed too) + per-step diagnostics instead of the conversion (diag_replay.py).
# Wrist calibration: $WRIST_CALIB_DIR/wrist_calib_<task>.json (dexverse_xlvla.measure_wrist).
set -euo pipefail
T=$1 S=$2 D=$3 MODE=$4 EPS=$5 O=$6 CKPT=${7:-}
PY=/workspace/envs/dexverse/dexlatent/bin/python; MAIN=/workspace/envs/dexverse/main/bin/python
CODE=/workspace/codes/dexverse/DexLatent; WT=${DEXVERSE_WORKTREE:-DexVerse}; HERE=$(cd $(dirname $0) && pwd)
WRIST_CALIB_DIR=${WRIST_CALIB_DIR:-/workspace/checkpoints/dexverse/xlvla/wrist_calib}
export XLVLA_PKL_ROOT=${XLVLA_PKL_ROOT:-/workspace/datasets/dexverse/teleop_pkl_bs1}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
bash /workspace/codes/_setup/gpu_watch.sh ensure >/dev/null 2>&1 || true
mkdir -p $O; O=$(cd $O && pwd)
TMP=/workspace/.tmp/xrep_$(basename $(dirname $O))_$(basename $O)_$$; rm -rf $TMP; mkdir -p $TMP/in
trap 'rm -rf $TMP' EXIT
( cd $CODE && PYTHONPATH=/workspace/codes/cross_dex $PY -m cross_dex.ours.diag.cross_build --task $T --src $S --dst $D \
    --mode $MODE ${CKPT:+--ckpt $CKPT} --wrist $WRIST_CALIB_DIR/wrist_calib_$T.json --episodes $EPS --out $O/build ) > $O/build.log 2>&1
NAME=$(basename $($MAIN -c "import json;print(json.load(open('$O/build/demos_spec.json'))['dst_pkl'])"))
$MAIN $HERE/make_replay_pkl.py $O/build/demos.pkl cross $EPS $TMP/in/$NAME $O/build/actions.npz >> $O/build.log 2>&1
rm -f $O/build/demos.pkl
if [ -n "${DIAG:-}" ]; then   # every episode rendered + per-step diagnostics (diag_replay.py) -> $O/diag
  ( source /workspace/codes/_setup/dexverse_env.sh $WT >/dev/null; cd $DEXVERSE_ROOT
    unset DEXVERSE_FINGER_DRIVE DEXVERSE_DRIVE_SPECS; export PYTHONUNBUFFERED=1 PYTHONPATH=$DEXVERSE_ROOT/source/dexverse DIAG_OUT=$O/diag
    RT=$(python -c "import pickle;print(pickle.load(open('$TMP/in/$NAME','rb'))['robot_type'])")
    TK=$(python -c "import pickle;print(pickle.load(open('$TMP/in/$NAME','rb'))['task'])")
    python $HERE/diag_replay.py --task $TK --robot_type $RT --replay_demos $TMP/in/$NAME --dataset_file $TMP/replay.pkl \
      --num_demos 0 --headless --datasets_dir $TMP/out --lerobot_root $TMP/out/ds \
      --lerobot_python /workspace/envs/dexverse/lerobot/bin/python
    while pgrep -f "spool_writer.*$TMP" >/dev/null; do sleep 5; done ) > $O/diag_replay.log 2>&1 || true
  mkdir -p $O/video; find $TMP/out -name "*.mp4" -path "*third*" | sort | xargs -r -I{} cp --backup=numbered {} $O/video/
  find $TMP/out -name "*.mp4" | sed "s#$TMP/out/##" > $O/video/all_mp4.txt
  cat $O/diag/summary.jsonl; exit 0
fi
( source /workspace/codes/_setup/dexverse_env.sh $WT >/dev/null; cd $DEXVERSE_ROOT
  unset DEXVERSE_FINGER_DRIVE DEXVERSE_DRIVE_SPECS; export PYTHONUNBUFFERED=1
  python scripts/data_tools/pickle_to_lerobot.py $TMP/in/$NAME --datasets_dir $TMP/out \
    --lerobot_python /workspace/envs/dexverse/lerobot/bin/python ) > $O/convert.log 2>&1 || true
cp $TMP/out/trajectories_replayed/*_replay.log $O/replay.log 2>/dev/null || true
if [ -n "${KEEP_VIDEO:-}" ]; then mkdir -p $O/video; find $TMP/out -name "*.mp4" | head -4 | xargs -r -I{} cp {} $O/video/; fi
grep -oE "replay exit [0-9]+: [0-9]+/[0-9]+ episode\(s\) reproduced(, not reproduced: [0-9, ]+)?" $O/convert.log \
  | sed -E 's/replay exit ([0-9]+): ([0-9]+)\/([0-9]+).*/\2 \3 exit\1/' | tee $O/result.txt
grep -oE "not reproduced: [0-9, ]+" $O/convert.log >> $O/result.txt || true
