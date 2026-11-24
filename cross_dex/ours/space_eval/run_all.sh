#!/bin/bash
# One space through every column of the "전체 지표" table (2026-11-19). Run inside the container (/workspace layout).
#   bash run_all.sh <adapter.py> <ckpt> <name> [bench=dv_right]
# env: SPACE_EVAL_DATA  eval sets root (default /workspace/datasets/dexverse/lad/eval_space; HF download, README)
#      ADAPTER_PY       python that can import your model (default dexlatent env); WORK  output dir
#      CUDA_VISIBLE_DEVICES as usual (empty = CPU, ~10 min per set on 8 cores)
set -euo pipefail
AD=$(readlink -f "$1"); CK=$2; N=$3; B=${4:-dv_right}
K=$(cd "$(dirname "$0")" && pwd)
DATA=${SPACE_EVAL_DATA:-/workspace/datasets/dexverse/lad/eval_space}
DL=${DEXLATENT_DIR:-/workspace/codes/dexverse/DexLatent}
PY=${DEXLATENT_PY:-/workspace/envs/dexverse/dexlatent/bin/python}
APY=${ADAPTER_PY:-$PY}
WORK=${WORK:-/workspace/.tmp/space_eval_kit/$N}
LADP=$K/../../baselines/lad
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8} PYTHONPATH=$LADP:${PYTHONPATH:-}
for s in $B ${B}_demopair ${B}_bundle; do
  mkdir -p $WORK/$s; for f in $DATA/$s/inputs*.npz; do ln -sf $f $WORK/$s/; done; done
echo "[1/4] decode (full metric set, demo-pair set)"
for s in $B ${B}_demopair; do (cd $DL && $APY $K/run_space.py decode --adapter $AD --ckpt $CK --name $N --out $WORK/$s); done
echo "[2/4] reconstruction metrics"
for s in $B ${B}_demopair; do (cd $DL && $PY -m dexverse_lad.eval_space metrics --out $WORK/$s | grep "^$N "); done
echo "[3/4] latents + pair clustering"
for s in demopair bundle; do (cd $DL && $APY $K/run_space.py zdump --adapter $AD --ckpt $CK --name $N --out $WORK/${B}_$s --zout $WORK/z_${s}_$N.npz); done
$PY $K/pair_metrics.py --demopair $WORK/${B}_demopair --z_demopair $WORK/z_demopair_$N.npz \
    --bundle $WORK/${B}_bundle --z_bundle $WORK/z_bundle_$N.npz --json $WORK/pair_metrics_$N.json
echo "[4/4] table"
$PY $K/full_row.py --work $WORK --name $N --bench $B
