#!/bin/bash
# One evaluation cell of a multi-hand DP (scheduler job): <hand>'s pickle episodes, closed loop in the common evaluator.
#   bash run_cell.sh <arm> <hand> <policy pretrained_model> <episodes comma> <out dir> <noise_seed|none> <gpu> [--video]
# FIX=<residual_fix.py output>: decode fingers as D_h(z) + f_h(z) (self-residual module, 2026-11-23).
# LOG_DIR=<dir>: per-episode npz of the policy server (policy_out, action, observed q).
# WRIST_F=palm|const:<json>|f:<ckpt> (+WRIST_SRC=source hands, XWO): wrist offset model at inference (wrist_pairs/f_wrist.py), space arms.
# out: server.json (policy descriptor), eval/metrics.json (+ videos with --video), eval.log. Done = metrics.json has all episodes.
set -uo pipefail
ARM=$1 HAND=$2 POL=$3 EPS=$4 O=$5 SEED=$6 GPU=$7 VID=${8:---no-video}
TASK=Dexverse-GraspKettle-v0; SP=/workspace/checkpoints/dexverse/crosshand_dp/spaces
WRIST=${WRIST:-/workspace/checkpoints/dexverse/xlvla/hf6/latent/wrist/wrist_calib_$TASK.json}
PKL=$(ls ${XLVLA_PKL_ROOT:-/workspace/datasets/dexverse/teleop_pkl_bs1}/*/$TASK/${TASK}_*_floating_${HAND}_right.pkl)
HERE=$(cd $(dirname $0) && pwd)
if [[ $EPS == unique* ]]; then   # first N episodes with distinct initial layouts (shared_eval/pi05/unique_eps.py, UNIQUE_LAYOUTS)
  EPS=$(/workspace/envs/dexverse/main/bin/python $HERE/../shared_eval/pi05/unique_eps.py $PKL $(seq -s, 0 $((${EPS#unique} - 1)))) || exit 1
fi
N=$(echo $EPS | tr , '\n' | wc -l)
export XLVLA_TELEOP_ROOT=${XLVLA_TELEOP_ROOT:-/workspace/datasets/dexverse/teleop_lerobot_goal}
export XLVLA_COMMON_DP=${XLVLA_COMMON_DP:-/workspace/codes/dexverse/DexVerse/baselines/lerobot_dp}   # image_resize for the policy server
n_done() { python3 -c "import json; print(json.load(open('$O/eval/metrics.json'))['num_episodes'])" 2>/dev/null || echo 0; }
[ "$(n_done)" = "$N" ] && exit 0
bash /workspace/codes/_setup/gpu_watch.sh ensure >/dev/null 2>&1 || true
mkdir -p $O; rm -rf $O/eval
CK=$SP/$([ $ARM = joint ] && echo xlvla || echo $ARM)/model.pt
python3 - <<EOF
import json
json.dump({"arm": "$ARM", "hand": "$HAND", "task": "$TASK", "policy_ckpt": "$POL", "latent_ckpt": "$CK", "wrist": "$WRIST",
           "episodes": [int(e) for e in "$EPS".split(",")], "noise_seed": None if "$SEED" == "none" else int("$SEED"),
           "hand_onehot": ["shadow", "inspire", "sharpa", "wuji", "allegro", "leap"], "log_dir": "${LOG_DIR:-}" or None,
           "residual_fix": "${FIX:-}" or None,
           "wrist_f": {"spec": "${WRIST_F:-}", "xwo": "${XWO:-/workspace/.tmp/xwo}", "sources": "${WRIST_SRC:-sharpa,wuji,allegro,leap}".split(",")} if "${WRIST_F:-}" else None},
          open("$O/server.json", "w"), indent=1)
EOF
# stall guard (2026-11-21): Isaac sometimes spins forever after "OgnSdOnNewFrame ... discarded" and ignores SIGTERM,
# so kill the cell's process group when no "[eval]" progress line has appeared for STALL_S (default 40 min).
setsid bash -c "source /workspace/codes/_setup/dexverse_env.sh DexVerse >/dev/null; cd \$DEXVERSE_ROOT
  unset DEXVERSE_FINGER_DRIVE DEXVERSE_DRIVE_SPECS
  CUDA_VISIBLE_DEVICES=$GPU exec python $HERE/run_eval.py --demos $PKL --ckpt $O --output_dir $O/eval \
    --episodes $EPS --max_steps ${MAX_STEPS:-1500} $VID --headless" > $O/eval.log 2>&1 &
PG=$!; last=-1; t=$(date +%s)
while kill -0 $PG 2>/dev/null; do
  sleep 60; c=$(grep -c '^\[eval\]' $O/eval.log 2>/dev/null); c=${c:-0}
  if [ "$c" != "$last" ]; then last=$c; t=$(date +%s)
  elif [ $(( $(date +%s) - t )) -gt ${STALL_S:-2400} ]; then echo "[cell] no progress for ${STALL_S:-2400}s -> kill" >> $O/eval.log; kill -9 -- -$PG; fi
done
[ "$(n_done)" = "$N" ]
