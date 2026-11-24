#!/bin/bash
# One multi-hand DP training (scheduler job; resumes from checkpoints/last when relaunched with the same arm/exp).
#   bash run_train.sh <arm: xlvla|ea|eh|e2c|joint> <exp: a6|b_noshadow|cg> <gpu> [num_workers=12]   env STEPS, SAVE, BATCH
#   cg = 4 tasks x 6 hands minus GraspKettle x shadow (make_data_mt.py); 2026-11-23: BATCH=192 STEPS=68304 as the multi-task
#   XL-VLA comparator (paper-based latent-dp-mt-diag)
# run folder : /workspace/experiments/<server>/RSS2027/comparison/dexverse-crosshand-dp/graspkettle-<a6|b-noshadow>/<run_id>
# checkpoint : /workspace/checkpoints/dexverse/crosshand_dp/<run_id>/checkpoints/<step>/pretrained_model (50k, 100k, 150k, 200k)
# run_id     : fixed at the first start on this server (<ckpt root>/runs/<exp>_<arm>.rid), so a retry resumes.
# Finished   : <run>/logs/trained. 200k steps, snapshots every 50k, seed 42, no evaluation during training.
set -uo pipefail
WS=${WORKSPACE:-/workspace}   # workspace root: /workspace in containers; a host folder on container-less servers (hosts ws=)
# container-less servers (bw2) have no system FFmpeg for torchcodec: the workspace copy (as pi05_space/train_mt.sh)
ulimit -n 65536 2>/dev/null || ulimit -n $(ulimit -Hn)   # bw2 soft limit 1024: loader workers hit "Too many open files"
[ -d $WS/envs/tools/ffmpeg/lib ] && export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$WS/envs/tools/ffmpeg/lib
ARM=$1 EXP=$2 GPU=$3 NW=${4:-12} STEPS=${STEPS:-200000} SAVE=${SAVE:-50000} BATCH=${BATCH:-64}
SERVER=$(cd $WS/experiments && ls -d */ | tr -d / | grep -v '^[_.]' | head -1)
HERE=$(cd $(dirname $0) && pwd); LP=$WS/envs/dexverse/lerobot/bin/python
CKR=$WS/checkpoints/dexverse/crosshand_dp; DATA=$WS/datasets/dexverse/crosshand_dp/$ARM/$EXP
mkdir -p $CKR/runs; RIDF=$CKR/runs/${EXP}_$ARM.rid
[ -s $RIDF ] || echo "$(date -u -d '+9 hours' +%Y%m%d-%H%M%S)_${SERVER}_xhdp-${EXP%%_*}-$ARM" > $RIDF
RID=$(cat $RIDF); C=$CKR/$RID
R=$WS/experiments/$SERVER/RSS2027/comparison/dexverse-crosshand-dp/graspkettle-${EXP/_/-}/$RID
mkdir -p $R/logs $R/artifacts
bash $WS/codes/_setup/gpu_watch.sh ensure >/dev/null 2>&1 || true
[ -f $R/logs/trained ] && { echo "[train] $RID already trained"; exit 0; }
GIT=$(git -C $WS/codes/cross_dex rev-parse HEAD 2>/dev/null || echo unknown)
if [ ! -f $R/meta.json ]; then
  python3 - "$R" "$RID" "$SERVER" "$ARM" "$EXP" "$DATA" "$C" "$GIT" "$*" "$NW" "$STEPS" "$SAVE" "$BATCH" <<'EOF'
import json, sys, datetime
r, rid, server, arm, exp, data, c, git, args, nw, steps, save, batch = sys.argv[1:]
now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9))).isoformat(timespec="seconds")
json.dump({"run_id": rid, "project": "dexverse-crosshand-dp", "server": server, "agent": "claude-code",
           "git_repo": "https://github.com/HyeonseokE/cross_dex", "git_commit": git, "git_dirty": True, "docker_image": None,
           "command": f"bash cross_dex/ours/crosshand_dp/run_train.sh {args}", "seed": 42, "start_time": now,
           "end_time": None, "status": "running"}, open(f"{r}/meta.json", "w"), indent=1)
hands = json.load(open(f"{data}/meta/crosshand_cells.json"))
extra = (f"tasks: {hands['tasks']}\nexcluded_cells: {hands['excluded']}\n" if "tasks" in hands else "")
open(f"{r}/config.yaml", "w").write(
    f"task: Dexverse-GraspKettle-v0\nexperiment: {exp}\narm: {arm}\nhands: {hands['hands']}\nspace_ckpt: {hands['ckpt']}\n"
    f"wrist: {hands['wrist']}\ndataset_root: {data}\nsource_root: {hands['source_root']}\ncheckpoints: {c}\n"
    f"policy: lerobot diffusion (separate RGB encoder per camera, 240x320 resize in the data loader, crop 216x288)\n"
    f"steps: {steps}\nsave_freq: {save}\nbatch_size: {batch}\nlr: 1.0e-4 (LeRobot diffusion default, cosine, warmup 500)\n"
    f"seed: 42\nnum_workers: {nw}\nhand_onehot: {hands['hand_onehot_order']}\npad: {hands['pad']}\n" + extra)
EOF
fi
for attempt in 1 2 3; do
  RES=; [ -e $C/checkpoints/last ] && RES=--resume
  [ -z "$RES" ] && rm -rf $C   # nothing saved yet: LeRobot refuses an existing output_dir without --resume
  echo "[train] $RID gpu $GPU workers $NW attempt $attempt $RES $(date -u -d '+9 hours' '+%F %T') KST" >> $R/logs/train.log
  env -u LD_PRELOAD -u PYTHONPATH CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 \
    $LP $HERE/train_dp.py --dataset_root $DATA --output_dir $C --steps $STEPS --save_freq $SAVE --batch_size $BATCH --num_workers $NW $RES >> $R/logs/train.log 2>&1
  S=$(python3 -c "import json; print(json.load(open('$C/checkpoints/last/training_state/training_step.json'))['step'])" 2>/dev/null)
  [ "$S" = "$STEPS" ] && { touch $R/logs/trained; echo "[train] $RID done"; exit 0; }
  echo "[train] $RID stopped at step ${S:-none}" >> $R/logs/train.log; sleep 120
done
exit 1
