#!/bin/bash
# ours: hand zero-shot pi0.5 datasets on our latent spaces (E-a, E-h, E2c), same recipe as the XL-VLA zero-shot baseline
# (pi05-hand-zeroshot: one task, 5 training hands = shadow held out, goal-marker renders teleop_lerobot_goal,
# wrist 9 + finger latent 32 = 41). CPU only. Finished datasets (all/meta/info.json) are skipped.
#   [TAGS="ea eh e2c"] [TASKS="GraspBleach GraspKettle GraspPan"] [PAR=3] bash make_zs.sh
# Output: $WS/datasets/dexverse/ours_latent/pi05-<task>-no-shadow-<tag>/all
set -u
WS=${WORKSPACE:-/workspace}; TAGS=${TAGS:-ea eh e2c}; TASKS=${TASKS:-GraspBleach GraspKettle GraspPan}; PAR=${PAR:-3}
HANDS=${HANDS:-inspire,sharpa,wuji,allegro,leap}
cd $WS/codes/dexverse/DexLatent
export XLVLA_TELEOP_ROOT=${XLVLA_TELEOP_ROOT:-$WS/datasets/dexverse/teleop_lerobot_goal} OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
WR=${WRIST_DIR:-$WS/checkpoints/dexverse/xlvla/hf6/_dl/latent/6hand-vae/wrist}; C=$WS/checkpoints/dexverse/ours_spaces
declare -A SPACE=([ea]=xlvla-local-pair-allegrofix [eh]=xlvla-local-pair-inspire60-hn [e2c]=xlvla-local-pair-staged-cont60k)
L=$WS/logs/pi05_ours/zs; mkdir -p $L
for tag in $TAGS; do for task in $TASKS; do
  out=$WS/datasets/dexverse/ours_latent/pi05-${task,,}-no-shadow-$tag
  [ -f $out/all/meta/info.json ] && { echo "$tag $task: exists"; continue; }
  while [ $(jobs -rp | wc -l) -ge $PAR ]; do sleep 10; done
  echo "$(date -u -d '+9 hours' '+%m-%d %H:%M') $tag $task: make -> $out"
  $WS/envs/dexverse/dexlatent/bin/python -m dexverse_xlvla.make_dataset --ckpt $C/${SPACE[$tag]}/model.pt --wrist $WR \
    --task Dexverse-$task-v0 --hands $HANDS --out $out > $L/make_${task}_$tag.log 2>&1 &
done; done
wait; echo "$(date -u -d '+9 hours' '+%m-%d %H:%M') all done"
