#!/bin/bash
# ours: self-residual modules f_h for the composition (diagonal) pi0.5 policies (train_mt.sh DIAG=1), 2026-11-25.
# The held-out cells GB:shadow, GK:sharpa, GP:inspire are unseen by the policy, so their f_h is fitted on that hand's demos of
# the other tasks of the diagonal training set only (crosshand_dp/residual_fix.py --other_task_hands); every other cell uses its
# own demos (episodes 0-44, as the multitask modules). Output: <space dir>/residual_fix_diag/<task>.{pt,json}, used by
#   TAG=<t> DIAG=1 RESID_FIX_DIR=<space dir>/residual_fix_diag bash eval_mt.sh
#   TAG=ea|eh|e2c [GPU=2] bash make_rfix_diag.sh        (dexlatent env; finished tasks are skipped)
set -eu
WS=${WORKSPACE:-/workspace}; TAG=${TAG:?ea|eh|e2c}; GPU=${GPU:-0}
declare -A SPACE=([ea]=xlvla-local-pair-allegrofix [eh]=xlvla-local-pair-inspire60-hn [e2c]=xlvla-local-pair-staged-cont60k)
declare -A HELD=([Dexverse-GraspBleach-v0]=shadow [Dexverse-GraspKettle-v0]=sharpa [Dexverse-GraspPan-v0]=inspire)
ALL=${ALL_TASKS:-Dexverse-GraspBleach-v0,Dexverse-GraspKettle-v0,Dexverse-GraspPan-v0,Dexverse-FunctionalHammerStrike-v0}
D=$WS/checkpoints/dexverse/ours_spaces/${SPACE[$TAG]}; O=$D/residual_fix_diag; mkdir -p $O
cd $WS/codes/dexverse/DexLatent
for t in "${!HELD[@]}"; do
  [ -s $O/$t.pt ] && [ -s $O/$t.json ] && { echo "$t: done, skipped"; continue; }
  others=$(echo $ALL | tr ',' '\n' | grep -vx $t | paste -sd,)
  CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=$WS/codes/cross_dex $WS/envs/dexverse/dexlatent/bin/python -m cross_dex.ours.crosshand_dp.residual_fix \
    --latent_ckpt $D/model.pt --task $t --train_tasks $others --other_task_hands ${HELD[$t]} --out $O/$t.pt.tmp --json $O/$t.json
  mv $O/$t.pt.tmp $O/$t.pt
done
echo "ALL DONE $O"
