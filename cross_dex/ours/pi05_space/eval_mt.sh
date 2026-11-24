#!/bin/bash
# ours: final evaluation of the one-arm pi0.5 multitask policies trained on our latent spaces (train_mt.sh), with the same
# convention as the XL-VLA / joint pi0.5 one-arm goal evaluations (evq jobs mt-joint-goal-r*): GraspBleach, GraspKettle,
# GraspPan x 6 hands, 30 episodes (0-29, distinct initial layouts), MAX_STEPS 1500, one shared openpi server per run
# (shared_eval/pi05/eval_shared.sh, called unchanged), repeat r uses NOISE=episode NOISE_SEED=r. One run dir per repeat.
#   TAG=ea|eh|e2c [SEED=42] [REPS="1 2 3"] GPU=2 WORKERS=4 bash eval_mt.sh
# Repeats run one after another; a repeat whose run dir already finished (eval/QUEUE_DONE = ok) is skipped, an unfinished
# one is resumed (finished cells skipped by eval_shared.sh). Workspace root from WORKSPACE (default /workspace). Times are KST via date -u +9 h (containers may lack tzdata).
set -u
WS=${WORKSPACE:-/workspace}; GPU=${GPU:-0}; WORKERS=${WORKERS:-4}; REPS=${REPS:-1 2 3}
TAG=${TAG:?ea|eh|e2c}; SEED=${SEED:-42}; SFX=; [ $SEED = 42 ] || SFX=-s$SEED
# RESID_FIX_DIR=<dir with <task>.pt from crosshand_dp/residual_fix.py>: decode fingers as D_h(z) + f_h(z) (run slug +-rfix, 2026-11-23)
[ -n "${RESID_FIX_DIR:-}" ] && { export RESID_FIX_DIR; RFX=-rfix; }
# Policy variant (2026-11-24): default = 6-hand multitask (train_mt.sh); DIAG=1 = composition (diagonal) model (train_mt.sh DIAG=1),
# evaluated on the same 18 cells (held-out GB:shadow, GK:sharpa, GP:inspire among them); ZS_TASK=GraspKettle = hand zero-shot model
# (train_zs.sh, shadow held out), evaluated on that task x 6 hands.
TASKS=${TASKS:-Dexverse-GraspBleach-v0,Dexverse-GraspKettle-v0,Dexverse-GraspPan-v0}
RUN=ours-pi05-6hand-mt-goal-$TAG; REPO=pi05-6hand-mt-goal-$TAG; EXP=$TAG-mt-bs32$SFX; STEP=19999; KIND=mt
# TAG=xl (2026-11-24): the XL-VLA baseline pi0.5 multitask policy and its 6-hand VAE (for ours' residual module on the XL space)
[ $TAG = xl ] && { RUN=xlvla-pi05-6hand-mt-goal; REPO=pi05-6hand-mt-goal; EXP=vae6-mt-bs32$SFX; }
if [ -n "${DIAG:-}" ]; then RUN=ours-pi05-6hand-mt-goal-diag-$TAG; REPO=pi05-6hand-mt-goal-diag-$TAG; EXP=$TAG-mt-diag-bs32$SFX; KIND=mt-diag
elif [ -n "${ZS_TASK:-}" ]; then k=${ZS_TASK,,}; RUN=ours-pi05-$k-no-shadow-$TAG; REPO=pi05-$k-no-shadow-$TAG; EXP=$TAG-5hand-5k; STEP=4999
  KIND=zs-$k; TASKS=Dexverse-$ZS_TASK-v0; fi
NCELL=$(( $(echo $TASKS | tr ',' '\n' | wc -l) * 6 ))
SERVER=${SERVER:-$(find $WS/experiments -mindepth 1 -maxdepth 1 -type d ! -name '_*' ! -name '.*' -printf '%f\n' | head -1)}
declare -A SPACE=([ea]="E-a xlvla-local-pair-allegrofix" [eh]="E-h xlvla-local-pair-inspire60-hn" [e2c]="E2c xlvla-local-pair-staged-cont60k" [xl]="XL-VLA-6hand-VAE -")
set -- ${SPACE[$TAG]}; NAME=$1; HFN=$2
B=$WS/experiments/$SERVER/RSS2027/ours/dexverse-ours-pi05/pi05-space-mt-eval
H6=$WS/checkpoints/dexverse/xlvla/hf6
export CK=${CK:-$WS/checkpoints/dexverse/xlvla/pi05/checkpoints/$RUN/$EXP/$STEP}
XLAT=$WS/checkpoints/dexverse/ours_spaces/$HFN/model.pt; [ $TAG = xl ] && XLAT=$WS/checkpoints/dexverse/xlvla/hf6/_dl/latent/6hand-vae/checkpoint_epoch_25000.pt
LSD="ours: $NAME (HF Dexverse-CSI/shared-action-spaces ablation/dexverse/$HFN/right-6hand/model.pt)"; [ $TAG = xl ] && LSD="XL-VLA 6-hand VAE (hf6 checkpoint_epoch_25000)"
export LATENT=${LATENT:-$XLAT} WRIST_DIR=${WRIST_DIR:-$H6/_dl/latent/6hand-vae/wrist}
export MODE=pi05_latent PI05_RUN=$RUN PI05_REPO=$REPO PI05_ADIM=41 TASKS \
  OPENPI_DATA_HOME=$WS/.cache/openpi EPS=$(seq -s, 0 29) MAX_STEPS=1500 SHARED=1 GATHER_MS=5 NOISE=episode \
  HANDS=shadow,inspire,sharpa,wuji,allegro,leap
E=$WS/codes/cross_dex/cross_dex/ours/shared_eval/pi05/eval_shared.sh
ls $CK/params $CK/assets/$PI05_REPO $LATENT >/dev/null || { echo "missing inputs"; exit 1; }
COMMIT=$(git -C $WS/codes/cross_dex rev-parse --short HEAD 2>/dev/null || echo 기록 없음)
mkdir -p $WS/logs/pi05_ours

for rep in $REPS; do
  slug=pi05-$TAG-$KIND-goal-ueps-r$rep$SFX${RFX:-}
  export R=$(ls -d $B/*_${SERVER}_$slug 2>/dev/null | tail -1) NOISE_SEED=$rep
  if [ -n "$R" ] && [ "$(cat $R/eval/QUEUE_DONE 2>/dev/null)" = ok ]; then echo "$slug: done, skipped"; continue; fi
  if [ -z "$R" ]; then
    R=$B/$(date -u -d "+9 hours" +%Y%m%d-%H%M%S)_${SERVER}_$slug; mkdir -p $R/logs
    cat > $R/meta.json <<JSON
{"run_id": "$(basename $R)", "project": "pi05-space-mt-eval", "server": "$SERVER", "agent": "claude-code", "eval_repeat": $rep,
 "command": "TAG=$TAG${DIAG:+ DIAG=$DIAG}${ZS_TASK:+ ZS_TASK=$ZS_TASK}${RESID_FIX_DIR:+ RESID_FIX_DIR=$RESID_FIX_DIR} SEED=$SEED REPS=$rep GPU=$GPU WORKERS=$WORKERS bash cross_dex/ours/pi05_space/eval_mt.sh -> eval_shared.sh (NOISE=episode NOISE_SEED=$rep, EPS 0-29, MAX_STEPS 1500)",
 "seed": $SEED, "latent_space": "$LSD",
 "policy_ckpt": "$CK", "residual_fix_dir": "${RESID_FIX_DIR:-}", "git_repo": "HyeonseokE/cross_dex", "git_commit": "$COMMIT", "git_dirty": null, "docker_image": null,
 "start_time": "$(date -u -d "+9 hours" +%Y-%m-%dT%H:%M:%S+09:00)", "end_time": null, "status": "running"}
JSON
  fi
  echo "$(date -u -d "+9 hours" "+%m-%d %H:%M") $slug: GPU$GPU $WORKERS sessions -> $R"
  GPU=$GPU WORKERS=$WORKERS bash $E >> $WS/logs/pi05_ours/eval_${slug}.out 2>&1
  n=$(grep -h 'want 30' $R/logs/eval_shared.log 2>/dev/null | sed 's/^\[[^]]*\] //' | sort -u | grep -c '/30')
  st=failed; [ "$n" -ge $NCELL ] && st=done
  mkdir -p $R/eval; echo $([ $st = done ] && echo ok || echo fail) > $R/eval/QUEUE_DONE
  python3 - $R/meta.json $st <<'PY'
import json, sys, subprocess
p, st = sys.argv[1:3]; m = json.load(open(p)); m["status"] = st
m["end_time"] = subprocess.run(["date", "-u", "-d", "+9 hours", "+%Y-%m-%dT%H:%M:%S+09:00"], capture_output=True, text=True).stdout.strip()
json.dump(m, open(p, "w"), ensure_ascii=False, indent=1)
PY
  echo "$(date -u -d "+9 hours" "+%m-%d %H:%M") $slug: $n/$NCELL cells ($st)"
  [ $st = done ] || exit 1
done
