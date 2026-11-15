#!/bin/bash
# Check 3 of the latent checks during training (checks 1-2: latent_monitor.py): latent replay in the simulator
# (paper Tab. 4 protocol on a few episodes) on each latent checkpoint train_latent.py --monitor_every saves:
#   self  A -> latent -> A          episodes SELF_EPS of A's demos
#   cross A -> latent -> B (A != B) episodes CROSS_EPS of A's demos, on B with A's scene
# for every task and hand pair. Runs next to the training (container, nohup), until the final checkpoint is done:
#   R=<run dir> CK=<ckpt dir> TASKS="<t1> <t2>" HANDS="shadow inspire sharpa wuji" GPUS="0 1" bash replay_monitor.sh
# env: EVERY (checkpoint steps checked: multiples of it; default 1000; the final step always), SELF_EPS (default 0-2),
#      CROSS_EPS (default 0), DEXVERSE_WORKTREE (default DexVerse), XLVLA_PKL_ROOT, XLVLA_COMMON_DP (default the
#      worktree's baselines/lerobot_dp)
#      LAST_STEP (default config.yaml num_steps): the step whose check ends the monitor (also: cross replays with video)
# Wrist: wrist actions are absolute joint targets offset by the TASK's default joints, so every hand's wrist calibration
# is measured in each task's env first -> <run>/artifacts/wrist/wrist_calib_<task>.json (dexverse_xlvla.measure_wrist).
# Results: <run>/replay/<step>/<task>/<src>_to_<dst>/{eval,eval.log}; rows (step, eval, replay/<task>/<self|cross>_success,
# rate) in <run>/artifacts/monitor/replay.csv; one line per step and task in <run>/logs/replay_monitor.log.
# Reference (step "raw", once): every hand's own demos replayed unchanged (no latent) -> the ceiling of self replay.
# ONLY_RAW=1 runs just that reference (e.g. next to a running monitor). CROSS_ONLY=1 runs only the cross cells (no raw, no self;
# e.g. more cross episodes on a finished run: new R with its config.yaml + artifacts/wrist, CK with the chosen checkpoint).
# Cells are claimed with mkdir, one worker per GPU; rerunning skips finished cells.
set -u
: "${R:?}" "${CK:?}" "${TASKS:?}"
HANDS=${HANDS:-"shadow inspire sharpa wuji"}; GPUS=${GPUS:-0}; EVERY=${EVERY:-1000}
SELF_EPS=${SELF_EPS:-0-2}; CROSS_EPS=${CROSS_EPS:-0}
LAST=${LAST_STEP:-$(grep '^num_steps:' $R/config.yaml | awk '{print $2}')}   # LAST_STEP: last step to check when CK holds only some checkpoints
PY=/workspace/envs/dexverse/dexlatent/bin/python; CODE=/workspace/codes/dexverse/DexLatent
WT=${DEXVERSE_WORKTREE:-DexVerse}
export XLVLA_COMMON_DP=${XLVLA_COMMON_DP:-/workspace/codes/dexverse/$WT/baselines/lerobot_dp}
export XLVLA_PKL_ROOT=${XLVLA_PKL_ROOT:-/workspace/datasets/dexverse/teleop_pkl_bs1}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
W=$R/artifacts/wrist; M=$R/artifacts/monitor; Q=$R/replay
mkdir -p $W $M $Q/.claims $R/logs
skip() { [ $1 = raw ] && [ $2 != $3 ] && return 0; [ "${CROSS_ONLY:-0}" = 1 ] && [ $2 = $3 ] && return 0; return 1; }  # step src dst
log() { echo "[replay-monitor $(date -u -d '+9 hours' +%m-%d\ %H:%M)] $*" >> $R/logs/replay_monitor.log; }
pkl() { ls $XLVLA_PKL_ROOT/*/$1/${1}_*_floating_$2_right.pkl; }
n_done() { $PY -c "import json,sys; print(json.load(open(sys.argv[1]))['num_episodes'])" $1/metrics.json 2>/dev/null || echo 0; }
n_succ() { $PY -c "import json,sys; print(sum(e['success'] for e in json.load(open(sys.argv[1]))['episodes']))" $1/metrics.json 2>/dev/null || echo 0; }
eps() { seq -s, ${1%-*} ${1#*-}; }                       # 0-2 -> 0,1,2 ; 0 -> 0
neps() { eps $1 | tr , '\n' | wc -l; }
sim() {  # gpu cmd...  (main env, Isaac Sim)
  local g=$1; shift
  ( source /workspace/codes/_setup/dexverse_env.sh $WT >/dev/null; cd $DEXVERSE_ROOT
    unset DEXVERSE_FINGER_DRIVE DEXVERSE_DRIVE_SPECS; CUDA_VISIBLE_DEVICES=$g "$@" )
}

calib() {  # gpu task -> wrist_calib_<task>.json (one Isaac process per hand)
  local g=$1 t=$2 d=$W/$2 h
  [ -s $W/wrist_calib_$t.json ] && return 0
  mkdir -p $d
  for h in $HANDS; do
    [ -s $d/$h.json ] || sim $g timeout 900 python $CODE/dexverse_xlvla/measure_wrist.py --pkls $(pkl $t $h) --out $d/$h.json \
      --headless > $d/$h.log 2>&1
    [ -s $d/$h.json ] || { log "wrist calibration $t $h FAILED ($d/$h.log)"; return 1; }
  done
  $PY - $W/wrist_calib_$t.json $d $HANDS <<'EOF' >> $R/logs/replay_monitor.log
import json, sys, numpy as np
out, d, hands = sys.argv[1], sys.argv[2], sys.argv[3:]
cal = {}
for h in hands:
    cal.update(json.load(open(f"{d}/{h}.json")))
for h in hands:   # sanity: wrist screws non-zero, PoE model vs simulator on random wrist configurations
    c = cal[h]
    sn = min(np.linalg.norm(np.r_[w, v]) for w, v in c["screws"])
    print(f"[wrist] {out.split('wrist_calib_')[-1][:-5]} {h}: screw_min {sn:.2f}, check {c['check_pos_err_max_m'] * 1000:.3f} mm / "
          f"{c['check_rot_err_max_deg']:.4f} deg, offset {np.round(c['action_offset'], 3).tolist()}")
json.dump(cal, open(out, "w"), indent=1)
EOF
}

cell() {  # gpu step task src dst eps  -> 0 when the cell has all its episodes
  local g=$1 s=$2 t=$3 a=$4 b=$5 e=$6 o=$Q/$2/$3/${4}_to_$5 demos ms video=--no-video
  [ "$(n_done $o/eval)" = "$(neps $e)" ] && return 0
  rm -rf $o; mkdir -p $o
  if [ $s = raw ]; then
    $PY -m dexverse_xlvla.replay --task $t --src $a --dst $b --mode raw --out $o --episodes $e > $o/build.log 2>&1 || return 1
  else
    $PY -m dexverse_xlvla.replay --task $t --src $a --dst $b --mode latent --ckpt $CK/checkpoint_epoch_$s.pt \
      --wrist $W/wrist_calib_$t.json --out $o --episodes $e > $o/build.log 2>&1 || return 1
  fi
  if [ $a = $b ]; then demos=$(pkl $t $a); else demos=$o/demos.pkl; fi
  [ $s != raw ] && [ $((10#$s)) = $LAST ] && [ $a != $b ] && video=""                # final checkpoint: cross replays with video (visual check)
  ms=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['max_steps'])" $o/replay.json)
  sim $g python $CODE/dexverse_xlvla/run_eval.py --demos $demos --ckpt $o/policy --output_dir $o/eval \
    --episodes $(eps $e) --max_steps $ms $video --headless > $o/eval.log 2>&1
  [ "$(n_done $o/eval)" = "$(neps $e)" ]
}

summary() {  # step task -> csv rows + log line (once, by the worker that finishes the task's last cell)
  local s=$1 t=$2 a b ss=0 sn=0 cs=0 cn=0 per=""
  for a in $HANDS; do for b in $HANDS; do
    skip $s $a $b && continue
    [ -f $Q/$s/$t/${a}_to_$b/done ] || return 0                                                  # not finished yet
  done; done
  mkdir $Q/.claims/sum_${s}_$t 2>/dev/null || return 0
  for a in $HANDS; do for b in $HANDS; do
    skip $s $a $b && continue
    local k=$(n_succ $Q/$s/$t/${a}_to_$b/eval) n=$(n_done $Q/$s/$t/${a}_to_$b/eval)
    if [ $a = $b ]; then ss=$((ss + k)); sn=$((sn + n)); per="$per $a=$k/$n"; else cs=$((cs + k)); cn=$((cn + n)); fi
  done; done
  ( flock 9
    [ -s $M/replay.csv ] || echo "step,split,metric,value" >&9
    echo "$s,eval,replay/$t/self_success,$($PY -c "print(round($ss/max($sn,1),4))")" >&9
    echo "$s,eval,replay/$t/cross_success,$($PY -c "print(round($cs/max($cn,1),4))")" >&9 ) 9>>$M/replay.csv
  log "step $s $t: self $ss/$sn ($per ) cross $cs/$cn"
}

worker() {  # gpu: take cells of finished-but-unchecked checkpoints, oldest step first
  local g=$1 s t a b c
  while true; do
    for s in $([ "${CROSS_ONLY:-0}" = 1 ] || echo raw) $([ "${ONLY_RAW:-0}" = 1 ] || ls $CK 2>/dev/null | sed -n 's/^checkpoint_epoch_\([0-9]*\)\.pt$/\1/p' | sort -n); do
      [ $s = raw ] || [ $((10#$s % EVERY)) = 0 ] || [ $((10#$s)) = $LAST ] || continue   # names are zero-padded (0500)
      for t in $TASKS; do for a in $HANDS; do for b in $HANDS; do
        skip $s $a $b && continue
        c=${s}_${t}_${a}_$b
        [ -e $Q/.claims/$c ] && continue
        mkdir $Q/.claims/$c 2>/dev/null || continue
        if cell $g $s $t $a $b $([ $a = $b ] && echo $SELF_EPS || echo $CROSS_EPS); then touch $Q/$s/$t/${a}_to_$b/done
        else log "step $s $t $a->$b FAILED ($Q/$s/$t/${a}_to_$b)"; touch $Q/$s/$t/${a}_to_$b/done; fi
        summary $s $t
      done; done; done
    done
    fin=$([ "${ONLY_RAW:-0}" = 1 ] && echo raw || echo $LAST)
    [ -z "$(for t in $TASKS; do [ -d $Q/.claims/sum_${fin}_$t ] || echo x; done)" ] && break   # final step summarised
    sleep 60
  done
}

log "start: tasks $TASKS, hands $HANDS, GPUs $GPUS, every $EVERY (last $LAST), self eps $SELF_EPS, cross eps $CROSS_EPS, cross only ${CROSS_ONLY:-0}, worktree $WT"
G0=${GPUS%% *}
for t in $TASKS; do calib $G0 $t || { log "stop: no wrist calibration for $t"; exit 1; }; done
log "wrist calibrations ready"
for g in $GPUS; do worker $g & done
wait
log "ALL DONE"
