#!/bin/bash
# PC background: bring vast-isaac2 into the multi-hand DP schedule without the main agent (2026-11-20).
#  1 wait for the lerobot env -> scheduler: vast-isaac2 on, training cap 1/GPU -> 2 measurement trainings (a6/ea, a6/xlvla)
#  2 measure (measure.py, 600 s) -> cap 2/GPU + 2 more trainings -> measure again -> keep the cap whose per-GPU step rate
#    is >= 5 % higher (EXPERIMENT_RULES 7.1); CPU (46 cores) and RAM (259 GB x 0.8) also bound the cap
#  3 release every remaining training (measured_done) ; 4 wait for the main env -> eval_ready_vast-isaac2
#   setsid nohup bash vast_bringup.sh >> ~/.cache/hscho_share/xhdp/bringup.log 2>&1 &
export SSH_AUTH_SOCK=~/.ssh/agent.sock
ST=~/.cache/hscho_share/xhdp; HERE=$(cd $(dirname $0) && pwd); CFG=$HERE/sched_servers.json
V="ssh -o LogLevel=ERROR -o ClearAllForwardings=yes vast-isaac2"
log() { echo "[bringup $(date '+%m-%d %H:%M')] $*"; }
restart_sched() {
  for p in $(pgrep -f '^python3 codes/_setup/gpu_sched/sched.py run'); do
    tr '\0' '\n' < /proc/$p/environ 2>/dev/null | grep -q "GPU_SCHED_STATE=$HOME/.cache/hscho_share/gpu_sched_xhdp" && kill $p
  done
  sleep 2; cd /mnt/d/hscho_share
  GPU_SCHED_STATE=$HOME/.cache/hscho_share/gpu_sched_xhdp GPU_SCHED_SERVERS=$CFG setsid nohup python3 codes/_setup/gpu_sched/sched.py run \
    >> $HOME/.cache/hscho_share/gpu_sched_xhdp/daemon.log 2>&1 < /dev/null &
}
set_cap() {   # vast cap in measured.json + pending/running train jobs
  python3 - $1 <<'PY'
import json, sys, glob, os
cap = int(sys.argv[1]); st = os.path.expanduser("~/.cache/hscho_share/xhdp/measured.json")
m = json.load(open(st)); m["max_per_gpu"]["vast-isaac2"] = cap; json.dump(m, open(st, "w"), indent=1)
for p in glob.glob(os.path.expanduser("~/.cache/hscho_share/gpu_sched_xhdp/*/xhdp-train-*.json")):
    j = json.load(open(p)); j.setdefault("max_per_gpu", {})["vast-isaac2"] = cap; json.dump(j, open(p, "w"), indent=1)
PY
}
measure() { $V "python3 /workspace/codes/cross_dex/cross_dex/ours/crosshand_dp/measure.py --sec 600" | tail -n 1; }
until $V 'grep -q rc=0 /workspace/logs/env_lerobot.txt' 2>/dev/null; do
  $V 'grep -q "rc=[1-9]" /workspace/logs/env_lerobot.txt' 2>/dev/null && { log "lerobot env install FAILED on vast-isaac2"; exit 1; }
  sleep 60; done
$V '/workspace/envs/dexverse/lerobot/bin/python -c "import torch, lerobot; assert torch.cuda.is_available(); print(torch.__version__)"' || { log "lerobot env broken"; exit 1; }
log "lerobot env ready"
python3 - <<'PY'
import json; p="/mnt/d/hscho_share/codes/cross_dex/cross_dex/ours/crosshand_dp/sched_servers.json"
c=json.load(open(p)); c["servers"]["vast-isaac2"]["off"]=False; c["servers"]["vast-isaac2"]["off_reason"]="on since lerobot env ready (2026-11-20)"; json.dump(c,open(p,"w"),indent=1)
PY
set_cap 1; restart_sched
echo '["a6/ea", "a6/xlvla"]' > $ST/measure_runs.json; log "measurement trainings a6/ea, a6/xlvla queued (cap 1/GPU)"
ntrain() { $V 'pgrep -fc "^/workspace/envs/dexverse/lerobot/bin/python .*crosshand_dp/train_dp.py"' 2>/dev/null | tail -n 1 | tr -dc 0-9; }
until [ "$(ntrain)" -ge 2 ] 2>/dev/null; do sleep 60; done
sleep 600; M1=$(measure); echo "$M1" > $ST/measure_vast_1.json; log "measure 1: $M1"
python3 - "$M1" > $ST/cap_try.txt <<'PY'
import json, sys
m = json.loads(sys.argv[1]); j = m["jobs"]
cpu = max(x["cpu_cores"] for x in j); ram = max(x["rss_gb"] for x in j); util = min(x["gpu_util_mean"] for x in j)
n_cpu, n_ram = int(46 // max(cpu, 1)), int(259 * 0.8 // max(ram, 1))
print(2 if min(n_cpu, n_ram) >= 4 and util < 95 else 1)
PY
if [ "$(cat $ST/cap_try.txt)" = 2 ]; then
  set_cap 2; echo '["a6/ea", "a6/xlvla", "a6/joint", "b_noshadow/eh"]' > $ST/measure_runs.json
  log "trying 2 trainings per vast GPU"
  until [ "$(ntrain)" -ge 4 ] 2>/dev/null; do sleep 60; done
  sleep 600; M2=$(measure); echo "$M2" > $ST/measure_vast_2.json; log "measure 2: $M2"
  CAP=$(python3 - "$M1" "$M2" <<'PY'
import json, sys
a, b = (json.loads(x)["jobs"] for x in sys.argv[1:3])
r1 = sum(x["steps_per_s"] or 0 for x in a) / 2; r2 = sum(x["steps_per_s"] or 0 for x in b) / 2   # per GPU
print(2 if r2 >= 1.05 * r1 else 1)
PY
)
else CAP=1; fi
set_cap $CAP; log "vast-isaac2 training cap per GPU = $CAP"
python3 - "$CAP" <<'PY'
import json, sys, os, glob
st = os.path.expanduser("~/.cache/hscho_share/xhdp"); m = json.load(open(f"{st}/measured.json"))
m["vast-isaac2"] = {"measure_1_jobs_1_per_gpu": json.load(open(f"{st}/measure_vast_1.json")),
                    "measure_2_jobs_2_per_gpu": json.load(open(f"{st}/measure_vast_2.json")) if os.path.exists(f"{st}/measure_vast_2.json") else None,
                    "N": int(sys.argv[1]), "decision": "N = trainings per GPU whose per-GPU step rate is >= 5 % above N-1, bounded by 46 cores (cpu.max) and 0.8 x 259 GB RAM"}
json.dump(m, open(f"{st}/measured.json", "w"), indent=1)
PY
touch $ST/measured_done; log "all remaining trainings released"
until $V 'grep -q rc= /workspace/logs/env_main.txt' 2>/dev/null; do sleep 120; done
if $V 'grep -q rc=0 /workspace/logs/env_main.txt && source /workspace/codes/_setup/dexverse_env.sh DexVerse >/dev/null && python -c "import isaacsim, dexverse"'; then
  touch $ST/eval_ready_vast-isaac2; log "main env ready: vast-isaac2 takes evaluations"
else log "main env install FAILED on vast-isaac2: evaluations stay on p1"; fi
