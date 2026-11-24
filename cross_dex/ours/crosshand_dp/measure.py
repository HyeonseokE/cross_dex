#!/usr/bin/env python3
"""Per-job resource use of running multi-hand DP trainings (EXPERIMENT_RULES 7.1 사전 측정), stable phase.
Samples every 10 s for --sec: per training (process tree = trainer + data loader workers) RSS GB and CPU cores,
GPU memory of the trainer (nvidia-smi compute apps), GPU utilisation, load average, and the step rate from the LeRobot log.

  python3 measure.py --sec 300 [--match crosshand_dp/train_dp.py]      (in the container) -> JSON on stdout
"""
import argparse
import glob
import json
import os
import re
import subprocess
import time

ap = argparse.ArgumentParser()
ap.add_argument("--sec", type=int, default=300)
ap.add_argument("--match", default="crosshand_dp/train_dp.py")
a = ap.parse_args()
HZ = os.sysconf("SC_CLK_TCK")


def procs():
    out = {}
    for p in glob.glob("/proc/[0-9]*"):
        try:
            st = open(f"{p}/stat").read().rsplit(")", 1)[1].split()
            out[int(p[6:])] = {"ppid": int(st[1]), "cpu": int(st[11]) + int(st[12]),
                               "rss": int(open(f"{p}/statm").read().split()[1]) * 4096,
                               "cmd": open(f"{p}/cmdline").read().replace("\0", " ")}
        except Exception:
            pass
    return out


def tree(ps, root):
    kids, todo = {root}, [root]
    while todo:
        x = todo.pop()
        for p, v in ps.items():
            if v["ppid"] == x and p not in kids:
                kids.add(p)
                todo.append(p)
    return kids


def gpu():
    q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    mem = {int(l.split(",")[0]): int(l.split(",")[1]) for l in q.splitlines() if l.strip()}
    u = subprocess.run(["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return mem, {int(l.split(",")[0]): (int(l.split(",")[1]), int(l.split(",")[2])) for l in u.splitlines()}


def steps(log):
    """Number of LeRobot log lines so far (one every 200 steps, --log_freq=200) x 200."""
    return 200 * sum(1 for line in open(log, errors="ignore") if " step:" in line and " loss:" in line)


ps = procs()
roots = [p for p, v in ps.items() if a.match in v["cmd"] and "train_dp.py" in v["cmd"] and ps.get(v["ppid"], {}).get("cmd", "").find("train_dp.py") < 0]
info = {}
for r in roots:
    env = open(f"/proc/{r}/environ").read().split("\0")
    g = next((e.split("=")[1] for e in env if e.startswith("CUDA_VISIBLE_DEVICES=")), "?")
    out_dir = re.search(r"--output_dir (\S+)", ps[r]["cmd"]).group(1)
    log = glob.glob(f"/workspace/experiments/*/RSS2027/comparison/dexverse-crosshand-dp/*/{os.path.basename(out_dir)}/logs/train.log")
    nw = re.search(r"--num_workers (\d+)", ps[r]["cmd"])
    info[r] = {"gpu": g, "run": os.path.basename(out_dir), "log": log[0] if log else None, "workers": int(nw.group(1)) if nw else None,
               "rss": [], "cpu0": sum(ps[p]["cpu"] for p in tree(ps, r)), "vram": [], "util": [], "s0": steps(log[0]) if log else None}
t0, loads = time.time(), []
while time.time() - t0 < a.sec:
    time.sleep(10)
    ps = procs()
    mem, gu = gpu()
    loads.append(float(open("/proc/loadavg").read().split()[0]))
    for r, d in info.items():
        t = tree(ps, r)
        d["rss"].append(sum(ps[p]["rss"] for p in t if p in ps) / 2**30)
        d["vram"].append(sum(mem.get(p, 0) for p in t) / 1024)
        if d["gpu"].isdigit() and int(d["gpu"]) in gu:
            d["util"].append(gu[int(d["gpu"])][0])
dt = time.time() - t0
ps = procs()
res = {"sec": round(dt), "load_mean": round(sum(loads) / len(loads), 1), "cores": os.cpu_count(),
       "cpu_max": open("/sys/fs/cgroup/cpu.max").read().strip() if os.path.exists("/sys/fs/cgroup/cpu.max") else None,
       "jobs": []}
for r, d in info.items():
    cpu1 = sum(ps[p]["cpu"] for p in tree(ps, r) if p in ps)
    s1 = steps(d["log"]) if d["log"] else None
    res["jobs"].append({"run": d["run"], "gpu": d["gpu"], "workers": d["workers"],
                        "rss_gb": round(max(d["rss"]), 1), "vram_gb": round(max(d["vram"]), 1),
                        "gpu_util_mean": round(sum(d["util"]) / max(len(d["util"]), 1)),
                        "cpu_cores": round((cpu1 - d["cpu0"]) / HZ / dt, 1),
                        "steps_per_s": round((s1 - d["s0"]) / dt, 2) if s1 is not None and d["s0"] is not None else None})
print(json.dumps(res))
