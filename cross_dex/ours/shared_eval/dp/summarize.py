#!/usr/bin/env python
"""Summarise bench.sh outputs: per-call latency, sim clock while waiting, throughput, VRAM; compare actions of runs.

  python summarize.py <out_dir> [<out_dir> ...]           one JSON line per run
  python summarize.py --compare <out_dir_a> <out_dir_b>   action differences per session/episode (same episodes, seeds)
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else None


def summarize(out):
    sessions = sorted(glob.glob(f"{out}/s[0-9]*/timing"))
    chunk_ms, other_ms, step_ms, loop_ms, steps, walls, clock_ok, episodes = [], [], [], [], 0, [], True, []
    for timing in sessions:
        metrics = json.load(open(f"{os.path.dirname(timing)}/metrics.json"))
        success = {e["episode"]: e["success"] for e in metrics["episodes"]}
        for path in sorted(glob.glob(f"{timing}/ep*.npz")):
            d = np.load(path)
            chunk = d["chunk"].astype(bool)
            chunk_ms += list(d["call_ms"][chunk])
            other_ms += list(d["call_ms"][~chunk])
            step_ms += list(d["step_ms"])
            loop_ms += list(d["loop_ms"])
            steps += len(d["actions"])
            walls.append(float(d["wall_s"]))
            clock_ok &= bool((d["clock_before"] == d["clock_after"]).all())
            ep = int(os.path.basename(path)[2:5])
            episodes.append({"session": timing.split("/")[-2], "episode": ep, "steps": len(d["actions"]),
                             "success": success.get(ep), "wall_s": round(float(d["wall_s"]), 1)})
    # concurrent window: barrier release (first reset) to the last episode end, from the session logs' mtimes
    starts, ends = [], []
    for log in glob.glob(f"{out}/s[0-9]*.log"):
        text = open(log, errors="replace").read()
        for line in text.splitlines():
            if line.startswith("[timing] barrier passed at"):
                starts.append(float(line.split()[-1]))
        ends.append(max(os.path.getmtime(p) for p in glob.glob(f"{out}/{os.path.basename(log)[:-4]}/timing/ep*.npz")))
    window = max(ends) - min(starts) if starts and ends else None
    forwards = defaultdict(list)
    memory = None
    for path in glob.glob(f"{out}/server*.jsonl"):
        for line in open(path):
            row = json.loads(line)
            if row["kind"] == "forward":
                forwards[row["batch"]].append(row["ms"])
            elif row["kind"] == "memory":
                memory = row
    gpu = np.loadtxt(f"{out}/gpu.csv", delimiter=",", ndmin=2) if os.path.exists(f"{out}/gpu.csv") else None
    return {
        "run": os.path.basename(out.rstrip("/")), "sessions": len(sessions), "episodes": len(episodes),
        "steps": steps,
        "chunk_call_ms": {"mean": float(np.mean(chunk_ms)), "p50": pct(chunk_ms, 50), "p95": pct(chunk_ms, 95),
                          "max": float(np.max(chunk_ms)), "n": len(chunk_ms)},
        "other_call_ms": {"mean": float(np.mean(other_ms)), "p50": pct(other_ms, 50), "p95": pct(other_ms, 95)},
        "step_ms_mean": float(np.mean(step_ms)), "loop_ms_mean": float(np.mean(loop_ms)) if loop_ms else None,
        "sim_clock_unchanged_while_waiting": clock_ok,
        "window_s": round(window, 1) if window else None,
        "steps_per_s_total": round(steps / window, 2) if window else None,
        "episodes_per_h_at_400": round(steps / window * 3600 / 400, 1) if window else None,
        "forward_ms_by_batch": {b: {"n": len(v), "mean": round(float(np.mean(v)), 1)} for b, v in sorted(forwards.items())},
        "server_torch_max_reserved_mb": memory and round(memory["max_reserved_mb"]),
        "gpu_mem_used_mb_max": int(gpu[:, 1].max()) if gpu is not None else None,
        "ram_used_mb_max": int(gpu[:, 3].max()) if gpu is not None else None,
        "load_max": float(gpu[:, 4].max()) if gpu is not None else None,
        "per_episode": episodes,
    }


def compare(a, b):
    rows = []
    for path in sorted(glob.glob(f"{a}/s[0-9]*/timing/ep*.npz")):
        rel = os.path.relpath(path, a)
        other = os.path.join(b, rel)
        if not os.path.exists(other):
            other_list = sorted(glob.glob(os.path.join(b, "s[0-9]*", "timing", os.path.basename(path))))
            if not other_list:
                continue
            other = other_list[0]
        x, y = np.load(path)["actions"], np.load(other)["actions"]
        n = min(len(x), len(y))
        diff = np.abs(x[:n] - y[:n]).max(axis=1)
        first = int(np.argmax(diff > 1e-5)) if (diff > 1e-5).any() else None
        da, db = np.load(path), np.load(other)
        if "img_crc" in da and "img_crc" in db:
            same = da["img_crc"][:n] == db["img_crc"][:n]
            obs_first = int(np.argmin(same)) if not same.all() else None
            state_diff = float(np.abs(da["state"][:n] - db["state"][:n]).max())
        else:
            obs_first, state_diff = "n/a", "n/a"
        rows.append({"a": rel, "b": os.path.relpath(other, b), "steps": [len(x), len(y)],
                     "max_abs_diff_first_chunk": float(diff[:8].max()), "max_abs_diff": float(diff.max()),
                     "first_step_diff_gt_1e-5": first,
                     "first_step_images_differ": obs_first, "state_max_abs_diff": state_diff, "identical": bool((diff == 0).all() and len(x) == len(y))})
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--brief", action="store_true", help="Leave out per-episode rows.")
    args = parser.parse_args()
    if args.compare:
        for row in compare(*args.dirs):
            print(json.dumps(row))
    else:
        for out in args.dirs:
            summary = summarize(out)
            if args.brief:
                summary.pop("per_episode")
            print(json.dumps(summary))
