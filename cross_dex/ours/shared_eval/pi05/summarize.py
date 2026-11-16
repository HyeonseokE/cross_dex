#!/usr/bin/env python
"""Summarise pi0.5 bench.sh / eval_shared.sh outputs (plain python3 + numpy).

  python summarize.py <out_dir> [...]                 one JSON line per run: sessions, episodes, steps, wall, steps/s, act /
                                                      chunk / sim-loop ms, forward batch sizes and ms, GPU memory, success
  python summarize.py --compare <ref_dir> <dir> [...] actions of every session s<i> vs the reference's s1 (and the
                                                      reference's own s2.. = old-vs-old repeatability): per episode max |diff|
                                                      of policy_out / action, first differing step, steps and success
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def pct(x, q):
    return round(float(np.percentile(x, q)), 1) if len(x) else None


def summarize(out):
    rows = []
    for path in sorted(glob.glob(f"{out}/s[0-9]*/timing.jsonl")):
        rows += [json.loads(line) for line in open(path)]
    chunk = [v for r in rows for v in r["chunk_ms"]]
    act = [v for r in rows for v in r["act_ms"]]
    loop = [v for r in rows for v in r["loop_ms"]]
    steps = sum(r["steps"] for r in rows)
    start = min((r["start"] for r in rows), default=None)
    end = max((r["start"] + r.get("wall_s", 0) for r in rows), default=None)
    success = []
    for path in sorted(glob.glob(f"{out}/s[0-9]*/eval/metrics.json")):
        success += [e["success"] for e in json.load(open(path))["episodes"]]
    summary = {"run": out, "sessions": len(glob.glob(f"{out}/s[0-9]*/")), "episodes": len(rows), "steps": steps,
               "wall_s": round(end - start, 1) if rows else None,
               "steps_per_s": round(steps / (end - start), 2) if rows and end > start else None,
               "chunk_ms_p50": pct(chunk, 50), "chunk_ms_p90": pct(chunk, 90), "act_ms_p50": pct(act, 50),
               "loop_ms_p50": pct(loop, 50), "success": f"{sum(success)}/{len(success)}"}
    server = f"{out}/server.jsonl"
    if os.path.exists(server):
        forwards = defaultdict(list)
        for line in open(server):
            row = json.loads(line)
            if row["kind"] == "forward":
                forwards[row["batch"]].append(row["ms"])
            elif row["kind"] == "memory":
                summary["server_memory_mb"] = {k: round(v) for k, v in row.items() if k != "kind"}
        summary["forwards_by_batch"] = {b: {"n": len(v), "ms_p50": pct(v, 50)} for b, v in sorted(forwards.items())}
    if os.path.exists(f"{out}/gpu.csv"):
        mem = [float(line.split(",")[1]) for line in open(f"{out}/gpu.csv") if line.count(",") >= 4]
        util = [float(line.split(",")[2]) for line in open(f"{out}/gpu.csv") if line.count(",") >= 4]
        summary["gpu_mem_max_mb"] = max(mem) if mem else None
        summary["gpu_util_mean"] = round(float(np.mean(util)), 1) if util else None
    return summary


def episodes(session):
    out = {}
    for path in sorted(glob.glob(f"{session}/actions/ep*.npz")):
        out[os.path.basename(path)] = np.load(path)
    metrics = f"{session}/eval/metrics.json"
    result = {e["episode"]: (e["success"], e["steps"]) for e in json.load(open(metrics))["episodes"]} if os.path.exists(metrics) else {}
    return out, result


def compare(a, b):
    """session b vs session a: per episode file"""
    ea, ra = episodes(a)
    eb, rb = episodes(b)
    rows = []
    for name in sorted(set(ea) & set(eb)):
        x, y = ea[name], eb[name]
        row = {"ep": name}
        for key in ("policy_out", "action"):
            n = min(len(x[key]), len(y[key]))
            d = np.abs(x[key][:n] - y[key][:n]).reshape(n, -1).max(axis=1) if n else np.zeros(0)
            row[f"{key}_max_abs"] = float(d.max()) if n else None
            row[f"{key}_first_diff_step"] = int(np.argmax(d > 0)) if (d > 0).any() else None
            row["len"] = [len(x[key]), len(y[key])]
        rows.append(row)
    return {"a": a, "b": b, "episodes": rows, "results_a": ra, "results_b": rb, "same_results": ra == rb}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("dirs", nargs="+")
    a = ap.parse_args()
    if not a.compare:
        for out in a.dirs:
            print(json.dumps(summarize(out)))
        return
    ref, others = a.dirs[0], a.dirs[1:]
    ref_s1 = f"{ref}/s1"
    for session in sorted(glob.glob(f"{ref}/s[0-9]*"))[1:]:
        print(json.dumps(compare(ref_s1, session)))
    for out in others:
        for session in sorted(glob.glob(f"{out}/s[0-9]*")):
            print(json.dumps(compare(ref_s1, session)))


if __name__ == "__main__":
    main()
