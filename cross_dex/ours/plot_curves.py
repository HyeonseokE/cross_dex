"""Loss curves of a latent-space training run (2026-11-19, user: record the loss curve of every training).

Reads <run_dir>/metrics.csv (long format step,split,metric,value, written by dexverse_xlvla.train_latent and the ours
trainers) or <run_dir>/logs/train_log.jsonl (dexverse_lad.train_cam) and writes <run_dir>/artifacts/loss_curves.png:
one small panel per loss term (log y for positive terms), moving average over --smooth logged points.

  python -m cross_dex.ours.plot_curves <run_dir> [<run_dir> ...] [--smooth 20]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict

SKIP = ("pair_hand_pairs", "pair_refresh_sec", "exp_dis", "sr_w_cyc", "v2_lam", "v2_w", "stage", "pair_tau", "epoch", "sec",
        "loss_rec_hand")


def load(run):
    series = defaultdict(list)
    p = os.path.join(run, "metrics.csv")
    if os.path.exists(p):
        with open(p) as f:
            for r in csv.DictReader(f):
                if r.get("split", "train") != "train":                # eval rows = monitor checks, not losses
                    continue
                try:
                    series[r["metric"]].append((float(r["step"]), float(r["value"])))
                except (ValueError, KeyError):
                    pass
        return series
    p = os.path.join(run, "logs", "train_log.jsonl")
    if os.path.exists(p):                                          # LAD: one row per evaluation, x = cumulative epoch
        off, last_stage, last_ep = 0, None, 0
        for line in open(p):
            d = json.loads(line)
            if d.get("stage") != last_stage and last_stage is not None:
                off += last_ep
            last_stage, last_ep = d.get("stage"), d.get("epoch", 0)
            for k, v in d.items():
                if isinstance(v, (int, float)) and k not in SKIP:
                    series[f"s{d.get('stage')}_{k}"].append((off + d.get("epoch", 0), float(v)))
    return series


def smooth(v, n):
    if n <= 1 or len(v) <= n:
        return v
    out, s = [], 0.0
    for i, x in enumerate(v):
        s += x
        if i >= n:
            s -= v[i - n]
        out.append(s / min(i + 1, n))
    return out


def plot(run, n):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    series = {k: v for k, v in load(run).items() if k not in SKIP and len(v) > 1}
    if not series:
        print(f"[curves] {run}: no metrics")
        return
    keys = sorted(series)
    cols = 4
    rows = math.ceil(len(keys) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 2.6 * rows), squeeze=False)
    for ax, k in zip(axes.flat, keys):
        xs, ys = zip(*series[k])
        ax.plot(xs, ys, color="#c9d3df", lw=0.8)
        ax.plot(xs, smooth(list(ys), n), color="#2a78d6", lw=1.6)
        if all(y > 0 for y in ys) and max(ys) / max(min(ys), 1e-12) > 20:
            ax.set_yscale("log")
        ax.set_title(k, fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in list(axes.flat)[len(keys):]:
        ax.axis("off")
    fig.suptitle(os.path.basename(os.path.normpath(run)), fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.join(run, "artifacts"), exist_ok=True)
    out = os.path.join(run, "artifacts", "loss_curves.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"[curves] {out} ({len(keys)} terms)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--smooth", type=int, default=20)
    a = ap.parse_args()
    for r in a.runs:
        plot(r, a.smooth)


if __name__ == "__main__":
    main()
