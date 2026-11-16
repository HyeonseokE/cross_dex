#!/usr/bin/env python
"""Offline GPU checks of the batched pi0.5 path (openpi venv, no Isaac Sim); the pi0.5 counterpart of dp/check_equivalence.py.

A  B=1: BatchedPolicy.infer_batch with a process-mode SessionKey == openpi Policy.infer of a fresh policy (its default
   key(0)), call after call -> the shared server reproduces the baseline openpi process (pi05/serve.py) exactly.
B  three sessions batched (each its own observation stream, all process-mode keys) vs each stream alone at B=1
   -> how much the batched arithmetic changes a session's chunk (noise is the same by construction).
C  forward wall time and peak device memory for B = 1..--max_b.

Observations: --obs <s1/obs.npz> (bench.sh SAVE_OBS, real infer inputs) or synthetic ones (random images, state of the
norm-stats dimension, the task's training prompt).

  cd <DexLatent> && PYTHONPATH=$PWD XLA_PYTHON_CLIENT_PREALLOCATE=false python check_batch.py --server_json <cell>/server.json \
      [--obs <obs.npz>] --out <json>
"""
import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("--server_json", required=True, help="a cell's server.json (checkpoint, run, repo_id, action_dim, ...)")
ap.add_argument("--obs", default=None)
ap.add_argument("--calls", type=int, default=6, help="sequential calls per stream (A, B)")
ap.add_argument("--max_b", type=int, default=6)
ap.add_argument("--out", required=True)
a = ap.parse_args()
c = json.loads(pathlib.Path(a.server_json).read_text())
sys.path.insert(0, os.environ.get("OPENPI_ROOT", "/workspace/codes/dexverse/openpi"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from openpi.models import model as _model  # noqa: E402

from batched_policy import BatchedPolicy, SessionKey  # noqa: E402
from openpi_server import build_parser, load_policy  # noqa: E402

args = build_parser().parse_args(["--ckpt", c["policy_ckpt"], "--run", c["run"], "--repo_id", c["repo_id"], "--action_dim",
                                  str(c["action_dim"]), "--port", "1", "--authkey", "x"]
                                 + (["--max_token_len", str(c["max_token_len"])] if c.get("max_token_len") else []))


def observations(count, offset=0):
    if a.obs:
        raw = np.load(a.obs)
        n = 1 + max(int(k.split("|")[0]) for k in raw.files)
        rows = [{k.split("|", 1)[1]: raw[k] for k in raw.files if k.startswith(f"{i}|")} for i in range(n)]
        for r in rows:
            r["prompt"] = str(r["prompt"])
        return [rows[(i + offset) % n] for i in range(count)]
    from openpi.training import checkpoints as _ck
    from dexverse_xlvla.pi05.v3_dataset import PROMPTS
    dim = _ck.load_norm_stats(pathlib.Path(c["policy_ckpt"]) / "assets", c["repo_id"])["state"].mean.shape[-1]
    rng = np.random.default_rng(1000 + offset)
    keys = ["image", "wrist_image"] + (["left_wrist_image"] if c.get("bimanual") else [])
    return [{**{k: rng.integers(0, 256, (240, 320, 3), dtype=np.uint8) for k in keys},
             "state": rng.uniform(-1, 1, dim).astype(np.float32), "prompt": PROMPTS[c["task"]]} for _ in range(count)]


def diff(x, y):
    return {"max_abs": float(np.abs(x - y).max()), "identical": bool((x == y).all())}


t0 = time.time()
policy = load_policy(args)
results = {"server_json": a.server_json, "obs": a.obs or "synthetic", "load_s": round(time.time() - t0, 1)}
batched = BatchedPolicy(policy, _model.Observation.from_dict, jax, jnp)

# A: B=1 shared path vs Policy.infer (policy._rng starts at key(0) and is only advanced by infer)
streams = [observations(a.calls, offset=k) for k in range(3)]
reference = np.stack([np.asarray(policy.infer(o)["actions"]) for o in streams[0]])
key = SessionKey(jax, "process")
ours = np.stack([batched.infer_batch([o], [key.next()])[0]["actions"] for o in streams[0]])
results["A_b1_vs_policy_infer"] = diff(ours, reference)

# B: three streams batched vs each alone
alone = []
for s in streams:
    k = SessionKey(jax, "process")
    alone.append(np.stack([batched.infer_batch([o], [k.next()])[0]["actions"] for o in s]))
keys = [SessionKey(jax, "process") for _ in streams]
together = [[] for _ in streams]
for call in range(a.calls):
    outs = batched.infer_batch([s[call] for s in streams], [k.next() for k in keys])
    for i, o in enumerate(outs):
        together[i].append(o["actions"])
results["B_batched3_vs_alone"] = [diff(np.stack(t), al) for t, al in zip(together, alone)]
results["B_alone_stream0_vs_A"] = diff(alone[0], ours)

# C: forward time by batch size
timing = {}
zero = jax.random.key(0)
for b in range(1, a.max_b + 1):
    obs = observations(b)
    batched.infer_batch(obs, [zero] * b)  # compile
    times = []
    for _ in range(3):
        t = time.perf_counter()
        batched.infer_batch(obs, [zero] * b)
        times.append((time.perf_counter() - t) * 1e3)
    stats = jax.devices()[0].memory_stats() or {}
    timing[b] = {"ms": round(float(np.median(times)), 1), "peak_mb": round(stats.get("peak_bytes_in_use", 0) / 2**20)}
results["C_forward_ms_by_batch"] = timing
pathlib.Path(a.out).write_text(json.dumps(results, indent=1))
print(json.dumps(results, indent=1))
