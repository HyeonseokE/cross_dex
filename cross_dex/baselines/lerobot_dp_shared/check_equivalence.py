#!/usr/bin/env python
"""Offline checks of shared_server.py on observations recorded in the sim (run_eval_dp.py --save_obs K); lerobot env.

A  B=1, cuda generator: shared_server actions == LeRobot DiffusionPolicy.select_action after torch.cuda.manual_seed(seed)
   (same queues, same draws) -> the reimplemented sampling is LeRobot's.
B  cpu generators: one session alone vs. three sessions batched together (two with the same seed, one other seed)
   -> a session's actions do not depend on who shares its batch.
C  LeRobot generate_actions wall time for batch 1..5 (one model call = one 8-action chunk).

  python check_equivalence.py --ckpt <pretrained_model> --obs <timing>/obs.npz --out <json>
"""
import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import threading
import time
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--ckpt", required=True)
parser.add_argument("--obs", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--dp_dir", default=os.path.join(os.environ.get("DEXVERSE_ROOT", "/workspace/codes/dexverse/DexVerse"),
                                                     "baselines", "lerobot_dp"))
args = parser.parse_args()
SERVER = Path(__file__).resolve().parent / "shared_server.py"

raw = np.load(args.obs)
count = 1 + max(int(k.split("|")[0]) for k in raw.files)
OBS = [{k.split("|", 1)[1]: raw[k] for k in raw.files if k.startswith(f"{i}|")} for i in range(count)]
MESSAGES = [{k: (v.shape, str(v.dtype), v.tobytes()) for k, v in m.items()} for m in OBS]


def start(sessions, generator_device, log):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    key = secrets.token_hex(16)
    server = subprocess.Popen([sys.executable, str(SERVER), "--ckpt", args.ckpt, "--port", str(port), "--authkey", key,
                               "--sessions", str(sessions), "--generator_device", generator_device, "--gather_ms", "50",
                               "--log", log, "--dp_dir", args.dp_dir])
    return server, port, key


def connect(port, key):
    for _ in range(120):
        try:
            return Client(("127.0.0.1", port), authkey=key.encode())
        except ConnectionRefusedError:
            time.sleep(1)
    raise RuntimeError("server did not start")


def session(port, key, seed, out, index, barrier=None):
    conn = connect(port, key)
    if barrier:
        barrier.wait()
    conn.send(("reset", {"seed": seed}))
    conn.recv()
    actions = []
    for message in MESSAGES:
        conn.send(("act", message))
        actions.append(np.frombuffer(conn.recv(), dtype=np.float32).copy())
    conn.send(("close", None))
    conn.close()
    out[index] = np.stack(actions)


def serve_sessions(seeds, generator_device, log):
    server, port, key = start(len(seeds), generator_device, log)
    out, barrier = [None] * len(seeds), threading.Barrier(len(seeds))
    threads = [threading.Thread(target=session, args=(port, key, s, out, i, barrier)) for i, s in enumerate(seeds)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    server.wait(timeout=60)
    return out


def diff(a, b):
    return {"max_abs": float(np.abs(a - b).max()), "identical": bool((a == b).all())}


results = {"obs": args.obs, "calls": len(MESSAGES)}
log_dir = Path(args.out).parent

# A: shared_server (B=1, cuda generator) vs LeRobot's own select_action
served = serve_sessions([args.seed], "cuda", str(log_dir / "eq_A.jsonl"))[0]
sys.path.insert(0, args.dp_dir)
from image_resize import install_resize_for  # noqa: E402

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402

policy = DiffusionPolicy.from_pretrained(args.ckpt).to("cuda").eval()
install_resize_for(policy.config)
pre, post = make_pre_post_processors(policy.config, pretrained_path=args.ckpt,
                                     preprocessor_overrides={"device_processor": {"device": "cuda"}})


def to_batch(obs):
    batch = {}
    for key, value in obs.items():
        value = torch.from_numpy(value.copy())
        batch[key] = value.permute(2, 0, 1).float() / 255.0 if value.ndim == 3 else value.float()
    return batch


policy.reset()
torch.cuda.manual_seed(args.seed)
reference = []
with torch.inference_mode():
    for obs in OBS:
        reference.append(post(policy.select_action(pre(to_batch(obs))))[0].cpu().numpy().astype(np.float32))
results["A_shared_b1_vs_lerobot_select_action"] = diff(served, np.stack(reference))

# B: one session alone vs. three batched (seeds s, s, s+1), cpu generators
alone = serve_sessions([args.seed], "cpu", str(log_dir / "eq_B1.jsonl"))[0]
log_b3 = log_dir / "eq_B3.jsonl"
three = serve_sessions([args.seed, args.seed, args.seed + 1], "cpu", str(log_b3))
batches = [json.loads(line)["batch"] for line in open(log_b3) if '"forward"' in line]
results["B_batch_sizes"] = batches
results["B_alone_vs_batched_same_seed"] = [diff(alone, three[0]), diff(alone, three[1])]
results["B_alone_vs_other_seed"] = diff(alone, three[2])
results["B_vs_A_cpu_vs_cuda_noise"] = diff(alone, served)

# C: model call time by batch size (LeRobot generate_actions, 100 DDPM steps)
with torch.inference_mode():
    batch = dict(pre(to_batch(OBS[0])))
    images = torch.stack([batch[k] for k in policy.config.image_features], dim=-4)
    single = {"observation.state": torch.stack([batch["observation.state"]] * policy.config.n_obs_steps, dim=1),
              "observation.images": torch.stack([images] * policy.config.n_obs_steps, dim=1)}
    timing = {}
    for b in (1, 2, 3, 4, 5):
        many = {k: v.repeat(b, *[1] * (v.ndim - 1)) for k, v in single.items()}
        policy.diffusion.generate_actions(many)
        times = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            policy.diffusion.generate_actions(many)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1e3)
        timing[b] = {"ms": round(float(np.median(times)), 1), "max_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20)}
    results["C_generate_actions_ms_by_batch"] = timing
Path(args.out).write_text(json.dumps(results, indent=1))
print(json.dumps(results, indent=1))
