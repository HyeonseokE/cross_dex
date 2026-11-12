#!/usr/bin/env python
"""Policy server for the unchanged common evaluator, started in its place by dp_run_eval.py (same protocol as
dexverse_xlvla/server.py): ("reset", None) -> "ok"; ("act", obs) -> float32 action bytes; ("close", None).
--ckpt points at a folder with server.json: {"mode": "chunk_latent_dp", ...} (dp_policy.py) or
{"mode": "replay", "actions": <npz>, "episodes": [...]} (open-loop replay of decoded actions, check ④)."""
import argparse
import json
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--authkey", required=True)
args = parser.parse_args()
cfg = json.loads((Path(args.ckpt) / "server.json").read_text())


class Replay:
    def __init__(self, c):
        data = np.load(c["actions"])
        self.seq = [data[str(k)] for k in c["episodes"]]
        self.ep, self.t = -1, 0

    def reset(self):
        self.ep, self.t = self.ep + 1, 0

    def act(self, _obs):
        a = self.seq[self.ep][min(self.t, len(self.seq[self.ep]) - 1)]
        self.t += 1
        return a


if cfg["mode"] == "replay":
    policy = Replay(cfg)
else:
    from cross_dex.latent.chunk_latent.dp_policy import ChunkLatentPolicyServer
    policy = ChunkLatentPolicyServer(cfg)

with Listener(("127.0.0.1", args.port), authkey=args.authkey.encode()) as listener, listener.accept() as conn:
    while True:
        kind, message = conn.recv()
        if kind == "close":
            getattr(policy, "close", lambda: None)()
            break
        if kind == "reset":
            policy.reset()
            conn.send("ok")
            continue
        conn.send(np.asarray(policy.act(message), dtype=np.float32).tobytes())
