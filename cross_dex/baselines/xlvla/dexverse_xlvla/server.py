#!/usr/bin/env python
"""Policy server for the unchanged common evaluator (eval_lerobot.py), started in its place by run_eval.py.
Same protocol as teleop_dp_image/policy_server.py: ("reset", None) -> "ok"; ("act", obs) -> float32 action bytes.

--ckpt points at a folder with server.json:
  {"mode": "replay", "actions": <npz>, "episodes": [...]}   open-loop replay (step 4); episodes in evaluator order
  {"mode": "latent_dp", ...}                                 latent Diffusion Policy (step 5, dexverse_xlvla.policy)
  {"mode": "pi05_latent" | "pi05_joint", "bimanual": true, ...}  bimanual pi0.5 (dexverse_xlvla.pi05.eval_bimanual)
"""
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
        a = self.seq[self.ep][min(self.t, len(self.seq[self.ep]) - 1)]   # hold the last action after the demo ends
        self.t += 1
        return a


if cfg["mode"] == "replay":
    policy = Replay(cfg)
elif cfg.get("bimanual"):   # bimanual pi0.5 (pi05/eval_bimanual.py)
    from dexverse_xlvla.pi05.eval_bimanual import Pi05BimanualPolicyServer
    policy = Pi05BimanualPolicyServer(cfg)
elif cfg["mode"] in ("pi05_latent", "pi05_joint", "pi05_lad"):
    from dexverse_xlvla.pi05.eval_policy import Pi05PolicyServer
    policy = Pi05PolicyServer(cfg)
else:
    from dexverse_xlvla.policy import LatentPolicyServer
    policy = LatentPolicyServer(cfg)

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
