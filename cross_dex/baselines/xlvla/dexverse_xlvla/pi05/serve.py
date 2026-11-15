"""pi0.5 inference process (openpi venv) for the DexVerse evaluator. Started by eval_policy.Pi05PolicyServer (DexLatent
venv, Isaac-side policy server) and talks to it over multiprocessing.connection on localhost:
  ("infer", {"image", "wrist_image": HWC uint8, "state": (D,), "prompt": str}) -> actions (H, action_dim) float32
  ("close", None) -> exit
The policy is openpi's own create_trained_policy with the training config of openpi_train.make_config (same transforms
and norm stats as training; norm stats are read from <ckpt>/assets/<repo_id>).

  python -m dexverse_xlvla.pi05.serve --ckpt <checkpoints/<run>/<exp>/<step>> --run <run> --repo_id <id> --action_dim 41
         --port <p> --authkey <k>
"""
import argparse
import os
import threading
import time
import pathlib
import sys
from multiprocessing.connection import Listener

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--action_dim", type=int, required=True)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--max_token_len", type=int, default=None, help="as in training (bimanual: 360 / 280)")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--authkey", required=True)
    a = ap.parse_args()
    parent = os.getppid()

    def orphan_guard():   # the Isaac-side server can die without "close" (evaluator crash / kill): do not keep the GPU
        while True:
            if os.getppid() != parent:
                os._exit(0)
            time.sleep(5)
    threading.Thread(target=orphan_guard, daemon=True).start()
    sys.path.insert(0, os.environ.get("OPENPI_ROOT", "/workspace/codes/dexverse/openpi"))
    from openpi.policies import policy_config
    from dexverse_xlvla.pi05.openpi_train import make_config

    cfg_args = argparse.Namespace(run=a.run, exp="eval", action_dim=a.action_dim, horizon=a.horizon, repo_id=a.repo_id, max_token_len=a.max_token_len,
                                  batch=1, accum=1, workers=0, warmup=1, lr=1e-4, steps=1, decay_lr=1e-6, ema=0, seed=42,
                                  log_interval=1, save_interval=1, overwrite=False, resume=False, fsdp=1)
    policy = policy_config.create_trained_policy(make_config(cfg_args), pathlib.Path(a.ckpt))
    print(f"[pi05-serve] loaded {a.ckpt}", flush=True)
    with Listener(("127.0.0.1", a.port), authkey=a.authkey.encode()) as listener, listener.accept() as conn:
        while True:
            kind, obs = conn.recv()
            if kind == "close":
                break
            conn.send(np.asarray(policy.infer(obs)["actions"], np.float32))


if __name__ == "__main__":
    main()
