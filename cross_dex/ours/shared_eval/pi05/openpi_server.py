#!/usr/bin/env python
"""One pi0.5 (openpi, JAX) model shared by several evaluation sessions on one GPU (openpi venv).

Replaces the per-cell openpi process of the baseline path (dexverse_xlvla/pi05/serve.py, one ~8.8 GB process per cell).
Same request protocol as serve.py, so the baseline's own Isaac-side policy classes (Pi05PolicyServer /
Pi05BimanualPolicyServer, run by session_server.py) talk to it unchanged:
  ("infer", {"image", "wrist_image"[, "left_wrist_image"]: HWC uint8, "state": (D,), "prompt": str}) -> (H, action_dim) float32
plus ("reset", {"episode": int}) -> "ok" (sent by session_server.py at every episode; used by --noise episode) and
("close", None). Every infer needs the model, so all infer requests are gathered (../batch_server.py) and run as one
batched sample_actions (batched_policy.py), each row with its own session's noise key.

The model is built exactly as serve.py builds it (openpi create_trained_policy with dexverse_xlvla.pi05.openpi_train.make_config,
same arguments); serve.py itself is not imported because it builds and serves in one function.

  PYTHONPATH=<DexLatent> XLA_PYTHON_CLIENT_PREALLOCATE=false python openpi_server.py --ckpt <step dir> --run <run> \
      --repo_id <id> --action_dim 41 [--max_token_len 360] --port <p> --authkey <k> [--gather_ms 1000] [--log server.jsonl]
"""
import argparse
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # ours/shared_eval (batch_server)
sys.path.insert(0, str(HERE))
from batch_server import BatchServer, add_server_args  # noqa: E402


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--repo_id", required=True)
    ap.add_argument("--action_dim", type=int, required=True)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--max_token_len", type=int, default=None, help="as in training (bimanual: 360 / 280)")
    ap.add_argument("--noise", choices=("process", "episode"), default="process",
                    help="process: each session's noise sequence = that of a fresh baseline serve.py process (key(0), split "
                         "per chunk); episode: key(seed*100000 + episode) at every episode reset.")
    ap.add_argument("--seed", type=int, default=0, help="--noise episode base seed.")
    ap.add_argument("--precompile", type=int, default=0,
                    help="Compile batch sizes 1..K on the first request (copies of it, zero noise; no session key used), "
                         "so later batch sizes do not stall all sessions with a compile.")
    add_server_args(ap, gather_ms=1000.0, max_batch=8)
    return ap


def load_policy(a):
    """= dexverse_xlvla.pi05.serve.main up to the policy (same TrainConfig arguments)."""
    from openpi.policies import policy_config
    from dexverse_xlvla.pi05.openpi_train import make_config

    cfg_args = argparse.Namespace(run=a.run, exp="eval", action_dim=a.action_dim, horizon=a.horizon, repo_id=a.repo_id,
                                  max_token_len=a.max_token_len, batch=1, accum=1, workers=0, warmup=1, lr=1e-4, steps=1,
                                  decay_lr=1e-6, ema=0, seed=42, log_interval=1, save_interval=1, overwrite=False,
                                  resume=False, fsdp=1)
    return policy_config.create_trained_policy(make_config(cfg_args), pathlib.Path(a.ckpt))


class Pi05Backend:
    def __init__(self, batched, jax, noise="process", seed=0, precompile=0, say=print):
        from batched_policy import SessionKey
        self.batched, self.jax, self.noise, self.seed, self.precompile, self.say = batched, jax, noise, seed, precompile, say
        self.SessionKey = SessionKey

    def opened(self, s):
        s.key, s.obs, s.actions = self.SessionKey(self.jax, self.noise, self.seed), None, None

    def reset(self, s, message):
        s.key.reset(message.get("episode") if isinstance(message, dict) else None)
        return "ok"

    def observe(self, s, message):
        s.obs = message
        return True

    def _precompile(self, obs):
        zero = self.jax.random.key(0)
        for b in range(1, self.precompile + 1):
            t0 = time.perf_counter()
            self.batched.infer_batch([obs] * b, [zero] * b)
            self.say(f"precompiled batch {b}: {time.perf_counter() - t0:.1f} s")
        self.precompile = 0

    def forward(self, sessions):
        if self.precompile:
            self._precompile(sessions[0].obs)
        outputs = self.batched.infer_batch([s.obs for s in sessions], [s.key.next() for s in sessions])
        for s, out in zip(sessions, outputs):
            s.actions, s.obs = out["actions"], None

    def reply(self, s):
        return np.asarray(s.actions, np.float32)  # = serve.py

    def synchronize(self):  # infer_batch ends in np.asarray (blocks)
        pass

    def memory(self):
        try:
            stats = self.jax.devices()[0].memory_stats() or {}
        except Exception:
            stats = {}
        return {k: stats[k] / 2**20 for k in ("peak_bytes_in_use", "bytes_in_use", "bytes_limit") if k in stats}


def main():
    a = build_parser().parse_args()
    sys.path.insert(0, os.environ.get("OPENPI_ROOT", "/workspace/codes/dexverse/openpi"))  # as serve.py
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model
    from batched_policy import BatchedPolicy

    t0 = time.time()
    policy = load_policy(a)
    print(f"[pi05-shared] loaded {a.ckpt} in {time.time() - t0:.0f} s, noise {a.noise}, gather {a.gather_ms} ms, "
          f"max batch {a.max_batch}", flush=True)
    server = BatchServer(None, a, tag="pi05-shared")
    server.backend = Pi05Backend(BatchedPolicy(policy, _model.Observation.from_dict, jax, jnp), jax, a.noise, a.seed,
                                 min(a.precompile, a.max_batch), server.say)
    server.serve()


if __name__ == "__main__":
    main()
