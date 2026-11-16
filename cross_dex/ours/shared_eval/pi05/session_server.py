#!/usr/bin/env python
"""Isaac-side policy server of one evaluation session (DexLatent venv), started by the unchanged common evaluator in
place of its policy_server.py (through run_eval_shared.py). Same loop and protocol as dexverse_xlvla/server.py.

The policy is the baseline's own class, unchanged: dexverse_xlvla.pi05.eval_policy.Pi05PolicyServer (one hand) or
pi05.eval_bimanual.Pi05BimanualPolicyServer (bimanual) -> state encoding (latent / joint / LAD), wrist canonicalisation,
chunk of 10 executed open loop (replan), decoding to joint actions. Only where the chunk comes from changes:
  --mode dedicated  the class as is: it spawns its own openpi process (pi05/serve.py)  = the baseline path
  --mode shared     _spawn is replaced: no process, it connects to the GPU's openpi_server.py (--shared_port,
                    key $SHARED_EVAL_AUTHKEY) and sends ("reset", {"episode": <pickle index>}) at every episode
Both modes add (no change to actions): --barrier DIR:N (first reset waits until N sessions reached it, so benchmark
sessions run concurrently), --timing <jsonl> (per episode: steps, act ms with/without a new chunk, sim time between
acts), --save_obs K (the first K infer inputs -> <timing dir>/obs.npz, for check_batch.py).

  python session_server.py --ckpt <cell folder with server.json> --port <p> --authkey <k> --mode shared --shared_port <q>
         --episodes 0,1,...   (all arguments after --authkey are added by run_eval_shared.py)
"""
import argparse
import json
import os
import time
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", required=True)
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--authkey", required=True)
parser.add_argument("--mode", choices=("shared", "dedicated"), default="shared")
parser.add_argument("--shared_port", type=int, default=0)
parser.add_argument("--episodes", default="", help="The evaluator's episode list (comma); default 0..49.")
parser.add_argument("--barrier", default=None, metavar="DIR:N")
parser.add_argument("--timing", default=None)
parser.add_argument("--save_obs", type=int, default=0)
args = parser.parse_args()
cfg = json.loads((Path(args.ckpt) / "server.json").read_text())
EPISODES = [int(i) for i in args.episodes.split(",")] if args.episodes else list(range(50))

if cfg.get("bimanual"):
    from dexverse_xlvla.pi05.eval_bimanual import Pi05BimanualPolicyServer as Base
elif cfg["mode"] in ("pi05_latent", "pi05_joint", "pi05_lad"):
    from dexverse_xlvla.pi05.eval_policy import Pi05PolicyServer as Base
else:
    raise SystemExit(f"session_server: pi0.5 modes only, got {cfg['mode']}")


class Attached:
    """Stands in for the openpi subprocess handle when the model lives in the shared server."""

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


class RecordingConn:
    def __init__(self, conn, path, count):
        self.conn, self.path, self.count, self.saved = conn, path, count, []

    def send(self, item):
        kind, x = item
        if kind == "infer" and len(self.saved) < self.count:
            self.saved.append(x)
            if len(self.saved) == self.count:
                np.savez(self.path, **{f"{i}|{k}": np.asarray(v) for i, m in enumerate(self.saved) for k, v in m.items()})
        self.conn.send(item)

    def recv(self):
        return self.conn.recv()


class Session(Base):
    def _spawn(self, c):
        if args.mode == "dedicated":
            return super()._spawn(c)
        key = os.environ.get("SHARED_EVAL_AUTHKEY")
        if not (args.shared_port and key):
            raise SystemExit("--mode shared needs --shared_port and $SHARED_EVAL_AUTHKEY")
        self.proc, self.port, self.key, self.conn, self.t0 = Attached(), args.shared_port, key, None, time.time()
        self.ep, self.log, self.queue = -1, None, []

    def _connect(self):
        super()._connect()
        if args.save_obs and not isinstance(self.conn, RecordingConn):
            self.conn = RecordingConn(self.conn, Path(args.timing or args.ckpt).parent / "obs.npz", args.save_obs)

    def reset(self):
        super().reset()  # flushes the previous episode's log, ep += 1, empty chunk queue
        if args.mode == "shared":
            if self.conn is None:
                self._connect()
            self.conn.send(("reset", {"episode": EPISODES[self.ep] if self.ep < len(EPISODES) else None}))
            self.conn.recv()


policy = Session(cfg)


class Timing:
    def __init__(self):
        self.out = open(args.timing, "a", buffering=1) if args.timing else None
        self.pos, self.row, self.last = -1, None, None

    def flush(self):
        if self.out and self.row:
            self.row["wall_s"] = time.time() - self.row["start"]
            self.out.write(json.dumps(self.row) + "\n")
        self.row = None

    def reset(self):
        self.flush()
        self.pos += 1
        self.row = {"pos": self.pos, "episode": EPISODES[self.pos] if self.pos < len(EPISODES) else None,
                    "mode": args.mode, "pid": os.getpid(), "start": time.time(), "steps": 0,
                    "chunk_ms": [], "act_ms": [], "loop_ms": []}
        self.last = None

    def act(self, started, chunk):
        now = time.perf_counter()
        if self.row is None:
            return
        if self.last is not None:
            self.row["loop_ms"].append(round((started - self.last) * 1e3, 2))
        self.row["chunk_ms" if chunk else "act_ms"].append(round((now - started) * 1e3, 2))
        self.row["steps"] += 1
        self.last = now


def barrier():
    folder, count = args.barrier.rsplit(":", 1)
    Path(folder).mkdir(parents=True, exist_ok=True)
    (Path(folder) / str(os.getpid())).touch()
    while len(os.listdir(folder)) < int(count):
        time.sleep(0.2)
    print(f"[session] barrier passed at {time.time():.3f}", flush=True)


timing, first = Timing(), True
with Listener(("127.0.0.1", args.port), authkey=args.authkey.encode()) as listener, listener.accept() as conn:
    while True:  # = dexverse_xlvla/server.py
        kind, message = conn.recv()
        if kind == "close":
            timing.flush()
            getattr(policy, "close", lambda: None)()
            break
        if kind == "reset":
            if first and args.barrier:
                barrier()
            first = False
            timing.reset()
            policy.reset()
            conn.send("ok")
            continue
        t0 = time.perf_counter()
        action = np.asarray(policy.act(message), dtype=np.float32).tobytes()
        timing.act(t0, bool(policy.log["new_chunk"][-1]))
        conn.send(action)
