#!/usr/bin/env python
"""Run DexVerse's unchanged DP evaluator (baselines/lerobot_dp/eval_lerobot.py) with timing, against one of three policy
servers. Run in the main env (Isaac Sim) like the evaluator; all evaluator arguments pass through.

  --mode own        the evaluator's own policy_server.py, one model per session (today's path; unseeded noise)
  --mode dedicated  shared_server.py started for this session only (--sessions 1); per-episode seeded noise
  --mode shared     connect to a running shared_server.py (--address host:port, --authkey or $DP_SHARED_AUTHKEY)

Per episode it saves <timing_dir>/ep<k>.npz: actions, call_ms (send->reply), step_ms (env.step), chunk flags and the sim
clock (episode_length_buf, common_step_counter) read before the request and after the reply, to show the simulation
does not advance while the evaluator waits. --inject_delay_ms adds a sleep after each chunk reply (latency test).

  source /workspace/codes/_setup/dexverse_env.sh DexVerse
  python run_eval_dp.py --mode shared --address 127.0.0.1:7700 --seed 0 --demos <pkl> --ckpt <pretrained_model> \
      --output_dir <dir> --headless --episodes 0,1 --max_steps 400 --no-video
"""
import argparse
import importlib.util
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import zlib
from multiprocessing.connection import Client
from pathlib import Path

own = argparse.ArgumentParser(add_help=False)
own.add_argument("--mode", choices=("own", "dedicated", "shared"), required=True)
own.add_argument("--address", default=None, help="host:port of shared_server.py (--mode shared).")
own.add_argument("--authkey", default=os.environ.get("DP_SHARED_AUTHKEY"))
own.add_argument("--seed", type=int, default=0, help="Noise seed of episode k = seed * 100000 + k (dedicated/shared).")
own.add_argument("--timing_dir", default=None, help="Default <output_dir>/timing.")
own.add_argument("--inject_delay_ms", type=float, default=0.0)
own.add_argument("--barrier", default=None, metavar="DIR:N",
                 help="Before the first episode wait until N sessions have touched files in DIR (all envs built).")
own.add_argument("--save_obs", type=int, default=0, help="Save the first K observations of the first episode.")
own.add_argument("--server_log", default=None, help="dedicated: jsonl log of the started shared_server.py.")
mine, rest = own.parse_known_args()

EVALUATOR = Path(os.environ.get("DEXVERSE_ROOT", "/workspace/codes/dexverse/DexVerse")) / "baselines/lerobot_dp/eval_lerobot.py"
SERVER = Path(__file__).resolve().parent / "shared_server.py"
sys.argv = [str(EVALUATOR)] + rest
spec = importlib.util.spec_from_file_location("eval_lerobot", EVALUATOR)
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)  # parses the evaluator arguments and starts Isaac Sim, as when run directly

import numpy as np  # noqa: E402

N_ACTION_STEPS = json.loads((Path(ev.args.ckpt) / "config.json").read_text())["n_action_steps"]
EPISODES = [int(i) for i in ev.args.episodes.split(",")] if ev.args.episodes else list(range(50))
timing_dir = Path(mine.timing_dir or Path(ev.args.output_dir) / "timing")
timing_dir.mkdir(parents=True, exist_ok=True)
captured = {}


class NoServer:
    def wait(self, timeout=None):
        return 0


def connect(host, port, authkey, server=None):
    for _ in range(300):
        try:
            return Client((host, port), authkey=authkey.encode())
        except ConnectionRefusedError:
            if server is not None and server.poll() is not None:
                raise RuntimeError("policy server exited")
            time.sleep(1)
    raise RuntimeError("policy server did not start")


def start_server():
    if mine.mode == "own":
        server, conn = original_start_server()
    elif mine.mode == "dedicated":
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        authkey = secrets.token_hex(16)
        environ = {k: v for k, v in os.environ.items() if k not in ("LD_PRELOAD", "PYTHONPATH", "CONDA_PREFIX")}
        cmd = [ev.LEROBOT_PYTHON, str(SERVER), "--ckpt", ev.args.ckpt, "--port", str(port), "--authkey", authkey,
               "--sessions", "1", "--dp_dir", str(EVALUATOR.parent)]
        if mine.server_log:
            cmd += ["--log", mine.server_log]
        server = subprocess.Popen(cmd, env=environ)
        conn = connect("127.0.0.1", port, authkey, server)
    else:
        host, port = mine.address.rsplit(":", 1)
        server, conn = NoServer(), connect(host, int(port), mine.authkey)
    return server, TimedConnection(conn)


class TimedConnection:
    """The evaluator's policy connection, timed; seeds each episode's noise (dedicated/shared)."""

    def __init__(self, conn):
        self.conn, self.episode_pos, self.rows, self.pending = conn, -1, None, None

    def clock(self):
        base = captured["base"]
        return int(base.episode_length_buf[0]), int(base.common_step_counter)

    def flush(self):
        if self.rows and self.rows["actions"]:
            episode = EPISODES[self.episode_pos]
            np.savez(timing_dir / f"ep{episode:03d}.npz", wall_s=time.perf_counter() - self.started,
                     **{k: np.asarray(v) for k, v in self.rows.items()})

    def send(self, item):
        kind, message = item
        if kind == "reset":
            if self.episode_pos < 0 and mine.barrier:
                folder, count = mine.barrier.rsplit(":", 1)
                Path(folder).mkdir(parents=True, exist_ok=True)
                (Path(folder) / f"{os.getpid()}").touch()
                while len(os.listdir(folder)) < int(count):
                    time.sleep(0.2)
                print(f"[timing] barrier passed at {time.time():.3f}", flush=True)
            self.flush()
            self.episode_pos += 1
            self.started = time.perf_counter()
            self.rows = {k: [] for k in ("actions", "call_ms", "step_ms", "chunk", "clock_before", "clock_after",
                                         "loop_ms", "img_crc", "state")}
            if mine.mode != "own":
                message = {"seed": mine.seed * 100000 + EPISODES[self.episode_pos]}
        elif kind == "act":
            now = time.perf_counter()
            if self.pending is not None:  # time from the previous reply to this request: env.step + probe + observe
                self.rows["loop_ms"].append((now - self.pending) * 1e3)
            self.rows["clock_before"].append(self.clock())
            crc = 0
            for key in sorted(message):
                if key.startswith("observation.images"):
                    crc = zlib.crc32(message[key][2], crc)
            self.rows["img_crc"].append(crc)
            if self.episode_pos == 0 and len(self.rows["img_crc"]) <= mine.save_obs:
                self.saved = getattr(self, "saved", [])
                self.saved.append({k: np.frombuffer(v[2], dtype=v[1]).reshape(v[0]).copy() for k, v in message.items()})
                if len(self.saved) == mine.save_obs:
                    np.savez(timing_dir / "obs.npz", **{f"{i}|{k}": v for i, m in enumerate(self.saved) for k, v in m.items()})
            shape, dtype, data = message["observation.state"]
            self.rows["state"].append(np.frombuffer(data, dtype=dtype).reshape(shape).copy())
            self.sent = time.perf_counter()
        elif kind == "close":
            self.flush()
        self.kind = kind
        self.conn.send((kind, message))

    def recv(self):
        reply = self.conn.recv()
        if self.kind == "act":
            call_ms = (time.perf_counter() - self.sent) * 1e3
            chunk = len(self.rows["actions"]) % N_ACTION_STEPS == 0
            if chunk and mine.inject_delay_ms:
                time.sleep(mine.inject_delay_ms / 1e3)
            self.rows["clock_after"].append(self.clock())
            self.rows["call_ms"].append(call_ms)
            self.rows["chunk"].append(chunk)
            self.rows["actions"].append(np.frombuffer(reply, dtype=np.float32).copy())
            self.pending = time.perf_counter()
        elif self.kind == "reset":
            self.pending = None
        return reply


def make_env(payload):
    env, env_cfg, probe = original_make_env(payload)
    captured["base"] = env.unwrapped
    step = env.step

    def timed_step(action):
        t0 = time.perf_counter()
        out = step(action)
        conn = captured.get("conn")
        if conn is not None and conn.rows is not None:
            conn.rows["step_ms"].append((time.perf_counter() - t0) * 1e3)
        return out

    env.step = timed_step
    return env, env_cfg, probe


original_start_server, original_make_env = ev.start_server, ev.make_env


def start_server_captured():
    server, conn = start_server()
    captured["conn"] = conn
    return server, conn


ev.start_server, ev.make_env = start_server_captured, make_env

if __name__ == "__main__":
    started = time.time()
    try:
        ev.main()
    except BaseException:
        import traceback

        traceback.print_exc()
        raise
    finally:
        (timing_dir / "run.json").write_text(json.dumps({**vars(mine), "evaluator_args": rest, "wall_s": time.time() - started,
                                                         "pid": os.getpid()}))
        ev.simulation_app.close()
