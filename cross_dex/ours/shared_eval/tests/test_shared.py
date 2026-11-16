#!/usr/bin/env python
"""CPU tests of the shared-model evaluation logic with fake policies (no GPU, no openpi / LeRobot / Isaac Sim).

1  batch_server: N socket sessions, each drawing its noise from its own generator, served concurrently with lockstep
   gathering (forwards of batch > 1) give exactly the outputs of the same sessions served one at a time.
2  pi05 batched_policy (needs jax): a fake openpi Policy whose infer() is openpi's Policy.infer logic line for line,
   with a fake model drawing noise = jax.random.normal(rng, (B, H, D)) when none is given:
   a  B=1 BatchedPolicy + process-mode SessionKey == Policy.infer of a fresh policy, call after call
   b  sessions batched together == each alone (exactly: the fake model is row-wise)
   c  the full server (BatchServer + Pi05Backend, sockets, lockstep) == a fresh Policy.infer per session (process mode)
   d  episode mode: an episode's chunks depend only on (seed, episode), not on the episode order or the batch

  uv run --no-project --with "jax[cpu]" --with numpy python tests/test_shared.py
"""
import argparse
import os
import secrets
import socket
import sys
import threading
import time
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pi05"))
from batch_server import BatchServer, add_server_args, episode_seed  # noqa: E402


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def server_args(sessions, gather_ms=1000.0, max_batch=8):
    p = add_server_args(argparse.ArgumentParser())
    return p.parse_args(["--port", str(free_port()), "--authkey", secrets.token_hex(8), "--sessions", str(sessions),
                         "--gather_ms", str(gather_ms), "--max_batch", str(max_batch)])


def run_server(backend, args):
    server = BatchServer(backend, args, tag="test")
    server.say = lambda text: None
    forwards = []
    run = server.run_batches

    def record(pending):
        forwards.append(len(pending))
        run(pending)
    server.run_batches = record
    thread = threading.Thread(target=server.serve, daemon=True)
    thread.start()
    return thread, forwards


def connect(args):
    for _ in range(100):
        try:
            return Client(("127.0.0.1", args.port), authkey=args.authkey.encode())
        except ConnectionRefusedError:
            time.sleep(0.05)
    raise RuntimeError("no server")


def run_sessions(backend_factory, scripts, gather_ms=1000.0, concurrent=True):
    """scripts: per session a list of (kind, message); -> per session the replies of its requests"""
    outs = [None] * len(scripts)

    def client(i, args, barrier):
        conn = connect(args)
        if barrier:
            barrier.wait()
        replies = []
        for kind, message in scripts[i]:
            conn.send((kind, message))
            reply = conn.recv()
            if kind != "reset":
                replies.append(np.asarray(reply))
            time.sleep(0.001 * ((i * 7 + len(replies)) % 5))  # sessions at different speeds
        conn.send(("close", None))
        conn.close()
        outs[i] = np.stack(replies)

    forwards = []
    if concurrent:
        args = server_args(len(scripts), gather_ms)
        thread, forwards = run_server(backend_factory(), args)
        barrier = threading.Barrier(len(scripts))
        threads = [threading.Thread(target=client, args=(i, args, barrier)) for i in range(len(scripts))]
        [t.start() for t in threads]
        [t.join() for t in threads]
        thread.join(timeout=10)
    else:
        for i in range(len(scripts)):
            args = server_args(1, gather_ms)
            thread, f = run_server(backend_factory(), args)
            client(i, args, None)
            thread.join(timeout=10)
            forwards += f
    return outs, forwards


# ---------------------------------------------------------------- 1: generic batching with per-session generators
class FakeBackend:
    """Every 'act' needs a forward when its queue is empty (chunks of 3); noise from the session's numpy generator."""

    def opened(self, s):
        s.rng, s.queue, s.obs = None, [], None

    def reset(self, s, message):
        s.rng, s.queue = np.random.default_rng(message["seed"]), []
        return "ok"

    def observe(self, s, message):
        s.obs = np.asarray(message, np.float64)
        return not s.queue

    def forward(self, sessions):
        obs = np.stack([s.obs for s in sessions])
        noise = np.stack([s.rng.standard_normal((3, obs.shape[1])) for s in sessions])
        chunk = obs[:, None, :] * 2.0 + noise  # row-wise model
        for i, s in enumerate(sessions):
            s.queue = list(chunk[i])

    def reply(self, s):
        return s.queue.pop(0)

    def synchronize(self):
        pass

    def memory(self):
        return {}


def test_generic(n=4):
    scripts = []
    for i in range(n):
        script = []
        for ep in range(2):
            script.append(("reset", {"seed": episode_seed(0, 10 * i + ep)}))
            script += [("act", [float(i), float(ep), float(t)]) for t in range(7 + i)]
        scripts.append(script)
    together, forwards = run_sessions(FakeBackend, scripts)
    alone, _ = run_sessions(FakeBackend, scripts, concurrent=False)
    same = all(np.array_equal(x, y) for x, y in zip(together, alone))
    print(f"1 generic: {n} sessions batched == alone: {same}; forward batch sizes {sorted(set(forwards))}")
    assert same and max(forwards) > 1


# ---------------------------------------------------------------- 2: pi0.5 batched policy (jax)
def pi05_tests(n=3, calls=5):
    import jax
    import jax.numpy as jnp

    from batched_policy import BatchedPolicy, SessionKey
    from openpi_server import Pi05Backend

    H, D = 4, 6

    class FakeModel:
        action_horizon, action_dim = H, D

        def sample_actions(self, rng, observation, *, num_steps=10, noise=None):
            b = observation["state"].shape[0]
            if noise is None:
                noise = jax.random.normal(rng, (b, H, D))
            cond = observation["state"].sum(-1) + observation["image"]["base"].astype(jnp.float32).mean((1, 2, 3)) \
                + observation["tokenized_prompt"].sum(-1) * 1e-3 + observation["image_mask"]["base"]
            return noise * 0.5 + cond[:, None, None]

    def input_transform(data):
        prompt = data.pop("prompt")
        return {"state": np.asarray(data["state"], np.float32) * 2, "image": {"base": data["image"]},
                "image_mask": {"base": np.True_},
                "tokenized_prompt": np.frombuffer(prompt.encode().ljust(8)[:8], np.uint8).astype(np.int32)}

    def output_transform(data):
        return {"actions": np.asarray(data["actions"]) * 3 + 1, "state": data["state"]}

    class FakePolicy:  # openpi Policy, JAX branch of __init__ / infer
        def __init__(self):
            self._model = FakeModel()
            self._input_transform, self._output_transform = input_transform, output_transform
            self._sample_kwargs, self._is_pytorch_model = {}, False
            self._sample_actions = self._model.sample_actions
            self._rng = jax.random.key(0)

        def infer(self, obs, *, noise=None):
            inputs = jax.tree.map(lambda x: x, obs)
            inputs = self._input_transform(inputs)
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng = jax.random.split(self._rng)
            kwargs = dict(self._sample_kwargs)
            if noise is not None:
                kwargs["noise"] = jnp.asarray(noise)[None] if noise.ndim == 2 else jnp.asarray(noise)
            outputs = {"state": inputs["state"], "actions": self._sample_actions(sample_rng, inputs, **kwargs)}
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
            return self._output_transform(outputs)

    def obs(i, t):
        rng = np.random.default_rng(100 * i + t)
        return {"image": rng.integers(0, 256, (6, 8, 3), dtype=np.uint8), "state": rng.uniform(-1, 1, 5).astype(np.float32),
                "prompt": f"task {i % 2}"}

    def batched():
        return BatchedPolicy(FakePolicy(), lambda d: d, jax, jnp)

    streams = [[obs(i, t) for t in range(calls)] for i in range(n)]

    # a
    ref = FakePolicy()
    reference = np.stack([ref.infer(o)["actions"] for o in streams[0]])
    bp, key = batched(), SessionKey(jax, "process")
    ours = np.stack([bp.infer_batch([o], [key.next()])[0]["actions"] for o in streams[0]])
    print(f"2a pi05 B=1 vs Policy.infer: identical {np.array_equal(ours, reference)} (max |d| {np.abs(ours - reference).max():.2e})")
    assert np.array_equal(ours, reference)

    # b
    alone = []
    for s in streams:
        k = SessionKey(jax, "process")
        alone.append(np.stack([bp.infer_batch([o], [k.next()])[0]["actions"] for o in s]))
    keys, together = [SessionKey(jax, "process") for _ in streams], [[] for _ in streams]
    for t in range(calls):
        for i, out in enumerate(bp.infer_batch([s[t] for s in streams], [k.next() for k in keys])):
            together[i].append(out["actions"])
    same = all(np.array_equal(np.stack(x), y) for x, y in zip(together, alone))
    print(f"2b pi05 {n} sessions batched == alone: {same}")
    assert same

    # c: full server, process mode, sessions send reset like session_server.py
    def backend():
        return Pi05Backend(batched(), jax, "process", 0, precompile=2, say=lambda text: None)
    scripts = [[("reset", {"episode": 0})] + [("infer", o) for o in s] for s in streams]
    served, forwards = run_sessions(backend, scripts)
    refs = []
    for s in streams:
        p = FakePolicy()
        refs.append(np.stack([p.infer(o)["actions"] for o in s]))
    same = all(np.array_equal(x, y) for x, y in zip(served, refs))
    print(f"2c pi05 server ({n} sessions, lockstep) == baseline process per session: {same}; batch sizes {sorted(set(forwards))}")
    assert same and max(forwards) > 1

    # d: episode mode
    def backend_ep():
        return Pi05Backend(batched(), jax, "episode", 7, say=lambda text: None)
    scripts = [[("reset", {"episode": e}), ("infer", streams[0][0]), ("infer", streams[0][1])] for e in (3, 1, 2)]
    scripts.append([("reset", {"episode": 2}), ("infer", streams[0][0]), ("infer", streams[0][1]),
                    ("reset", {"episode": 3}), ("infer", streams[0][0]), ("infer", streams[0][1])])
    served, _ = run_sessions(backend_ep, scripts)
    ok = np.array_equal(served[3][:2], served[2]) and np.array_equal(served[3][2:], served[0]) \
        and not np.array_equal(served[0], served[1])
    print(f"2d pi05 episode mode: chunks depend only on (seed, episode): {ok}")
    assert ok


if __name__ == "__main__":
    test_generic()
    try:
        import jax  # noqa: F401
    except ImportError:
        print("2 skipped (no jax)")
        sys.exit(0)
    pi05_tests()
    print("ALL OK")
