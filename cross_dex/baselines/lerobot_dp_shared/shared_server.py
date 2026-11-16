#!/usr/bin/env python
"""One LeRobot Diffusion Policy shared by several evaluation sessions (lerobot env).

Same protocol per connection as DexVerse baselines/lerobot_dp/policy_server.py, so the unchanged evaluator
(eval_lerobot.py, through run_eval_dp.py) can talk to it:
  ("reset", None | {"seed": int})  -> "ok"      new episode for this session (own obs/action queues, own noise generator)
  ("act", obs)                     -> float32 action bytes
  ("close", None)                  -> (no reply) the session ends
Any number of sessions connect to one port. Each keeps its own DP observation history and action queue (n_obs_steps /
n_action_steps, like DiffusionPolicy.select_action). A call whose action queue still holds actions is answered at once;
calls that need a new chunk are gathered (up to --gather_ms, or until every connected session waits) and denoised in one
batched pass. Noise comes from a per-session CPU generator (initial sample and every DDPM step, row by row), so a
session's actions do not depend on which other sessions share its batch.

  python shared_server.py --ckpt <pretrained_model> --port 7700 --authkey <hex> [--sessions N] [--log calls.jsonl]
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
from multiprocessing.connection import Listener

import numpy as np
import torch

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--ckpt", required=True, help="pretrained_model folder of a checkpoint.")
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--authkey", required=True)
parser.add_argument("--sessions", type=int, default=0, help="Exit after this many sessions closed (0 = run until killed).")
parser.add_argument("--gather_ms", type=float, default=5.0, help="Wait this long for more chunk requests to batch.")
parser.add_argument("--max_batch", type=int, default=16)
parser.add_argument("--generator_device", default="cpu",
                    help="Noise generators on cpu (independent of batching) or cuda (B=1 matches LeRobot's own sampling).")
parser.add_argument("--log", default=None, help="jsonl with one line per call and per batched forward.")
parser.add_argument("--dp_dir", default=os.path.join(os.environ.get("DEXVERSE_ROOT", "/workspace/codes/dexverse/DexVerse"),
                                                     "baselines", "lerobot_dp"),
                    help="DexVerse baselines/lerobot_dp (image_resize.py, read only).")
args = parser.parse_args()

sys.path.insert(0, args.dp_dir)
from image_resize import install_resize_for  # noqa: E402

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.policies.utils import populate_queues  # noqa: E402

device = torch.device("cuda")
policy = DiffusionPolicy.from_pretrained(args.ckpt).to(device).eval()
print(f"[shared] 240x320 resize: {install_resize_for(policy.config)}", flush=True)
preprocessor, postprocessor = make_pre_post_processors(
    policy.config, pretrained_path=args.ckpt, preprocessor_overrides={"device_processor": {"device": str(device)}})
policy.reset()
QUEUE_KEYS = list(policy._queues)
ACTION = next(k for k in QUEUE_KEYS if k == "action")
IMAGES = next((k for k in QUEUE_KEYS if k == "observation.images"), None)
CFG = policy.config
log = open(args.log, "a", buffering=1 << 16) if args.log else None


def write_log(**row):
    if log:
        log.write(json.dumps(row) + "\n")


def to_batch(message: dict) -> dict:  # = policy_server.to_batch
    batch = {}
    for key, (shape, dtype, data) in message.items():
        value = torch.from_numpy(np.frombuffer(data, dtype=dtype).reshape(shape).copy())
        batch[key] = value.permute(2, 0, 1).float() / 255.0 if value.ndim == 3 else value.float()
    return batch


class Session:
    def __init__(self, sid, conn):
        self.sid, self.conn = sid, conn
        self.queues, self.generator, self.chunk_obs, self.episode, self.call = None, None, None, -1, 0

    def reset(self, message):
        policy.reset()  # fresh deques of the right maxlen, then keep them on the session
        self.queues, policy._queues = policy._queues, None
        seed = message.get("seed") if isinstance(message, dict) else None
        if seed is None:
            seed = int(np.random.SeedSequence().entropy % (1 << 63))
        self.generator = torch.Generator(device=args.generator_device).manual_seed(int(seed))
        self.episode, self.call = self.episode + 1, 0

    def observe(self, message) -> bool:
        """DiffusionPolicy.select_action up to the model call; True if a new chunk is needed."""
        batch = dict(preprocessor(to_batch(message)))
        batch.pop(ACTION, None)
        if IMAGES:
            batch[IMAGES] = torch.stack([batch[key] for key in CFG.image_features], dim=-4)
        populate_queues(self.queues, batch)
        if len(self.queues[ACTION]):
            return False
        self.chunk_obs = {k: torch.stack(list(self.queues[k]), dim=1) for k in batch if k in self.queues}
        return True

    def reply(self):
        action = postprocessor(self.queues[ACTION].popleft())
        self.conn.send(action[0].cpu().numpy().astype(np.float32).tobytes())
        self.call += 1


@torch.inference_mode()
def denoise(sessions):
    """DiffusionModel.generate_actions for a batch, with each row's noise from its session's generator."""
    model = policy.diffusion
    obs = {k: torch.cat([s.chunk_obs[k] for s in sessions]) for k in sessions[0].chunk_obs}
    global_cond = model._prepare_global_conditioning(obs)
    generators = [s.generator for s in sessions]
    shape = (1, CFG.horizon, CFG.action_feature.shape[0])
    sample = torch.cat([torch.randn(shape, generator=g, device=g.device, dtype=global_cond.dtype)
                        for g in generators]).to(device)
    model.noise_scheduler.set_timesteps(model.num_inference_steps)
    for t in model.noise_scheduler.timesteps:
        output = model.unet(sample, torch.full(sample.shape[:1], t, dtype=torch.long, device=device),
                            global_cond=global_cond)
        sample = model.noise_scheduler.step(output, t, sample, generator=generators).prev_sample
    start = CFG.n_obs_steps - 1
    actions = sample[:, start:start + CFG.n_action_steps]
    for i, s in enumerate(sessions):
        s.queues[ACTION].extend(actions[i:i + 1].transpose(0, 1))
        s.chunk_obs = None


requests = queue.Queue()
active, lock = set(), threading.Lock()
counter = {"next": 0, "closed": 0}


def reader(session):
    try:
        while True:
            kind, message = session.conn.recv()
            requests.put((session, kind, message, time.perf_counter()))
            if kind == "close":
                return
    except (EOFError, OSError):
        requests.put((session, "close", None, time.perf_counter()))


def acceptor(listener):
    while True:
        conn = listener.accept()
        with lock:
            session = Session(counter["next"], conn)
            counter["next"] += 1
            active.add(session)
        print(f"[shared] session {session.sid} connected ({len(active)} active)", flush=True)
        threading.Thread(target=reader, args=(session,), daemon=True).start()


def handle(item, pending):
    session, kind, message, arrived = item
    if kind == "close":
        with lock:
            if session in active:
                active.discard(session)
                counter["closed"] += 1
        print(f"[shared] session {session.sid} closed ({len(active)} active)", flush=True)
        return
    if kind == "reset":
        session.reset(message)
        session.conn.send("ok")
        return
    with torch.inference_mode():
        needs_chunk = session.observe(message)
    if needs_chunk:
        pending.append(item)
    else:
        session.reply()
        write_log(kind="call", sid=session.sid, ep=session.episode, call=session.call - 1, fwd=False, batch=0,
                  wait_ms=(time.perf_counter() - arrived) * 1e3)


def serve():
    with Listener((args.host, args.port), authkey=args.authkey.encode()) as listener:
        threading.Thread(target=acceptor, args=(listener,), daemon=True).start()
        print(f"[shared] listening on {args.host}:{args.port}", flush=True)
        while not (args.sessions and counter["closed"] >= args.sessions):
            try:
                item = requests.get(timeout=1.0)
            except queue.Empty:
                continue
            pending = []
            handle(item, pending)
            while True:  # whatever is already waiting
                try:
                    handle(requests.get_nowait(), pending)
                except queue.Empty:
                    break
            if not pending:
                continue
            deadline = time.perf_counter() + args.gather_ms / 1e3
            while len(pending) < min(len(active), args.max_batch):
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    handle(requests.get(timeout=remaining), pending)
                except queue.Empty:
                    break
            for i in range(0, len(pending), args.max_batch):
                chunk = pending[i:i + args.max_batch]
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                denoise([s for s, *_ in chunk])
                torch.cuda.synchronize()
                forward_ms = (time.perf_counter() - t0) * 1e3
                write_log(kind="forward", t=time.time(), batch=len(chunk), ms=forward_ms, sids=[s.sid for s, *_ in chunk])
                for session, _, _, arrived in chunk:
                    session.reply()
                    write_log(kind="call", sid=session.sid, ep=session.episode, call=session.call - 1, fwd=True,
                              batch=len(chunk), wait_ms=(time.perf_counter() - arrived) * 1e3)
    write_log(kind="memory", max_allocated_mb=torch.cuda.max_memory_allocated() / 2**20,
              max_reserved_mb=torch.cuda.max_memory_reserved() / 2**20)
    if log:
        log.close()
    print("[shared] all sessions closed", flush=True)


serve()
