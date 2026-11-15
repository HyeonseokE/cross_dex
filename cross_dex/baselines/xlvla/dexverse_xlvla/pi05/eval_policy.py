"""pi0.5 policies (latent = XL-VLA, joint = plain pi0.5 baseline) served to the common evaluator through server.py /
run_eval.py, like the latent DP (dexverse_xlvla.policy). The openpi model runs in its own venv (pi05/serve.py, started here
as a subprocess); this side does the DexVerse <-> policy-space conversion with the same code as the latent DP:

server.json:
  {"mode": "pi05_latent" | "pi05_joint", "hand": h, "task": t, "policy_ckpt": <openpi checkpoints/<run>/<exp>/<step>>,
   "run": <TrainConfig name>, "repo_id": <norm-stats asset id>, "action_dim": 41 | 32,
   "latent_ckpt": <latent.pt (VAE; joint mode: hand models only)>, "wrist": <wrist_calib_<task>.json>,
   "replan": 10, "log_dir": null}
state  : evaluator joint_pos -> [canonical wrist 9 | E_h(fingers) 32] (latent) or [wrist 9 | fingers zero-padded to 22]
         (joint; make_joint_dataset), fingers = angles_to_normalized(...).clamp(-1, 1) as in the datasets
images : observation.images.third_person -> "image", observation.images.eye_in_hand -> "wrist_image" (HWC uint8)
prompt : v3_dataset.PROMPTS[task] (training prompts)
actions: openpi chunk (H=10); the first `replan` steps are executed open loop, then a new chunk.
         wrist 9 -> wrist joints by IK (warm start = observed wrist joints); fingers = D_h(z) (latent) or the first n_h
         padded values (joint), normalized -> joint angles with the hand's limits.
"""
from __future__ import annotations

import os
import secrets
import socket
import subprocess
import time
from multiprocessing.connection import Client

import numpy as np
import sys

import torch

OPENPI_PY = os.environ.get("XLVLA_OPENPI_PYTHON", "/workspace/envs/dexverse/openpi/bin/python")
CODE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # DexLatent root
IMG = {"image": "observation.images.third_person", "wrist_image": "observation.images.eye_in_hand"}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Pi05PolicyServer:
    def __init__(self, cfg: dict):
        from dexverse_xlvla.eval_latent import load
        from dexverse_xlvla.pi05.v3_dataset import PROMPTS
        from dexverse_xlvla.replay import action_names
        from dexverse_xlvla.wrist import load_wrists

        self.cfg, self.latent = cfg, cfg["mode"] == "pi05_latent"
        self.hand, self.prompt, self.replan = cfg["hand"], PROMPTS[cfg["task"]], int(cfg.get("replan", 10))
        self.tr = load(cfg["latent_ckpt"])
        for a in self.tr.autoencoders.values():
            a.eval()
        self.m = self.tr.hand_models[self.hand]
        self.names = action_names(cfg["task"], self.hand)
        self.idx = self.m.data_index(self.names)
        self.wrist = load_wrists(cfg["wrist"], {self.hand: self.m})[self.hand]
        self.pad = int(cfg.get("pad", 22))
        self.lad = None
        if cfg["mode"] == "pi05_lad":   # LAD embedding as the finger space (robot_clip dexverse_lad.pi05_space), frozen
            sys.path.insert(0, os.environ.get("XLVLA_LAD_CODE", "/workspace/codes/dexverse/robot_clip"))
            from dexverse_lad.pi05_space import LadSpace
            self.lad = LadSpace(cfg["lad_ckpt"], self.tr.hand_models)
        port, key = _free_port(), secrets.token_hex(8)
        env = dict(os.environ, PYTHONPATH=CODE, XLA_PYTHON_CLIENT_PREALLOCATE="false")
        self.proc = subprocess.Popen([OPENPI_PY, "-m", "dexverse_xlvla.pi05.serve", "--ckpt", cfg["policy_ckpt"],
                                      "--run", cfg["run"], "--repo_id", cfg["repo_id"], "--action_dim",
                                      str(cfg["action_dim"]), "--port", str(port), "--authkey", key]
                                     + (["--max_token_len", str(cfg["max_token_len"])] if cfg.get("max_token_len") else []),
                                     cwd=CODE, env=env)
        self.port, self.key, self.conn, self.t0 = port, key, None, time.time()
        self.ep, self.log, self.queue = -1, None, []

    def _connect(self):
        """lazy: the evaluator gives its policy server 300 s to accept, model load + jit may take longer"""
        while self.conn is None:
            try:
                self.conn = Client(("127.0.0.1", self.port), authkey=self.key.encode())
            except ConnectionRefusedError:
                if self.proc.poll() is not None or time.time() - self.t0 > 1800:
                    raise RuntimeError(f"pi0.5 server did not start (exit {self.proc.poll()})")
                time.sleep(2)
        print(f"[pi05] {self.cfg['mode']} {self.hand} {self.cfg['task']}: server up after {time.time() - self.t0:.0f} s, "
              f"replan {self.replan}", flush=True)

    def _flush(self):
        if self.log and self.cfg.get("log_dir"):
            os.makedirs(self.cfg["log_dir"], exist_ok=True)
            np.savez(f"{self.cfg['log_dir']}/ep{self.ep:03d}.npz", **{k: np.asarray(v) for k, v in self.log.items()})

    def reset(self):
        self._flush()
        self.queue = []
        self.ep, self.log = self.ep + 1, {"policy_out": [], "action": [], "new_chunk": []}

    def _state(self, q: np.ndarray) -> np.ndarray:
        n = len(self.names)
        wrist = self.wrist.vec_from_joints(q[None, :6])[0]
        fingers = self.m.angles_to_normalized(torch.tensor(q[None, :n][:, self.idx], dtype=torch.float32)).clamp(-1, 1)
        if self.lad is not None:
            f = self.lad.encode(self.hand, fingers)[0].numpy()
        elif self.latent:
            from dexverse_xlvla.eval_latent import enc
            with torch.no_grad():
                f = enc(self.tr, self.hand, fingers)[0].numpy()
        else:
            f = np.zeros(self.pad, np.float32)
            f[:fingers.shape[1]] = fingers[0].numpy()
        return np.concatenate([wrist, f]).astype(np.float32)

    def act(self, message: dict) -> np.ndarray:
        obs = {}
        for key, (shape, dtype, data) in message.items():
            obs[key] = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        q = obs["observation.state"].astype(np.float64)
        new_chunk = not self.queue
        if new_chunk:
            self._connect() if self.conn is None else None
            x = {k: obs[v].astype(np.uint8) for k, v in IMG.items()}
            x.update(state=self._state(q), prompt=self.prompt)
            self.conn.send(("infer", x))
            self.queue = list(self.conn.recv()[: self.replan])
        out = np.asarray(self.queue.pop(0), np.float64)
        act = np.zeros(len(self.names))
        act[:6] = self.wrist.actions_from_vec(out[None, :9], q_init=q[:6])[0]
        if self.lad is not None:
            qn = self.lad.decode(self.hand, torch.tensor(out[None, 9:9 + self.lad.dim], dtype=torch.float32))[0]
        elif self.latent:
            from dexverse_xlvla.eval_latent import dec
            with torch.no_grad():
                qn = dec(self.tr, self.hand, torch.tensor(out[None, 9:41], dtype=torch.float32))[0].clamp(-1, 1)
        else:
            qn = torch.tensor(out[9:9 + len(self.idx)], dtype=torch.float32).clamp(-1, 1)
        lo, hi = self.m._lower, self.m._upper
        act[self.idx] = ((qn + 1) * 0.5 * (hi - lo) + lo).numpy()
        self.log["policy_out"].append(out)
        self.log["action"].append(act)
        self.log["new_chunk"].append(new_chunk)
        return act.astype(np.float32)

    def close(self):
        self._flush()
        try:
            if self.conn is not None:
                self.conn.send(("close", None))
            self.proc.wait(timeout=60)
        except (OSError, EOFError, subprocess.TimeoutExpired):
            self.proc.kill()
