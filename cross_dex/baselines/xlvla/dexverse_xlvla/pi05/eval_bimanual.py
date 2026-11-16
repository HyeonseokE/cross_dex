"""Bimanual pi0.5 policies (pi05_latent = pi05_XL, pi05_joint = pi05) for the common evaluator, through server.py /
run_eval.py like the one-hand Pi05PolicyServer (eval_policy.py); only the DexVerse <-> policy-space conversion differs
(dexverse_xlvla.bimanual, simulator-read calibration of measure_wrist_bimanual.py).

server.json: the one-hand keys plus
  "bimanual": true, "hand": <family, e.g. "allegro">, "wrist": <bimanual wrist_calib_<task>.json>,
  "latent_ckpt": <right 6-hand latent>, "latent_ckpt_left": <left 6-hand latent> (pi05_latent),
  "action_dim": 82 (latent) | 62 (joint), "max_token_len": 360 | 280 (as trained), "pad": 22 (joint)
state  : evaluator joint_pos (action-term order first) -> [R wrist 9 | R z/fingers | L wrist 9 | L z/fingers]
images : third_person -> "image", eye_in_hand (right wrist camera) -> "wrist_image", eye_in_hand_left -> "left_wrist_image"
actions: chunk (H=10), the first `replan` executed open loop; per hand wrist 9 -> wrist joints by IK (warm start = the
         observed wrist joints of that hand), fingers = D(z) (latent) or the first n values (joint) -> env action.
"""
from __future__ import annotations

import numpy as np

from dexverse_xlvla.pi05.eval_policy import Pi05PolicyServer

IMG = {"image": "observation.images.third_person", "wrist_image": "observation.images.eye_in_hand",
       "left_wrist_image": "observation.images.eye_in_hand_left"}


class Pi05BimanualPolicyServer(Pi05PolicyServer):
    def __init__(self, cfg: dict):   # replaces the one-hand set-up (no super().__init__)
        from dexverse_xlvla import bimanual as B
        from dexverse_xlvla.pi05.v3_dataset import PROMPTS

        if cfg["mode"] not in ("pi05_latent", "pi05_joint"):
            raise ValueError(f"bimanual: mode {cfg['mode']} not supported")
        self.cfg, self.latent = cfg, cfg["mode"] == "pi05_latent"
        self.hand, self.prompt, self.replan = cfg["hand"], PROMPTS[cfg["task"]], int(cfg.get("replan", 10))
        self.space = B.build(cfg["wrist"], self.hand, "latent" if self.latent else "joint", cfg.get("latent_ckpt"),
                             cfg.get("latent_ckpt_left"), int(cfg.get("pad", B.PAD)))
        if self.space.dim != int(cfg["action_dim"]):
            raise ValueError(f"action_dim {cfg['action_dim']} != bimanual {cfg['mode']} space {self.space.dim}")
        self.sides = B.SIDES
        self._spawn(cfg)
        self.ep, self.log, self.queue = -1, None, []

    def act(self, message: dict) -> np.ndarray:
        obs = {}
        for key, (shape, dtype, data) in message.items():
            obs[key] = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        q = obs["observation.state"].astype(np.float64)
        new_chunk = not self.queue
        if new_chunk:
            self._connect() if self.conn is None else None
            x = {k: obs[v].astype(np.uint8) for k, v in IMG.items()}
            x.update(state=self.space.to_policy(q[None, : self.space.n_act], "state")[0], prompt=self.prompt)
            self.conn.send(("infer", x))
            self.queue = list(self.conn.recv()[: self.replan])
        out = np.asarray(self.queue.pop(0), np.float64)
        q0 = {s: q[self.space.side[s]["wdims"]] for s in self.sides}     # observed wrist joints (state = action order)
        act = self.space.to_action(out[None, : self.space.dim], q0)[0]
        self.log["policy_out"].append(out)
        self.log["action"].append(act)
        self.log["new_chunk"].append(new_chunk)
        return act.astype(np.float32)
