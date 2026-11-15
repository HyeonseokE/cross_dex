"""Step 5/6: serve the latent Diffusion Policy to the common evaluator (through server.py / run_eval.py).

server.json (mode latent_dp):
  {"mode": "latent_dp", "hand": "allegro", "policy_ckpt": <LeRobot pretrained_model>, "latent_ckpt": <latent.pt>,
   "wrist": <calib.json>, "task": "Dexverse-GraspKettle-v0", "log_dir": <dir or null>}
  {"mode": "joint_dp", "policy_ckpt": <LeRobot pretrained_model>, "log_dir": ...}   (joint-space DP, logging only)
  optional "task_onehot": [<task>, ...]  multi-task policy: the one-hot of "task" in that list is appended to the state
  (as make_dataset --task_onehot); "wrist" is then the calibration of "task".
  optional "noise_scheduler_type" ("DDPM" | "DDIM") / "num_inference_steps": sampler at evaluation (default: as trained).

latent_dp: observation.state (joint_pos, evaluator order = action order first) -> [canonical wrist pose | E_h(fingers)]
-> LeRobot DP (unchanged; same image resize as the joint DP server) -> [wrist pose | z] -> wrist joints by IK (warm start
= current wrist joints), finger joints = D_h(z) -> action in the hand's action order.
Both modes log per episode (step 6): the policy output, the executed action and whether a new chunk started.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

def _server() -> str:   # the container's one /workspace/experiments/<server> folder
    return next(d for d in sorted(os.listdir("/workspace/experiments"))
                if os.path.isdir(f"/workspace/experiments/{d}") and d[0] not in "_.")


COMMON_DP = os.environ.get("XLVLA_COMMON_DP") or \
    f"/workspace/experiments/{_server()}/20261031_teleop_single/teleop_dp_image"   # image_resize (called, not changed)


class LatentPolicyServer:
    def __init__(self, cfg: dict):
        sys.path.insert(0, COMMON_DP)
        from image_resize import install_resize_for
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from lerobot.policies.factory import make_pre_post_processors

        self.cfg, self.dev = cfg, torch.device("cuda")
        pcfg = None
        if cfg.get("noise_scheduler_type") or cfg.get("num_inference_steps"):   # e.g. DDIM 10 at evaluation (trained DDPM 100)
            from lerobot.configs.policies import PreTrainedConfig
            pcfg = PreTrainedConfig.from_pretrained(cfg["policy_ckpt"])
            pcfg.noise_scheduler_type = cfg.get("noise_scheduler_type") or pcfg.noise_scheduler_type
            pcfg.num_inference_steps = cfg.get("num_inference_steps") or pcfg.num_inference_steps
        self.policy = DiffusionPolicy.from_pretrained(cfg["policy_ckpt"], config=pcfg).to(self.dev).eval()
        d = self.policy.diffusion
        print(f"[policy] sampler {type(d.noise_scheduler).__name__} {d.num_inference_steps} steps", flush=True)
        print(f"[policy] 240x320 resize: {install_resize_for(self.policy.config)}", flush=True)
        self.pre, self.post = make_pre_post_processors(
            self.policy.config, pretrained_path=cfg["policy_ckpt"],
            preprocessor_overrides={"device_processor": {"device": str(self.dev)}})
        self.latent = cfg["mode"] == "latent_dp"
        if self.latent:
            from dexverse_xlvla.eval_latent import load
            from dexverse_xlvla.replay import action_names
            from dexverse_xlvla.wrist import load_wrists
            self.hand = cfg["hand"]
            self.tr = load(cfg["latent_ckpt"])
            for a in self.tr.autoencoders.values():
                a.eval()
            self.m = self.tr.hand_models[self.hand]
            self.names = action_names(cfg["task"], self.hand)
            self.idx = self.m.data_index(self.names)
            self.wrist = load_wrists(cfg["wrist"], {self.hand: self.m})[self.hand]
            tl = cfg.get("task_onehot")
            self.onehot = None if not tl else np.asarray([float(t == cfg["task"]) for t in tl], np.float32)
            if self.onehot is not None and self.onehot.sum() != 1:
                raise ValueError(f"task {cfg['task']} not in task_onehot {tl}")
        self.ep, self.log = -1, None

    def _flush(self):
        if self.log and self.cfg.get("log_dir"):
            os.makedirs(self.cfg["log_dir"], exist_ok=True)
            np.savez(f"{self.cfg['log_dir']}/ep{self.ep:03d}.npz", **{k: np.asarray(v) for k, v in self.log.items()})

    def reset(self):
        self._flush()
        self.policy.reset()
        self.pre.reset()
        self.post.reset()
        self.ep, self.log = self.ep + 1, {"policy_out": [], "action": [], "new_chunk": []}

    def _encode_state(self, q: np.ndarray) -> np.ndarray:
        from dexverse_xlvla.eval_latent import enc
        n = len(self.names)
        wrist = self.wrist.vec_from_joints(q[None, :6])[0]
        fingers = self.m.angles_to_normalized(torch.tensor(q[None, :n][:, self.idx], dtype=torch.float32)).clamp(-1, 1)
        with torch.no_grad():
            z = enc(self.tr, self.hand, fingers)[0].numpy()
        v = np.concatenate([wrist, z]).astype(np.float32)
        return v if self.onehot is None else np.concatenate([v, self.onehot])

    def act(self, message: dict) -> np.ndarray:
        batch, q = {}, None
        for key, (shape, dtype, data) in message.items():
            v = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
            if key == "observation.state":
                q = v.astype(np.float64)
                if self.latent:
                    v = self._encode_state(q)
            t = torch.from_numpy(v)
            batch[key] = t.permute(2, 0, 1).float() / 255.0 if t.ndim == 3 else t.float()
        new_chunk = len(self.policy._queues["action"]) == 0
        with torch.inference_mode():
            out = self.post(self.policy.select_action(self.pre(batch)))[0].cpu().numpy().astype(np.float64)
        if self.latent:
            from dexverse_xlvla.eval_latent import dec
            act = np.zeros(len(self.names))
            act[:6] = self.wrist.actions_from_vec(out[None, :9], q_init=q[:6])[0]   # q = observed joints
            with torch.no_grad():
                qn = dec(self.tr, self.hand, torch.tensor(out[None, 9:], dtype=torch.float32))[0].clamp(-1, 1)
            lo, hi = self.m._lower, self.m._upper
            act[self.idx] = ((qn + 1) * 0.5 * (hi - lo) + lo).numpy()
        else:
            act = out
        self.log["policy_out"].append(out)
        self.log["action"].append(act)
        self.log["new_chunk"].append(new_chunk)
        return act.astype(np.float32)

    def close(self):
        self._flush()
