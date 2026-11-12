"""Serve the chunk-latent Diffusion Policy to the common evaluator (through dp_server.py / dp_run_eval.py).

server.json: {"mode": "chunk_latent_dp", "hand": ..., "policy_ckpt": <LeRobot pretrained_model>, "latent_ckpt": <best.pt>,
  "xlvla_ckpt": <XL-VLA latent .pt (wrist hand models)>, "wrist": <wrist_calib_<task>.json>, "task": ...,
  "task_onehot": [...] (optional), "noise_scheduler_type" / "num_inference_steps" (optional), "log_dir": <dir or null>}

Per step: observation.state (joint_pos, evaluator order = action order first) -> [wrist pose | mean_k E_h(static finger
chunk) | one-hot] -> LeRobot DP (unchanged) -> [wrist pose | Z]. When the DP starts a new chunk (its action queue was
empty), Z of that first step is decoded to the H-step finger chunk D_h(Z); the following n_action_steps steps take
their fingers from that chunk in order (wrist from each predicted step, IK warm-started at the observed wrist joints).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch


def _server() -> str:
    return next(d for d in sorted(os.listdir("/workspace/experiments"))
                if os.path.isdir(f"/workspace/experiments/{d}") and d[0] not in "_.")


COMMON_DP = os.environ.get("XLVLA_COMMON_DP") or \
    f"/workspace/experiments/{_server()}/20261031_teleop_single/teleop_dp_image"   # image_resize (called, not changed)


class ChunkLatentPolicyServer:
    def __init__(self, cfg: dict):
        sys.path.insert(0, COMMON_DP)
        from image_resize import install_resize_for
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from lerobot.policies.factory import make_pre_post_processors

        from cross_dex.latent.chunk_latent.model import HandCodec, load_model
        from dexverse_xlvla.eval_latent import load
        from dexverse_xlvla.replay import action_names
        from dexverse_xlvla.wrist import load_wrists

        self.cfg, self.dev = cfg, torch.device("cuda")
        pcfg = None
        if cfg.get("noise_scheduler_type") or cfg.get("num_inference_steps"):
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
        self.hand = cfg["hand"]
        model, ck = load_model(cfg["latent_ckpt"], self.dev)
        self.codec = HandCodec(model, ck, self.hand, self.dev)
        self.names = action_names(cfg["task"], self.hand)
        self.fidx = [self.names.index(n) for n in ck["finger_joint_names"][self.hand]]
        tr = load(cfg["xlvla_ckpt"])
        self.wrist = load_wrists(cfg["wrist"], {self.hand: tr.hand_models[self.hand]})[self.hand]
        tl = cfg.get("task_onehot")
        self.onehot = None if not tl else np.asarray([float(t == cfg["task"]) for t in tl], np.float32)
        if self.onehot is not None and self.onehot.sum() != 1:
            raise ValueError(f"task {cfg['task']} not in task_onehot {tl}")
        self.ep, self.log, self.chunk, self.k = -1, None, None, 0

    def _flush(self):
        if self.log and self.cfg.get("log_dir"):
            os.makedirs(self.cfg["log_dir"], exist_ok=True)
            np.savez(f"{self.cfg['log_dir']}/ep{self.ep:03d}.npz", **{k: np.asarray(v) for k, v in self.log.items()})

    def reset(self):
        self._flush()
        self.policy.reset()
        self.pre.reset()
        self.post.reset()
        self.ep, self.log, self.chunk, self.k = self.ep + 1, {"policy_out": [], "action": [], "new_chunk": []}, None, 0

    def _encode_state(self, q: np.ndarray) -> np.ndarray:
        wrist = self.wrist.vec_from_joints(q[None, :6])[0]
        f = np.repeat(q[None, self.fidx][:, None], self.codec.H, 1)
        zs = self.codec.encode(f).mean(1)[0].cpu().numpy()
        v = np.concatenate([wrist, zs]).astype(np.float32)
        return v if self.onehot is None else np.concatenate([v, self.onehot])

    def act(self, message: dict) -> np.ndarray:
        batch, q = {}, None
        for key, (shape, dtype, data) in message.items():
            v = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
            if key == "observation.state":
                q = v.astype(np.float64)
                v = self._encode_state(q)
            t = torch.from_numpy(v)
            batch[key] = t.permute(2, 0, 1).float() / 255.0 if t.ndim == 3 else t.float()
        new_chunk = len(self.policy._queues["action"]) == 0
        with torch.inference_mode():
            out = self.post(self.policy.select_action(self.pre(batch)))[0].cpu().numpy().astype(np.float64)
        if new_chunk or self.chunk is None:
            z = out[9:].reshape(1, self.codec.K, self.codec.d)
            self.chunk, self.k = self.codec.decode(z)[0].cpu().numpy(), 0
        act = np.zeros(len(self.names))
        act[:6] = self.wrist.actions_from_vec(out[None, :9], q_init=q[:6])[0]
        act[self.fidx] = self.chunk[min(self.k, len(self.chunk) - 1)]
        self.k += 1
        self.log["policy_out"].append(out)
        self.log["action"].append(act)
        self.log["new_chunk"].append(new_chunk)
        return act.astype(np.float32)

    def close(self):
        self._flush()
