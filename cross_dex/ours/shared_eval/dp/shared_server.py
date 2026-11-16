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
batched pass. Noise comes from a per-session generator (initial sample and every DDPM step, row by row), so a
session's actions do not depend on which other sessions share its batch.

  python shared_server.py --ckpt <pretrained_model> --port 7700 --authkey <hex> [--sessions N] [--log calls.jsonl]
Serving loop, gathering and timing log: ../batch_server.py (shared with the pi0.5 server).
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ours/shared_eval (batch_server: stdlib + numpy)
from batch_server import BatchServer, add_server_args, fresh_seed  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("--ckpt", required=True, help="pretrained_model folder of a checkpoint.")
add_server_args(parser, gather_ms=5.0, max_batch=16)
parser.add_argument("--generator_device", default="cuda",
                    help="Device of the per-session noise generators. cuda: no host->device copy per DDPM step, and B=1 "
                         "draws exactly what LeRobot's select_action draws after torch.cuda.manual_seed(seed); cpu: "
                         "~0.1 s slower per chunk on amp7 (100 copies). Either way each row uses only its own generator.")
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


def to_batch(message: dict) -> dict:  # = policy_server.to_batch
    batch = {}
    for key, (shape, dtype, data) in message.items():
        value = torch.from_numpy(np.frombuffer(data, dtype=dtype).reshape(shape).copy())
        batch[key] = value.permute(2, 0, 1).float() / 255.0 if value.ndim == 3 else value.float()
    return batch


class DPBackend:
    """Per session: own DP observation/action queues (DiffusionPolicy.select_action) and own noise generator."""

    def opened(self, s):
        s.queues, s.generator, s.chunk_obs = None, None, None

    def reset(self, s, message):
        policy.reset()  # fresh deques of the right maxlen, then keep them on the session
        s.queues, policy._queues = policy._queues, None
        seed = message.get("seed") if isinstance(message, dict) else None
        if seed is None:
            seed = fresh_seed()
        s.generator = torch.Generator(device=args.generator_device).manual_seed(int(seed))
        return "ok"

    def observe(self, s, message) -> bool:
        """DiffusionPolicy.select_action up to the model call; True if a new chunk is needed."""
        with torch.inference_mode():
            batch = dict(preprocessor(to_batch(message)))
            batch.pop(ACTION, None)
            if IMAGES:
                batch[IMAGES] = torch.stack([batch[key] for key in CFG.image_features], dim=-4)
            populate_queues(s.queues, batch)
            if len(s.queues[ACTION]):
                return False
            s.chunk_obs = {k: torch.stack(list(s.queues[k]), dim=1) for k in batch if k in s.queues}
            return True

    @torch.inference_mode()
    def forward(self, sessions):
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

    def reply(self, s):
        action = postprocessor(s.queues[ACTION].popleft())
        return action[0].cpu().numpy().astype(np.float32).tobytes()

    def synchronize(self):
        torch.cuda.synchronize()

    def memory(self):
        return {"max_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
                "max_reserved_mb": torch.cuda.max_memory_reserved() / 2**20}


BatchServer(DPBackend(), args).serve()
