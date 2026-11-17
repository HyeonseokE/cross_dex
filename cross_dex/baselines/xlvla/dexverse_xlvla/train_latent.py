"""Step 2: train the XL-VLA hand latent on the DexVerse hands (default all six, --hands for a subset) with DexLatent's
own trainer.

  python -m dexverse_xlvla.train_latent --variant code --run_dir <run> --ckpt_dir <dir> [--hands shadow,inspire,sharpa,wuji]
         [--monitor_every 500 --monitor_tasks <t1,t2,...>]     (checks during training, see latent_monitor.py)

Variants (see the bundle README "비교군과 다른 점"):
  code   DexLatent defaults: 50% pinch templates, lambda_kl 0, z = mean.
  paper  paper Sec. 3.2: random joint configurations only, beta 1e-5, z sampled by reparameterisation in training.

DexVerse changes, both variants (bundle README "비교군 구현"): hand latent = finger joints only (arm_dof 0); fingertips
in the hand-common canonical frame (dexverse_xlvla.hands); pinch templates = thumb-finger CONTACT configurations found
by minimising the thumb-finger distance from random starts (the official fixed pinch point in the wrist frame does not
fit every DexVerse hand: Allegro / Inspire stay 46-69 mm apart, step-1 check).
Writes metrics.csv (step,split,metric,value), config.yaml, the final checkpoint and pinch_templates.json.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import torch
from torch.nn import functional as F

import HandLatent.model as M
from dexverse_xlvla import hands as H

sys.path.insert(0, "/workspace/experiments")
from seed_control import DEFAULT_SEED, seed_everything  # noqa: E402


class SampledHandAutoencoder(M.HandAutoencoder):
    """Paper Sec. 3.2.1: z ~ N(mu, sigma) by reparameterisation while training, z = mu otherwise."""

    def forward(self, qpos):
        latent_arm, mean_hand, logvar_hand = self.encode(qpos)
        latent_hand = self.reparameterize(mean_hand, logvar_hand) if self.training else mean_hand
        qpos_arm, qpos_hand = self.decode_from_latents(latent_arm, latent_hand)
        return latent_arm, latent_hand, qpos_arm, qpos_hand, (mean_hand, logvar_hand)


class DexVerseTrainer(M.CrossEmbodimentTrainer):
    template_iters = 200
    template_lr = 0.05

    def _cache_pinch_templates(self) -> None:
        """Thumb-finger contact templates: for finger k, minimise |tip_0 - tip_k|^2 from random starts (other joints
        kept near the start for diversity). Same count / joint noise as DexLatent."""
        self.template_stats = {}
        count = int(self.config.pinch_template_count)
        for name in self.hand_names:
            model = self.hand_models[name]
            dof, tips = model.dof_count(), model.tip_count()
            per = -(-count // (tips - 1))
            sols, stats = [], {}
            for k in range(1, tips):
                start = torch.empty(per, dof).uniform_(-0.8, 0.8)
                u = torch.atanh(start.clamp(-0.999, 0.999)).clone().requires_grad_(True)
                opt = torch.optim.Adam([u], lr=self.template_lr)
                for _ in range(self.template_iters):
                    opt.zero_grad()
                    q = torch.tanh(u)
                    t = model.forward(q)
                    loss = ((t[:, 0] - t[:, k]) ** 2).sum(-1).mean() + 1e-4 * ((q - start) ** 2).sum(-1).mean()
                    loss.backward()
                    opt.step()
                with torch.no_grad():
                    q = torch.tanh(u)
                    d = (model.forward(q)[:, 0] - model.forward(q)[:, k]).norm(dim=-1) * 1000
                stats[f"0-{k}"] = {"median_mm": round(float(d.median()), 1), "p90_mm": round(float(d.quantile(0.9)), 1)}
                sols.append(q.detach())
            self._pinch_templates[name] = torch.cat(sols)[:count].cpu()
            self.template_stats[name] = stats
            print(f"[templates] {name}: {stats}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["code", "paper"], required=True)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--num_steps", type=int, default=1000)   # DexLatent train.py default (= official checkpoint)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--template_count", type=int, default=2048)   # DexLatent default; smaller only for tests
    ap.add_argument("--hands", default=",".join(H.HAND_NAMES),   # subset, e.g. the five-finger hands
                    help="comma-separated hands (default: all six)")
    ap.add_argument("--monitor_every", type=int, default=0,
                    help="save checkpoint_epoch_<step>.pt every N steps and run checks 1-2 on it (dexverse_xlvla."
                         "latent_monitor, background CPU process); check 3 (replay_monitor.sh) picks the checkpoints up")
    ap.add_argument("--monitor_tasks", default="Dexverse-GraspKettle-v0", help="comma-separated tasks for the checks")
    ap.add_argument("--resume", default="",
                    help="checkpoint_epoch_<step>.pt of the same setup: load model + optimizer and continue from step+1 "
                         "(constant lr, no scheduler; the data RNG restarts from --seed)")
    return ap


def main(argv=None, parser=None, trainer_cls=None, setup=None) -> None:
    """XL-VLA training as reproduced here. Variants built on it (``cross_dex.ours.*``) pass their own parser (built from
    ``build_parser()``), trainer subclass and ``setup(trainer, args) -> {extra config entries}``; a trainer attribute
    ``extra_metrics`` (dict) is logged with the trainer's own metrics."""
    ap = parser or build_parser()
    trainer_cls = trainer_cls or DexVerseTrainer
    args = ap.parse_args(argv)
    names = args.hands.split(",")
    bad = [h for h in names if h not in H.HANDS]          # right (HAND_NAMES) or left (LEFT_HAND_NAMES, bimanual)
    if bad or len(names) < 2:
        ap.error(f"--hands: unknown {bad} or fewer than two hands (known: {list(H.HANDS)})")

    seed_everything(args.seed)
    H.install()
    paper = args.variant == "paper"
    if paper:
        M.HandAutoencoder = SampledHandAutoencoder
    cfg = M.TrainingConfig(
        num_steps=args.num_steps, checkpoint_interval=0, checkpoint_dir=args.ckpt_dir, arm_dof=0,
        pinch_offset=(0.0, 0.0, 0.0),
        pinch_sampling_probability=0.0 if paper else 0.5,
        lambda_kl=1e-5 if paper else 0.0, pinch_template_count=args.template_count,
    )
    tr = trainer_cls(names, cfg)
    extra_conf = setup(tr, args) if setup else {}
    tr.checkpoint_dir = args.ckpt_dir
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.run_dir, exist_ok=True)
    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=cfg.device)
        if list(ck["hand_names"]) != names:
            ap.error(f"--resume: hands {ck['hand_names']} != {names}")
        tr.autoencoders.load_state_dict(ck["autoencoders"])
        tr.optimizer.load_state_dict(ck["optimizer"])
        start = int(ck["epoch"])
        print(f"[resume] {args.resume} -> continue from step {start + 1}", flush=True)
        extra_conf = {**extra_conf, "resume_from": args.resume, "resume_step": start}

    conf = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(cfg).items()}
    conf.update(device=str(cfg.device), variant=args.variant, hands=names, seed=args.seed,
                dof={h: tr.hand_dof_per_hand[h] for h in names}, tips={h: tr.hand_tip_count(h) for h in names},
                z_sampling="reparameterize (train)" if paper else "mean",
                pinch_templates="thumb-finger contact by distance minimisation" if not paper else "not used",
                template_iters=DexVerseTrainer.template_iters, **extra_conf)
    with open(os.path.join(args.run_dir, "config.yaml"), "w") as f:
        for k, v in conf.items():
            f.write(f"{k}: {json.dumps(v)}\n")

    t0 = time.time()
    if not paper:
        tr._cache_pinch_templates()
        torch.save(tr._pinch_templates, os.path.join(args.ckpt_dir, "pinch_templates.pt"))
        json.dump(tr.template_stats, open(os.path.join(args.run_dir, "pinch_templates.json"), "w"), indent=1)
    for a in tr.autoencoders.values():
        a.train()
    with open(os.path.join(args.run_dir, "metrics.csv"), "w") as mf:     # resume: a new run dir (rows from start + 1)
        mf.write("step,split,metric,value\n")
        monitors = []
        for step in range(start + 1, args.num_steps + 1):
            m = tr.step()
            if step % args.log_every == 0 or step == start + 1:
                m = {**m, **getattr(tr, "extra_metrics", {})}
                for k, v in m.items():
                    mf.write(f"{step},train,{k},{v:.6g}\n")
                mf.flush()
                print(f"step {step:5d} " + " ".join(f"{k}={v:.4f}" for k, v in m.items()), flush=True)
            if args.monitor_every and step % args.monitor_every == 0 and step < args.num_steps:
                monitors.append(monitor(tr.save_checkpoint(step), args))
    path = tr.save_checkpoint(args.num_steps)
    print(f"[done] {time.time() - t0:.0f}s checkpoint {path}", flush=True)
    if args.monitor_every:
        monitors.append(monitor(path, args))
        bad = [p.args[-1] for p in monitors if p.wait() != 0]
        print(f"[monitor] {len(monitors) - len(bad)}/{len(monitors)} checkpoints checked" + (f", FAILED {bad}" if bad else ""),
              flush=True)
        mon = os.path.join(args.run_dir, "artifacts", "monitor", "metrics.csv")   # eval rows after the train rows
        if os.path.exists(mon):
            with open(os.path.join(args.run_dir, "metrics.csv"), "a") as mf:
                mf.writelines(sorted(open(mon).readlines()[1:], key=lambda r: int(r.split(",", 1)[0])))


def monitor(ckpt: str, args) -> subprocess.Popen:
    """Checks 1-2 on a saved checkpoint in a separate CPU process (own RNG; training continues)."""
    log = os.path.join(args.run_dir, "logs", "monitor.log")
    os.makedirs(os.path.dirname(log), exist_ok=True)
    return subprocess.Popen([sys.executable, "-m", "dexverse_xlvla.latent_monitor", "--run_dir", args.run_dir,
                             "--tasks", args.monitor_tasks, "--ckpt", ckpt],
                            stdout=open(log, "a"), stderr=subprocess.STDOUT, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})


if __name__ == "__main__":
    main()
