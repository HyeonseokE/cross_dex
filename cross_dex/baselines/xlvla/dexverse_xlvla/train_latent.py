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


class LocalLossOptimizer:
    """Wraps the trainer's optimizer: just before each step, adds the gradient of the local-consistency loss, so the
    original CrossEmbodimentTrainer.step (DexLatent, unchanged) trains loss + w_local * local loss."""

    def __init__(self, inner, extra):
        self.inner, self.extra = inner, extra

    def zero_grad(self, *a, **k):
        return self.inner.zero_grad(*a, **k)

    def step(self, *a, **k):
        loss = self.extra()
        if loss is not None:
            loss.backward()
        return self.inner.step(*a, **k)

    def __getattr__(self, name):
        return getattr(self.inner, name)


class DexVerseTrainer(M.CrossEmbodimentTrainer):
    template_iters = 200
    template_lr = 0.05
    w_local, local_deg, local_batch = 0.0, 1.0, 1024
    last_local: dict = {}

    def enable_local(self, w: float, deg: float, batch: int) -> None:
        """Local-consistency loss (2026-11-16, user: "2번만 추가해서 한번 해보자"; ours, not the XL-VLA paper).
        Uniform pose q per hand (XL-VLA's pose generator) and q + d, d a random direction with RMS `deg` degrees over the
        joints. self: the decoded change on the same hand reproduces d; cross: the change of hand B decoded from hand A's
        latents moves B's thumb->finger fingertip lines like A's. Both relative to the input change (scale free)."""
        self.w_local, self.local_deg, self.local_batch = w, deg, batch
        self.optimizer = LocalLossOptimizer(self.optimizer, self._local_loss)

    def _local_loss(self):
        if not self.w_local:
            return None
        dev = self.config.device
        enc = lambda h, x: self.autoencoders[h].encode(x)[1]                                  # posterior mean
        dec = lambda h, z: self.autoencoders[h].decode_from_latents(z[:, :0], z)[1]
        q, q2, z, z2, lines = {}, {}, {}, {}, {}
        rel_self = []
        for h in self.hand_names:
            m = self.hand_models[h]
            half = (0.5 * (m._upper - m._lower)).to(dev).clamp_min(1e-6)
            x = torch.empty(self.local_batch, m.dof_count(), device=dev).uniform_(-1, 1)
            d = torch.randn_like(x)
            d = d / d.pow(2).mean(-1, keepdim=True).sqrt() * torch.deg2rad(torch.tensor(self.local_deg, device=dev)) / half
            x2 = (x + d).clamp(-1, 1)
            q[h], q2[h] = x, x2
            z[h], z2[h] = enc(h, x), enc(h, x2)
            dout = dec(h, z2[h]) - dec(h, z[h])
            din = x2 - x
            rel_self.append((dout - din).pow(2).sum(-1).mean() / din.pow(2).sum(-1).mean().clamp_min(1e-12))
            t1, t2 = m.forward(x), m.forward(x2)
            lines[h] = (t2[:, 1:] - t2[:, :1]) - (t1[:, 1:] - t1[:, :1])                           # change of thumb->finger lines
        rel_cross = []
        for s in self.hand_names:
            for t in self.hand_names:
                if s == t:
                    continue
                mt = self.hand_models[t]
                a1, a2 = mt.forward(dec(t, z[s]).clamp(-1, 1)), mt.forward(dec(t, z2[s]).clamp(-1, 1))
                dt = (a2[:, 1:] - a2[:, :1]) - (a1[:, 1:] - a1[:, :1])
                k = min(dt.shape[1], lines[s].shape[1])
                ds = lines[s][:, :k].detach()
                rel_cross.append((dt[:, :k] - ds).pow(2).sum(-1).mean() / ds.pow(2).sum(-1).mean().clamp_min(1e-12))
        ls, lc = sum(rel_self) / len(rel_self), sum(rel_cross) / len(rel_cross)
        self.last_local = {"local_self": float(ls), "local_cross": float(lc)}
        return self.w_local * (ls + lc)

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


def main() -> None:
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
    ap.add_argument("--w_local", type=float, default=0.0,
                    help="local-consistency loss weight (ours, 2026-11-16; 0 = XL-VLA as published): small pose changes kept "
                         "on the same hand and carried to the other hands (DexVerseTrainer.enable_local)")
    ap.add_argument("--local_deg", type=float, default=1.0, help="RMS size of the pose change d (deg)")
    ap.add_argument("--local_batch", type=int, default=1024)
    args = ap.parse_args()
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
    tr = DexVerseTrainer(names, cfg)
    if args.w_local:
        tr.enable_local(args.w_local, args.local_deg, args.local_batch)
    tr.checkpoint_dir = args.ckpt_dir
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.run_dir, exist_ok=True)

    conf = {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(cfg).items()}
    conf.update(device=str(cfg.device), variant=args.variant, hands=names, seed=args.seed,
                w_local=args.w_local, local_deg=args.local_deg, local_batch=args.local_batch,
                dof={h: tr.hand_dof_per_hand[h] for h in names}, tips={h: tr.hand_tip_count(h) for h in names},
                z_sampling="reparameterize (train)" if paper else "mean",
                pinch_templates="thumb-finger contact by distance minimisation" if not paper else "not used",
                template_iters=DexVerseTrainer.template_iters)
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
    with open(os.path.join(args.run_dir, "metrics.csv"), "w") as mf:
        mf.write("step,split,metric,value\n")
        monitors = []
        for step in range(1, args.num_steps + 1):
            m = tr.step()
            if step % args.log_every == 0 or step == 1:
                m = {**m, **tr.last_local}
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
