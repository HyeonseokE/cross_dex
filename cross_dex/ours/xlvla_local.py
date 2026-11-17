"""Ours, not the XL-VLA paper (2026-11-16, user: "2번만 추가해서 한번 해보자"): XL-VLA hand latent + a local-consistency
loss, so that small action changes survive the latent (fine dexterous tasks; adopted metric 10 "resolution").

Built on the reproduction ``dexverse_xlvla.train_latent`` (cross_dex/baselines/xlvla, DexLatent's trainer unchanged): its
trainer is subclassed and its optimizer wrapped, so CrossEmbodimentTrainer.step trains its own loss + w_local * local loss.

Local-consistency loss: every step, per hand, a uniform pose q in the joint limits (XL-VLA's pose generator) and q + d,
d a random direction with RMS ``--local_deg`` degrees over the joints.
  self   the decoded change on the same hand reproduces d:            |dec(enc(q+d)) - dec(enc(q)) - d|^2 / |d|^2
  cross  hand B decoded from hand A's latents moves B's thumb->finger fingertip lines like A's lines move (relative error)

  cd /workspace/codes/dexverse/DexLatent && PYTHONPATH=/workspace/codes/cross_dex \
      python -m cross_dex.ours.xlvla_local --variant paper --run_dir <run> --ckpt_dir <dir> --num_steps 30000 --w_local 1
(every option of dexverse_xlvla.train_latent works the same; checkpoints load with the reproduction's code)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.environ.get("DEXLATENT_ROOT", "/workspace/codes/dexverse/DexLatent"))

import torch  # noqa: E402

from dexverse_xlvla import train_latent as TL  # noqa: E402
from cross_dex.ours import fast_xl as FX  # noqa: E402


class LocalLossOptimizer:
    """Wraps the trainer's optimizer: just before each step it adds the gradient of the extra loss."""

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


class LocalTrainer(FX.FastXLMixin, TL.DexVerseTrainer):
    w_local, local_deg, local_batch = 0.0, 1.0, 1024

    def enable_local(self, w: float, deg: float, batch: int) -> None:
        self.w_local, self.local_deg, self.local_batch = w, deg, batch
        self.extra_metrics = {}
        self.optimizer = LocalLossOptimizer(self.optimizer, self._local_loss)

    def _local_loss(self):
        if not self.w_local:
            return None
        dev = self.config.device
        enc = lambda h, x: self.autoencoders[h].encode(x)[1]                                  # posterior mean
        dec = lambda h, z: self.autoencoders[h].decode_from_latents(z[:, :0], z)[1]
        z, z2, lines, rel_self = {}, {}, {}, []
        for h in self.hand_names:
            m = self.hand_models[h]
            half = (0.5 * (m._upper - m._lower)).to(dev).clamp_min(1e-6)
            x = torch.empty(self.local_batch, m.dof_count(), device=dev).uniform_(-1, 1)
            d = torch.randn_like(x)
            d = d / d.pow(2).mean(-1, keepdim=True).sqrt() * torch.deg2rad(torch.tensor(self.local_deg, device=dev)) / half
            x2 = (x + d).clamp(-1, 1)
            z[h], z2[h] = enc(h, x), enc(h, x2)
            dout, din = dec(h, z2[h]) - dec(h, z[h]), x2 - x
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
        self.extra_metrics = {"local_self": float(ls), "local_cross": float(lc)}
        return self.w_local * (ls + lc)


def main(argv=None) -> None:
    ap = TL.build_parser()
    ap.add_argument("--w_local", type=float, default=1.0, help="local-consistency loss weight")
    ap.add_argument("--local_deg", type=float, default=1.0, help="RMS size of the pose change d (deg)")
    ap.add_argument("--local_batch", type=int, default=1024)
    ap.add_argument("--fast", type=int, default=1, help="1 = cross_dex.ours.fast_xl (batched FK / decoding, same maths)")

    def setup(tr, args):
        if args.fast:
            FX.make_fast(tr)
            tr.fast = True
            tr._local_loss = lambda: FX.fast_local_loss(tr)
        tr.enable_local(args.w_local, args.local_deg, args.local_batch)
        return {"method": "ours: XL-VLA + local consistency (cross_dex.ours.xlvla_local)", "w_local": args.w_local,
                "local_deg": args.local_deg, "local_batch": args.local_batch, "fast": args.fast}

    TL.main(argv, parser=ap, trainer_cls=LocalTrainer, setup=setup)


if __name__ == "__main__":
    main()
