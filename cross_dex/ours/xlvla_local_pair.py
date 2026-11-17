"""Ours (2026-11-17): XL-VLA hand latent + local consistency (cross_dex.ours.xlvla_local) + pair alignment on
hand-to-hand pairs that cover the whole joint range (cross_dex.ours.uniform_pairs: XL-VLA's uniform poses retargeted
with DexPilot's objective).

  L = L_XL + w_local (L_self + L_cross) + w_pair (L_CR + L_NCE)
  L_CR   cross reconstruction  |dec_B(enc_A(q_A)) - q_B|^2 over ordered hand pairs (normalized joints, as L_rec)
  L_NCE  InfoNCE between enc_A(q_A) and enc_B(q_B) over unordered hand pairs (cosine / tau, both directions)

A pair (A, B) of one tuple counts only when both retargeting residuals are <= --pair_max_res_mm (the source hand of the
tuple has residual 0). Each step draws --pair_batch tuples uniformly from all source hands.

  cd /workspace/codes/dexverse/DexLatent && PYTHONPATH=/workspace/codes/cross_dex \
      python -m cross_dex.ours.xlvla_local_pair --variant paper --run_dir <run> --ckpt_dir <dir> --num_steps 30000 \
      --w_local 1 --pairs <uniform_pairs.npz> --w_pair 1
"""
from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from cross_dex.ours import xlvla_local as XLL
from dexverse_xlvla import train_latent as TL


class PairTrainer(XLL.LocalTrainer):
    w_pair = 0.0

    def enable_pair(self, path: str, w: float, batch: int, tau: float, max_res: float) -> dict:
        self.w_pair, self.pair_batch, self.tau = w, batch, tau
        dev = self.config.device
        d = np.load(path)
        srcs = [s for s in self.hand_names if f"{s}__{s}" in d.files]
        q = {h: [] for h in self.hand_names}
        ok = {h: [] for h in self.hand_names}
        for s in srcs:
            for h in self.hand_names:
                q[h].append(torch.from_numpy(d[f"{s}__{h}"]))
                ok[h].append(torch.ones(len(d[f"{s}__{s}"]), dtype=torch.bool) if h == s
                             else torch.from_numpy(d[f"{s}__{h}__res_mm"] <= max_res))
        self.pair_q = {h: torch.cat(v).to(dev) for h, v in q.items()}
        self.pair_ok = {h: torch.cat(v).to(dev) for h, v in ok.items()}
        self.pair_n = len(self.pair_q[self.hand_names[0]])
        info = {h: round(100 * float(v.float().mean()), 1) for h, v in self.pair_ok.items()}
        print(f"[pairs] {path}: {self.pair_n} tuples from {srcs}; usable % per hand {info}", flush=True)
        inner = self.optimizer.extra                       # LocalLossOptimizer's extra loss -> local + pair
        self.optimizer.extra = lambda: self._sum(inner(), self._pair_loss())
        return {"pair_tuples": self.pair_n, "pair_sources": srcs, "pair_usable_pct": info}

    @staticmethod
    def _sum(a, b):
        return b if a is None else (a if b is None else a + b)

    def _pair_loss(self):
        if not self.w_pair:
            return None
        idx = torch.randint(0, self.pair_n, (self.pair_batch,), device=self.config.device)
        z, q, ok = {}, {}, {}
        for h in self.hand_names:
            q[h], ok[h] = self.pair_q[h][idx], self.pair_ok[h][idx]
            z[h] = self.autoencoders[h].encode(q[h])[1]                                 # posterior mean
        cr, nce, n_cr, n_nce = 0.0, 0.0, 0, 0
        names = self.hand_names
        for i, a in enumerate(names):
            for b in names:
                if a == b:
                    continue
                m = ok[a] & ok[b]
                if m.sum() < 2:
                    continue
                pred = self.autoencoders[b].decode_from_latents(z[a][:, :0], z[a])[1]
                cr = cr + F.mse_loss(pred[m], q[b][m]); n_cr += 1
                if names.index(b) > i:
                    za, zb = F.normalize(z[a][m], dim=-1), F.normalize(z[b][m], dim=-1)
                    logits = za @ zb.T / self.tau
                    lab = torch.arange(len(za), device=za.device)
                    nce = nce + 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)); n_nce += 1
        cr, nce = cr / max(n_cr, 1), nce / max(n_nce, 1)
        self.extra_metrics = {**getattr(self, "extra_metrics", {}), "pair_cr": float(cr), "pair_nce": float(nce)}
        return self.w_pair * (cr + nce)


def main(argv=None) -> None:
    ap = TL.build_parser()
    ap.add_argument("--w_local", type=float, default=1.0)
    ap.add_argument("--local_deg", type=float, default=1.0)
    ap.add_argument("--local_batch", type=int, default=1024)
    ap.add_argument("--pairs", required=True, help="cross_dex.ours.uniform_pairs output (.npz)")
    ap.add_argument("--w_pair", type=float, default=1.0)
    ap.add_argument("--pair_batch", type=int, default=1024)
    ap.add_argument("--pair_tau", type=float, default=0.1)
    ap.add_argument("--pair_max_res_mm", type=float, default=15.0)

    def setup(tr, args):
        tr.enable_local(args.w_local, args.local_deg, args.local_batch)
        info = tr.enable_pair(args.pairs, args.w_pair, args.pair_batch, args.pair_tau, args.pair_max_res_mm)
        return {"method": "ours: XL-VLA + local consistency + pair alignment (cross_dex.ours.xlvla_local_pair)",
                "w_local": args.w_local, "local_deg": args.local_deg, "local_batch": args.local_batch,
                "pairs": args.pairs, "w_pair": args.w_pair, "pair_batch": args.pair_batch, "pair_tau": args.pair_tau,
                "pair_max_res_mm": args.pair_max_res_mm, **info}

    TL.main(argv, parser=ap, trainer_cls=PairTrainer, setup=setup)


if __name__ == "__main__":
    main()
