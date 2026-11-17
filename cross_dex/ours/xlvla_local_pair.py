"""Ours (2026-11-17): XL-VLA hand latent + local consistency (cross_dex.ours.xlvla_local) + pair alignment on
hand-to-hand pairs that cover the whole joint range, generated DURING training (user: online, like XL-VLA's sampler).

  L = L_XL + w_local (L_self + L_cross) + w_pair (L_CR + L_NCE)
  L_CR   cross reconstruction  |dec_B(enc_A(q_A)) - q_B|^2, both directions of every hand pair (normalized joints)
  L_NCE  InfoNCE between enc_A(q_A) and enc_B(q_B) (cosine / tau, both directions), LAD-size batch (16384)

Pair buffer (online with refresh): pair retargeting costs ~0.26 s per optimizer iteration for any batch up to ~40k
(fixed per-iteration FK cost, measured a4 2026-11-17), so instead of retargeting every step, every --refresh_every steps
--refresh_n new uniform poses of one source hand (round robin) are retargeted to the other hands with DexPilot's
objective (cross_dex.ours.uniform_pairs.retarget) and replace the oldest buffer entries (ring). The buffer starts from
--pairs (uniform_pairs output). A hand pair of one tuple counts only when both residuals are <= --pair_max_res_mm.
The pair losses are back-propagated per hand pair inside the step (re-encoding each time) so the 16384 x 16384 logits
of one hand pair at a time are kept, not of all 15.

  cd /workspace/codes/dexverse/DexLatent && PYTHONPATH=/workspace/codes/cross_dex \
      python -m cross_dex.ours.xlvla_local_pair --variant paper --run_dir <run> --ckpt_dir <dir> --num_steps 30000 \
      --w_local 1 --pairs <uniform_pairs.npz> --w_pair 1
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.nn import functional as F

from cross_dex.ours import uniform_pairs as UP
from cross_dex.ours import xlvla_local as XLL
from dexverse_xlvla import train_latent as TL

DP = {"huber_delta": 0.03, "project_dist": 0.03, "eta1": 1e-4, "eta2": 3e-2}   # DexPilotOptimizer defaults


class PairTrainer(XLL.LocalTrainer):
    w_pair = 0.0

    def enable_pair(self, a) -> dict:
        self.pa, self.w_pair = a, a.w_pair
        dev = self.config.device
        d = np.load(a.pairs)
        srcs = [s for s in self.hand_names if f"{s}__{s}" in d.files]
        q = {h: [] for h in self.hand_names}
        res = {h: [] for h in self.hand_names}
        for s in srcs:
            n = len(d[f"{s}__{s}"])
            for h in self.hand_names:
                q[h].append(torch.from_numpy(d[f"{s}__{h}"]))
                res[h].append(torch.zeros(n) if h == s else torch.from_numpy(d[f"{s}__{h}__res_mm"]))
        q = {h: torch.cat(v) for h, v in q.items()}
        res = {h: torch.cat(v) for h, v in res.items()}
        n_all = len(res[self.hand_names[0]])
        size = min(a.pair_buffer, n_all)
        pick = torch.randperm(n_all)[:size]
        self.buf_q = {h: q[h][pick].to(dev) for h in self.hand_names}
        self.buf_ok = {h: (res[h][pick] <= a.pair_max_res_mm).to(dev) for h in self.hand_names}
        self.buf_size, self.buf_ptr, self.pair_calls, self.refresh_src, self.refresh_sec = size, 0, 0, 0, 0.0
        info = self._usable()
        print(f"[pairs] buffer {size} tuples from {a.pairs} ({n_all}, sources {srcs}); usable % {info}", flush=True)
        inner = self.optimizer.extra
        self.optimizer.extra = lambda: (self._pair_backward(), inner())[1]
        return {"pair_init_tuples": n_all, "pair_buffer": size, "pair_init_usable_pct": info}

    def _usable(self):
        return {h: round(100 * float(v.float().mean()), 1) for h, v in self.buf_ok.items()}

    @torch.enable_grad()
    def _refresh(self):
        a, dev, t0 = self.pa, self.config.device, time.time()
        s = self.hand_names[self.refresh_src % len(self.hand_names)]
        self.refresh_src += 1
        n = a.refresh_n
        qs = torch.empty(n, self.hand_models[s].dof_count(), device=dev).uniform_(-1, 1)
        new_q, new_ok = {s: qs}, {s: torch.ones(n, dtype=torch.bool, device=dev)}
        for t in self.hand_names:
            if t == s:
                continue
            qt, rms, _ = UP.retarget(self.hand_models[s], self.hand_models[t], qs, a.refresh_iters, a.refresh_starts,
                                     0.05, 1e-6, DP)
            new_q[t], new_ok[t] = qt.detach(), rms <= a.pair_max_res_mm
        idx = (torch.arange(n, device=dev) + self.buf_ptr) % self.buf_size
        for h in self.hand_names:
            self.buf_q[h][idx], self.buf_ok[h][idx] = new_q[h], new_ok[h]
        self.buf_ptr = int((self.buf_ptr + n) % self.buf_size)
        self.refresh_sec += time.time() - t0
        print(f"[refresh] call {self.pair_calls}: {n} new {s} poses -> buffer, {time.time() - t0:.0f}s, "
              f"usable % {self._usable()}", flush=True)

    def _pair_backward(self, nce_only: bool = False, tau: float | None = None):
        if not self.w_pair:
            return
        a = self.pa
        tau = a.pair_tau if tau is None else tau
        self.pair_calls += 1
        if a.refresh_every and self.pair_calls % a.refresh_every == 0:
            self._refresh()
        idx = torch.randint(0, self.buf_size, (a.pair_batch,), device=self.config.device)
        names = self.hand_names
        hp = [(x, y) for i, x in enumerate(names) for y in names[i + 1:]]
        enc = lambda h, m: self.autoencoders[h].encode(self.buf_q[h][idx][m])[1]
        dec = lambda h, z: self.autoencoders[h].decode_from_latents(z[:, :0], z)[1]
        tot_cr, tot_nce, used = 0.0, 0.0, 0
        for x, y in hp:
            m = self.buf_ok[x][idx] & self.buf_ok[y][idx]
            if int(m.sum()) < 2:
                continue
            zx, zy = enc(x, m), enc(y, m)
            cr = (zx.new_zeros(()) if nce_only else
                  0.5 * (F.mse_loss(dec(y, zx), self.buf_q[y][idx][m]) + F.mse_loss(dec(x, zy), self.buf_q[x][idx][m])))
            logits = F.normalize(zx, dim=-1) @ F.normalize(zy, dim=-1).T / tau
            lab = torch.arange(len(zx), device=zx.device)
            nce = 0.5 * (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab))
            (self.w_pair * (cr + nce) / len(hp)).backward()
            tot_cr, tot_nce, used = tot_cr + float(cr), tot_nce + float(nce), used + 1
        self.extra_metrics = {**getattr(self, "extra_metrics", {}), "pair_cr": tot_cr / max(used, 1),
                              "pair_nce": tot_nce / max(used, 1), "pair_hand_pairs": used,
                              "pair_refresh_sec": self.refresh_sec}


class StagedPairTrainer(PairTrainer):
    """E2 = the same losses and data as PairTrainer, split into LAD's two stages (LAD Sec. III, Table II/III "FT"):
    stage 1 (steps 1..stage1_steps): encoders only, pair InfoNCE only, temperature annealed t0 -> pair_tau
    (exponential, as LAD 0.25 -> 0.16); stage 2: every loss of PairTrainer, encoder learning rate x enc_lr_scale."""

    stage1_steps, step_count, stage = 0, 0, 1

    def enable_stages(self, a) -> None:
        self.stage1_steps, self.t0, self.enc_lr_scale = a.stage1_steps, a.stage1_tau0, a.enc_lr_scale

    @staticmethod
    def _is_enc(name: str) -> bool:
        return ".hand_encoder_backbone." in name or ".hand_mean_head." in name or ".hand_logvar_head." in name

    def _enter_stage2(self) -> None:
        old = self.optimizer.inner
        named = list(self.autoencoders.named_parameters())
        enc = [p for n, p in named if self._is_enc("." + n)]
        rest = [p for n, p in named if not self._is_enc("." + n)]
        lr = old.param_groups[0]["lr"]
        new = torch.optim.AdamW([{"params": enc, "lr": lr * self.enc_lr_scale}, {"params": rest, "lr": lr}], lr=lr)
        for p in enc + rest:                                           # keep Adam moments of stage 1
            if p in old.state:
                new.state[p] = old.state[p]
        self.optimizer.inner = new
        self.stage = 2
        print(f"[stage] 2 from step {self.step_count}: all losses, encoder lr x{self.enc_lr_scale} "
              f"({len(enc)} encoder / {len(rest)} other tensors)", flush=True)

    def step(self):
        self.step_count += 1
        if self.step_count <= self.stage1_steps:
            frac = (self.step_count - 1) / max(self.stage1_steps - 1, 1)
            tau = self.t0 * (self.pa.pair_tau / self.t0) ** frac
            self.optimizer.inner.zero_grad(set_to_none=True)
            self._pair_backward(nce_only=True, tau=tau)
            self.optimizer.inner.step()                                # decoders: grad None -> untouched
            m = {"stage": 1.0, "pair_tau": tau, **{k: v for k, v in self.extra_metrics.items() if k.startswith("pair_")}}
            return m
        if self.stage == 1:
            self._enter_stage2()
        return {**super().step(), "stage": 2.0}


def main(argv=None) -> None:
    ap = TL.build_parser()
    ap.add_argument("--w_local", type=float, default=1.0)
    ap.add_argument("--local_deg", type=float, default=1.0)
    ap.add_argument("--local_batch", type=int, default=1024)
    ap.add_argument("--pairs", required=True, help="initial buffer: cross_dex.ours.uniform_pairs output (.npz)")
    ap.add_argument("--w_pair", type=float, default=1.0)
    ap.add_argument("--pair_batch", type=int, default=16384, help="LAD Table II batch")
    ap.add_argument("--pair_tau", type=float, default=0.16, help="LAD's final InfoNCE temperature")
    ap.add_argument("--pair_max_res_mm", type=float, default=15.0)
    ap.add_argument("--pair_buffer", type=int, default=200000)
    ap.add_argument("--refresh_every", type=int, default=500)
    ap.add_argument("--refresh_n", type=int, default=40000)
    ap.add_argument("--refresh_iters", type=int, default=300)
    ap.add_argument("--refresh_starts", type=int, default=1)
    ap.add_argument("--stage1_steps", type=int, default=0,
                    help="E2: >0 = LAD-style stage 1 (encoders, pair InfoNCE only) for this many steps, then all losses")
    ap.add_argument("--stage1_tau0", type=float, default=0.25, help="E2 stage 1 start temperature (LAD 0.25)")
    ap.add_argument("--enc_lr_scale", type=float, default=0.1, help="E2 stage 2 encoder lr factor (LAD FT 1e-5 vs 1e-4)")

    def setup(tr, args):
        tr.enable_local(args.w_local, args.local_deg, args.local_batch)
        info = tr.enable_pair(args)
        if args.stage1_steps:
            tr.enable_stages(args)
        keys = ("w_local", "local_deg", "local_batch", "pairs", "w_pair", "pair_batch", "pair_tau", "pair_max_res_mm",
                "pair_buffer", "refresh_every", "refresh_n", "refresh_iters", "refresh_starts", "stage1_steps",
                "stage1_tau0", "enc_lr_scale")
        return {"method": "ours: XL-VLA + local consistency + pair alignment, online pair buffer "
                          "(cross_dex.ours.xlvla_local_pair)" + (", LAD two-stage split (E2)" if args.stage1_steps else ""),
                "dexpilot": DP, **{k: getattr(args, k) for k in keys}, **info}

    args, _ = ap.parse_known_args(argv)
    TL.main(argv, parser=ap, trainer_cls=StagedPairTrainer if args.stage1_steps else PairTrainer, setup=setup)


if __name__ == "__main__":
    main()
