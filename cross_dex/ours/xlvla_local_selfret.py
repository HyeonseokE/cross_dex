"""Ours G (2026-11-18, user: uniform poses for coverage + similar poses of different hands clustered, with the adaptive
residual filter of E-b): F (XL-VLA + local consistency, cross_dex.ours.xlvla_local) + self-retargeting + latent cycle.

Every step, per source hand s, fresh uniform poses q_s (XL-VLA's generator) are encoded, z_s = enc_s(q_s), and decoded
to every other hand t: q_t = dec_t(z_s). No pair buffer and no precomputed retargeting: the cross decoding IS the pair.
  L_rt   DexPilot's objective (cross_dex.ours.uniform_pairs: finger-finger + wrist->tip vectors, length ratios, pinch
         projection, Huber, weights) between FK_t(q_t) and the source's vectors, for every pose (not only pinches as
         XL-VLA's exp(-12 d) weighted pinch loss) -> the cross decoder learns to retarget.  Gradient: enc_s, dec_t.
  L_cyc  |enc_t(stopgrad q_t) - stopgrad z_s|^2 / var(z_s) (mean over latent dims): the corresponding pose of hand t is
         encoded to the same point as the source -> same poses of different hands cluster. Only pulls (no negatives,
         unlike InfoNCE). Gradient: enc_t.
  adaptive filter (E-b's --pair_res_adaptive, per direction s -> t): residual = unweighted RMS of the DexPilot vector
         errors (mm) of q_t; threshold = max(--res_min_mm, min(EMA median of that direction, --res_cap_mm)); L_cyc only
         on poses at or below it (L_rt on all poses, so bad directions still learn to retarget).
G2 (--refine_iters N > 0, user 2026-11-18 "절충안"): the pair's target pose is no longer the decoder output itself:
  DexPilot's objective is solved for N Adam iterations starting from it (uniform_pairs.retarget's parametrisation, warm
  start instead of 0) -> q*_t. L_cyc and the residual filter use q*_t, and w_ref |dec_t(z_s) - q*_t|^2 (E's pair cross
  reconstruction toward the solved pose) is added. Fresh pairs every step at near-DexPilot quality, ~N x FK cost.
  --refine_until M: only the first M steps (user: from scratch, the refinement fixes the pairs while the decoder is still
  random; later the decoder output is the pair as in G). G2 run = from scratch (no --resume), 30k, refine 10 it until 3k.
  w_cyc ramps linearly from 0 over --cyc_warmup steps (from F the cycle loss starts at 7.9 and, at full weight, threw
  the self reconstruction from 0.005 to 0.18 in 20 steps; 2026-11-18 dry run).

G3 (--sr_batched 1, user 2026-11-19 "매스텝 다듬기 추가해서 돌려봐"): G2's DexPilot refinement on EVERY step
  (--refine_until 0) so the cycle target is always a near-DexPilot pair, made affordable by computing the 30 directions
  as ~9 batches (per target hand, the sources with the same fingertip count k decoded / FK'd / refined together; Adam
  is element-wise and the cost is a per-pose sum, so the batched solve equals the per-direction one), the residual
  filter as weights instead of boolean indexing, and one host sync per step (G/G2 path: ~32k kernels and a .item() per
  direction -> CPU-launch bound, GPU ~10 % busy). check_selfret_batched.py compares both paths on the same inputs.

  L = L_XL + w_local (L_self + L_cross) + w_rt L_rt + w_cyc L_cyc

  cd /workspace/codes/dexverse/DexLatent && PYTHONPATH=/workspace/codes/cross_dex \
      python -m cross_dex.ours.xlvla_local_selfret --variant paper --run_dir <run> --ckpt_dir <dir> \
      --resume <F checkpoint> --num_steps 45000 --w_local 1 --w_rt 1 --w_cyc 1
"""
from __future__ import annotations

import torch

from cross_dex.ours import fast_xl as FX
from cross_dex.ours import uniform_pairs as UP
from cross_dex.ours import xlvla_local as XLL
from dexverse_xlvla import train_latent as TL

DP = {"huber_delta": 0.03, "project_dist": 0.03, "eta1": 1e-4, "eta2": 3e-2}   # DexPilotOptimizer defaults


class SelfRetargetTrainer(XLL.LocalTrainer):
    w_rt = w_cyc = 0.0

    def enable_selfret(self, a) -> dict:
        self.sa, self.w_rt, self.w_cyc = a, a.w_rt, a.w_cyc
        H = self.hand_names
        self.med = {(s, t): None for s in H for t in H if s != t}             # EMA median residual (mm) per direction
        self.ratio = {}
        self.dir_idx = {d: i for i, d in enumerate(self.med)}                  # batched path: direction -> index
        self.med_t = torch.full((len(self.med),), float(a.res_min_mm), device=self.config.device)
        for s in H:
            for t in H:
                if s != t:
                    ms, mt = self.hand_models[s], self.hand_models[t]
                    k = min(ms.tip_count(), mt.tip_count())
                    r_i = (UP.open_lengths(mt)[:k] / UP.open_lengths(ms)[:k]).to(self.config.device).detach()
                    self.ratio[(s, t)] = (k, r_i)
        inner = self.optimizer.extra
        self.optimizer.extra = lambda: (self._selfret_backward(), inner())[1]
        return {}

    def thr(self, s, t):
        a, m = self.sa, self.med[(s, t)]
        return a.res_min_mm if m is None else max(a.res_min_mm, min(m, a.res_cap_mm))

    def _selfret_backward(self, qs_of=None):
        if not (self.w_rt or self.w_cyc):
            return
        if getattr(self.sa, "sr_batched", 0):
            return self._selfret_backward_batched(qs_of)
        a, dev, H = self.sa, self.config.device, self.hand_names
        enc = lambda h, x: self.autoencoders[h].encode(x)[1]                                  # posterior mean
        dec = lambda h, z: self.autoencoders[h].decode_from_latents(z[:, :0], z)[1]
        huber = torch.nn.SmoothL1Loss(beta=DP["huber_delta"], reduction="none")
        n_dir = len(H) * (len(H) - 1)
        self._sr_n = getattr(self, "_sr_n", 0) + 1
        w_cyc = self.w_cyc * min(1.0, self._sr_n / max(a.cyc_warmup, 1))      # linear ramp (F start: cycle 7.9 at step 1)
        tot_rt, tot_cyc, res_med, pass_pct, dec_med = 0.0, 0.0, {}, {}, {}
        for s in H:
            ms = self.hand_models[s]
            qs = qs_of[s] if qs_of else torch.empty(a.sr_batch, ms.dof_count(), device=dev).uniform_(-1, 1)
            zs = enc(s, qs)
            with torch.no_grad():
                ts, ws = UP.keyvecs(ms, qs)
            var = zs.detach().var(0).mean().clamp_min(1e-8)
            loss_s = zs.new_zeros(())
            for t in H:
                if t == s:
                    continue
                mt = self.hand_models[t]
                k, r_i = self.ratio[(s, t)]
                with torch.no_grad():
                    sv, n_ff = UP.dexpilot_vectors(ts, ws, k)
                    ref, w, _ = UP.dexpilot_reference(sv, n_ff, k, r_i, r_i.mean(), DP["project_dist"], DP["eta1"], DP["eta2"])
                qt = dec(t, zs).clamp(-1, 1)
                tt, wt = UP.keyvecs(mt, qt)
                tv, _ = UP.dexpilot_vectors(tt, wt, k)
                d = (tv - ref).norm(dim=-1)                                                    # (B, n_vec) m
                rt = ((huber(d, torch.zeros_like(d)) * w).sum(-1) / ref.shape[1]).mean()
                res = d.detach().pow(2).mean(-1).sqrt() * 1000                                 # mm, as uniform_pairs
                dec_med[f"{s}>{t}"] = float(res.median())
                ref_loss = zs.new_zeros(())
                if a.refine_iters > 0 and (not a.refine_until or self._sr_n <= a.refine_until):   # G2: DexPilot solve from the decoder output
                    qp, res = self._refine(mt, k, ref, w, qt.detach(), huber)
                    ref_loss = (qt - qp).pow(2).mean()                                       # decoder -> refined pose
                    qt = qp
                m = float(res.median())
                self.med[(s, t)] = m if self.med[(s, t)] is None else a.res_ema * self.med[(s, t)] + (1 - a.res_ema) * m
                keep = res <= self.thr(s, t)
                if int(keep.sum()) > 0:
                    zt = enc(t, qt.detach()[keep])
                    cyc = (zt - zs.detach()[keep]).pow(2).mean() / var
                else:
                    cyc = zs.new_zeros(())
                loss_s = loss_s + (self.w_rt * a.rt_scale * rt + w_cyc * cyc + a.w_ref * ref_loss) / n_dir
                tot_rt, tot_cyc = tot_rt + float(rt), tot_cyc + float(cyc)
                res_med[f"{s}>{t}"], pass_pct[f"{s}>{t}"] = m, 100.0 * float(keep.float().mean())
            loss_s.backward()                                                  # per source hand: frees its graph
            self._sr_loss = getattr(self, "_sr_loss", 0.0) + float(loss_s)     # (equality check)
        self.extra_metrics = {**getattr(self, "extra_metrics", {}), "sr_w_cyc": w_cyc, "sr_rt": tot_rt / n_dir, "sr_cyc": tot_cyc / n_dir,
                              "sr_res_med_mm": sum(res_med.values()) / n_dir,
                              "sr_dec_res_med_mm": sum(dec_med.values()) / n_dir, "sr_pass_pct": sum(pass_pct.values()) / n_dir}
        self.sr_detail = (res_med, pass_pct)
        if self.step_calls_log():
            print("[selfret] res_med_mm/pass%/thr " + ", ".join(
                f"{k} {res_med[k]:.0f}/{pass_pct[k]:.0f}/{self.thr(*k.split('>')):.0f}" for k in res_med), flush=True)

    def _selfret_backward_batched(self, qs_of=None):
        """G3: same losses as _selfret_backward, directions batched per (target, fingertip count); see module doc."""
        a, dev, H = self.sa, self.config.device, self.hand_names
        enc = lambda h, x: self.autoencoders[h].encode(x)[1]
        dec = lambda h, z: self.autoencoders[h].decode_from_latents(z[:, :0], z)[1]
        huber = torch.nn.SmoothL1Loss(beta=DP["huber_delta"], reduction="none")
        n_dir = len(H) * (len(H) - 1)
        self._sr_n = getattr(self, "_sr_n", 0) + 1
        w_cyc = self.w_cyc * min(1.0, self._sr_n / max(a.cyc_warmup, 1))
        refine_on = a.refine_iters > 0 and (not a.refine_until or self._sr_n <= a.refine_until)
        B = a.sr_batch
        zs, var, kv, svc = {}, {}, {}, {}
        for s in H:
            ms = self.hand_models[s]
            qs = qs_of[s] if qs_of else torch.empty(B, ms.dof_count(), device=dev).uniform_(-1, 1)
            zs[s] = enc(s, qs)
            with torch.no_grad():
                kv[s] = UP.keyvecs(ms, qs)
            var[s] = zs[s].detach().var(0).mean().clamp_min(1e-8)
        loss = zs[H[0]].new_zeros(())
        names, rt_l, cyc_l, m_l, dm_l, pp_l = [], [], [], [], [], []
        first = self._sr_n == 1 or not getattr(self, "_med_init", False)
        for t in H:
            mt = self.hand_models[t]
            groups = {}
            for s in H:
                if s != t:
                    groups.setdefault(self.ratio[(s, t)][0], []).append(s)
            for k, srcs in groups.items():
                n = len(srcs)
                with torch.no_grad():
                    for s in srcs:
                        if (s, k) not in svc:
                            svc[(s, k)] = UP.dexpilot_vectors(*kv[s], k)
                    sv = torch.cat([svc[(s, k)][0] for s in srcs])
                    n_ff = svc[(srcs[0], k)][1]
                    r_i = torch.cat([self.ratio[(s, t)][1].expand(B, k) for s in srcs])
                    ref, w = _reference_rows(sv, n_ff, k, r_i, r_i.mean(-1))
                z = torch.cat([zs[s] for s in srcs])
                qt = dec(t, z).clamp(-1, 1)
                tt, wt = UP.keyvecs(mt, qt)
                tv, _ = UP.dexpilot_vectors(tt, wt, k)
                d = (tv - ref).norm(dim=-1)
                rt = ((huber(d, torch.zeros_like(d)) * w).sum(-1) / ref.shape[1]).view(n, B).mean(1)
                res = d.detach().pow(2).mean(-1).sqrt() * 1000
                dm_l.append(res.view(n, B).median(1).values)
                ref_l = rt.new_zeros(n)
                if refine_on:
                    qp, res = self._refine(mt, k, ref, w, qt.detach(), huber)
                    ref_l = (qt - qp).pow(2).view(n, B, -1).mean((1, 2))
                    qt = qp
                m = res.view(n, B).median(1).values
                idx = torch.tensor([self.dir_idx[(s, t)] for s in srcs], device=dev)
                med = m if first else a.res_ema * self.med_t[idx] + (1 - a.res_ema) * m
                self.med_t[idx] = med
                thr = med.clamp(min=a.res_min_mm, max=max(a.res_cap_mm, a.res_min_mm))
                keep = (res.view(n, B) <= thr[:, None]).float()
                zt = enc(t, qt.detach())
                err = (zt - z.detach()).pow(2).mean(-1).view(n, B)
                vv = torch.stack([var[s] for s in srcs])
                cyc = (err * keep).sum(1) / keep.sum(1).clamp_min(1) / vv
                loss = loss + (self.w_rt * a.rt_scale * rt + w_cyc * cyc + a.w_ref * ref_l).sum() / n_dir
                names += [f"{s}>{t}" for s in srcs]
                rt_l.append(rt.detach()); cyc_l.append(cyc.detach()); m_l.append(m); pp_l.append(100 * keep.mean(1))
        self._med_init = True
        loss.backward()
        st = torch.stack([torch.cat(x) for x in (rt_l, cyc_l, m_l, dm_l, pp_l)]).tolist()     # one host sync
        rt_v, cyc_v, res_v, dec_v, pp_v = st
        self._sr_loss = getattr(self, "_sr_loss", 0.0) + float(loss)
        res_med, pass_pct = dict(zip(names, res_v)), dict(zip(names, pp_v))
        for nm, v in res_med.items():                                        # mirror for thr() / logs / resume
            self.med[tuple(nm.split(">"))] = float(self.med_t[self.dir_idx[tuple(nm.split(">"))]])
        self.extra_metrics = {**getattr(self, "extra_metrics", {}), "sr_w_cyc": w_cyc, "sr_rt": sum(rt_v) / n_dir,
                              "sr_cyc": sum(cyc_v) / n_dir, "sr_res_med_mm": sum(res_v) / n_dir,
                              "sr_dec_res_med_mm": sum(dec_v) / n_dir, "sr_pass_pct": sum(pp_v) / n_dir}
        self.sr_detail = (res_med, pass_pct)
        if self.step_calls_log():
            print("[selfret] res_med_mm/pass%/thr " + ", ".join(
                f"{k} {res_med[k]:.0f}/{pass_pct[k]:.0f}/{self.thr(*k.split('>')):.0f}" for k in res_med), flush=True)

    @torch.enable_grad()
    def _refine(self, mt, k, ref, w, q0, huber):
        """G2: DexPilot's objective (as uniform_pairs.retarget, tanh-parametrised joints, Adam) solved for
        --refine_iters iterations from the decoder's pose q0 instead of from 0 -> the pair's target-hand pose."""
        a = self.sa
        u = torch.atanh(q0.clamp(-0.999, 0.999)).clone().requires_grad_(True)
        opt = torch.optim.Adam([u], lr=a.refine_lr)
        nv = ref.shape[1]
        for _ in range(a.refine_iters):
            opt.zero_grad(set_to_none=True)
            q = torch.tanh(u)
            tt, wt = UP.keyvecs(mt, q)
            tv, _ = UP.dexpilot_vectors(tt, wt, k)
            d = (tv - ref).norm(dim=-1)
            c = (huber(d, torch.zeros_like(d)) * w).sum(-1) / nv
            (c.sum() + 1e-6 * q.pow(2).sum()).backward()
            opt.step()
        with torch.no_grad():
            q = torch.tanh(u)
            tt, wt = UP.keyvecs(mt, q)
            tv, _ = UP.dexpilot_vectors(tt, wt, k)
            res = (tv - ref).norm(dim=-1).pow(2).mean(-1).sqrt() * 1000
        return q.detach(), res

    def step_calls_log(self) -> bool:
        self._sr_calls = getattr(self, "_sr_calls", 0) + 1
        return self._sr_calls == 1 or self._sr_calls % self.sa.sr_log_every == 0


def _reference_rows(sv, n_ff, k, r_i, r):
    """uniform_pairs.dexpilot_reference with per-row length ratios r_i (N, k), r (N,) (rows of several directions)."""
    ff, wr = sv[:, :n_ff], sv[:, n_ff:]
    dist = ff.norm(dim=-1)
    n1 = k - 1
    proj = torch.zeros_like(dist, dtype=torch.bool)
    proj[:, :n1] = dist[:, :n1] < DP["project_dist"]
    o2, t2 = zip(*[(j, i) for i in range(k - 2) for j in range(i + 1, k - 1)]) if k > 2 else ((), ())
    if o2:
        proj[:, n1:] = proj[:, list(o2)] & proj[:, list(t2)] & (dist[:, n1:] <= 0.03)
    pd = torch.tensor([DP["eta1"]] * n1 + [DP["eta2"]] * (n_ff - n1), device=ff.device)
    ref_ff = torch.where(proj[..., None], ff / (dist[..., None] + 1e-6) * pd[None, :, None], ff * r[:, None, None])
    ref = torch.cat([ref_ff, wr * r_i[:, :, None]], 1)
    w = torch.where(proj, torch.tensor([200.0] * n1 + [400.0] * (n_ff - n1), device=ff.device), torch.ones_like(dist))
    w = torch.cat([w, torch.full((ff.shape[0], k), float(n_ff + k), device=ff.device)], 1)
    return ref, w


def main(argv=None) -> None:
    ap = TL.build_parser()
    ap.add_argument("--w_local", type=float, default=1.0)
    ap.add_argument("--local_deg", type=float, default=1.0)
    ap.add_argument("--local_batch", type=int, default=1024)
    ap.add_argument("--w_rt", type=float, default=1.0, help="self-retargeting (DexPilot objective) loss weight")
    ap.add_argument("--rt_scale", type=float, default=1.0,
                    help="fixed scale of the DexPilot cost (raw cost is small: Huber in metres)")
    ap.add_argument("--w_cyc", type=float, default=1.0, help="latent cycle loss weight")
    ap.add_argument("--refine_iters", type=int, default=0,
                    help="G2: >0 = DexPilot solve of this many Adam iterations from the decoder output gives the pair")
    ap.add_argument("--refine_until", type=int, default=0,
                    help="G2: >0 = refine only in the first this-many steps of this run (0 = every step)")
    ap.add_argument("--refine_lr", type=float, default=0.05, help="as uniform_pairs (tanh-parametrised joints)")
    ap.add_argument("--w_ref", type=float, default=1.0, help="G2: decoder output -> refined pose MSE (normalized joints)")
    ap.add_argument("--cyc_warmup", type=int, default=2000, help="steps to ramp w_cyc 0 -> w_cyc")
    ap.add_argument("--sr_batch", type=int, default=1024, help="uniform poses per source hand per step")
    ap.add_argument("--res_min_mm", type=float, default=15.0, help="E-b pair_max_res_mm")
    ap.add_argument("--res_cap_mm", type=float, default=30.0, help="E-b pair_res_cap_mm")
    ap.add_argument("--res_ema", type=float, default=0.99, help="EMA of the per-direction median residual")
    ap.add_argument("--sr_log_every", type=int, default=500)
    ap.add_argument("--sr_batched", type=int, default=0,
                    help="G3: 1 = directions batched per (target, fingertip count), filter as weights, one sync/step")
    ap.add_argument("--fast", type=int, default=1, help="1 = cross_dex.ours.fast_xl (batched FK / decoding, same maths)")

    def setup(tr, args):
        if args.fast:
            FX.make_fast(tr)
            tr.fast = True
            tr._local_loss = lambda: FX.fast_local_loss(tr)
        tr.enable_local(args.w_local, args.local_deg, args.local_batch)
        tr.enable_selfret(args)
        keys = ("w_local", "local_deg", "local_batch", "w_rt", "rt_scale", "w_cyc", "sr_batch", "res_min_mm",
                "res_cap_mm", "res_ema", "cyc_warmup", "refine_iters", "refine_until", "refine_lr", "w_ref", "sr_batched", "fast")
        return {"method": "ours G: XL-VLA + local consistency + self-retargeting (DexPilot objective) + latent cycle, "
                          "adaptive residual filter (cross_dex.ours.xlvla_local_selfret)"
                          + (f", G2: pairs = DexPilot solve warm-started at the decoder output ({args.refine_iters} it)"
                             if args.refine_iters else ""),
                "dexpilot": DP, **{k: getattr(args, k) for k in keys}}

    TL.main(argv, parser=ap, trainer_cls=SelfRetargetTrainer, setup=setup)


if __name__ == "__main__":
    main()
