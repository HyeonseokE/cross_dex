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
E-b options (2026-11-18): --pair_direction best = for every hand pair only the tuples whose SOURCE is the better
direction's source hand are used (direction = lower median retarget residual, measured at start on --dir_probe_n poses;
e.g. 5-finger -> 4-finger, Inspire -> others), so 4 -> 5-finger pairs (little finger unconstrained) and pairs into a hand
that cannot follow are left out, and retarget-retarget pairs (neither hand the source) too. --pair_res_adaptive 1 =
threshold per direction max(pair_max_res_mm, min(median of that direction's latest block, pair_res_cap_mm)).
E-c option: --pair_align l2sg (needs --pair_direction best) = the InfoNCE term is replaced by
|enc_t(q_t) - stopgrad(enc_s(q_s))|^2 (mean over latent dims), s = the pair's source hand (exact, not retargeted pose):
only the retargeted hand's latent is pulled to the source's, so the source hand's space is not compressed by many-to-one
pairs. The cross reconstruction term is unchanged.
E-h hard negatives (2026-11-19, user: "hard negative ... 적용해보고 싶어"): --hn_anchors A > 0 = at every refresh, A of the
refresh_n poses are anchors, each with --hn_k neighbours q + d of the same source hand (d = random direction, RMS
--hn_deg_lo..--hn_deg_hi degrees per joint, as the local-consistency perturbation), stored as a contiguous group in the
ring buffer. Anchors are retargeted as usual; each neighbour is solved warm-started at its anchor's partner
(--hn_iters iterations) so close source poses get consistently solved partners (as the resolution evaluation set).
Every step --hn_frac of the pair batch is filled with whole groups, so the neighbours sit in the same InfoNCE batch
and act as hard negatives for each other (5-10 deg ~ 9-25 mm at the fingertips: above the pair noise, unlike the
1 deg of the local loss which stays the fine-scale term). The loss itself is unchanged.
E-g options (2026-11-19, user: E-b used only one direction for Inspire, try both with a relaxed filter):
--target_thr "inspire=60" = residual threshold (mm) of every direction INTO that hand (s -> inspire; other hands -> Inspire
pairs have 40-80 mm residuals because Inspire cannot follow, so 15 mm drops them all); directions out of it keep
--pair_max_res_mm. --relaxed_w 0.3 = weight of the rows that pass only through such a relaxed threshold (residual above
--pair_max_res_mm) in the pair losses (InfoNCE rows and cross reconstruction, weighted means), so these many-to-one pairs
pull less than the accurate ones.
E-f option (2026-11-19, user "LAD방식에서 뭘 추가해야 해 ... 진행 해"): --w_pair_l2 W adds W * |enc_A(q_A) - enc_B(q_B)|^2
/ var (mean over latent dims, var = mean per-dim variance of the batch's latents, both encoders get the gradient) to
the cosine InfoNCE: InfoNCE aligns only directions (LAD: the same pose had latent norms 0.28 .. 2.88 across hands,
L2 retrieval median rank 16 vs cosine 2.9), the decoders and a shared policy see the L2 geometry. InfoNCE is kept for
the negatives (E-c's L2 alone did not cluster).
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

from cross_dex.ours import fast_xl as FX
from cross_dex.ours import uniform_pairs as UP
from cross_dex.ours import xlvla_local as XLL
from dexverse_xlvla import train_latent as TL

DP = {"huber_delta": 0.03, "project_dist": 0.03, "eta1": 1e-4, "eta2": 3e-2}   # DexPilotOptimizer defaults


class PairTrainer(XLL.LocalTrainer):          # FastXLMixin comes with LocalTrainer
    w_pair = 0.0

    def enable_pair(self, a) -> dict:
        self.pa, self.w_pair = a, a.w_pair
        dev = self.config.device
        d = np.load(a.pairs)
        srcs = [s for s in self.hand_names if f"{s}__{s}" in d.files]
        q = {h: [] for h in self.hand_names}
        res = {h: [] for h in self.hand_names}
        src = []
        for s in srcs:
            n = len(d[f"{s}__{s}"])
            src.append(torch.full((n,), self.hand_names.index(s), dtype=torch.long))
            for h in self.hand_names:
                q[h].append(torch.from_numpy(d[f"{s}__{h}"]))
                res[h].append(torch.zeros(n) if h == s else torch.from_numpy(d[f"{s}__{h}__res_mm"]))
        q = {h: torch.cat(v) for h, v in q.items()}
        res = {h: torch.cat(v) for h, v in res.items()}
        n_all = len(res[self.hand_names[0]])
        size = min(a.pair_buffer, n_all)
        pick = torch.randperm(n_all)[:size]
        self.buf_q = {h: q[h][pick].to(dev) for h in self.hand_names}
        self.buf_res = {h: res[h][pick].float().to(dev) for h in self.hand_names}
        self.buf_src = torch.cat(src)[pick].to(dev)
        self.buf_grp = torch.full((size,), -1, dtype=torch.long, device=dev)    # E-h: group id (-1 = not a group row)
        self.buf_role = torch.full((size,), -1, dtype=torch.long, device=dev)   # E-h: 0 = anchor, 1..k = neighbour
        self.grp_next, self.grp_anchor_rows = 0, torch.zeros(0, dtype=torch.long, device=dev)
        self.buf_size, self.buf_ptr, self.pair_calls, self.refresh_src, self.refresh_sec = size, 0, 0, 0, 0.0
        H = self.hand_names
        self.thr = {(s, t): float(a.pair_max_res_mm) for s in H for t in H}       # residual threshold per direction
        self.tgt_thr = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.target_thr.split(",") if kv}
        for t, v in self.tgt_thr.items():                                         # E-g: relaxed directions into a hand
            for s in H:
                if s != t:
                    self.thr[(s, t)] = v
        self.dir_src = None                                                      # (x, y) -> source hand, best mode
        extra = {}
        if a.pair_direction == "best" or a.pair_res_adaptive:
            med = self._probe_medians(a.dir_probe_n)
            extra["pair_probe_median_mm"] = {f"{s}>{t}": round(v, 1) for (s, t), v in med.items()}
            if a.pair_res_adaptive:
                self._set_thr(med)
            if a.pair_direction == "best":
                self.dir_src = {(x, y): (x if med[(x, y)] <= med[(y, x)] else y)
                                for i, x in enumerate(H) for y in H[i + 1:]}
                extra["pair_direction_src"] = {f"{x}-{y}": s for (x, y), s in self.dir_src.items()}
                print("[pairs] direction (source) per hand pair: " +
                      ", ".join(f"{x}-{y}<-{s}" for (x, y), s in self.dir_src.items()), flush=True)
        info = self._usable()
        print(f"[pairs] buffer {size} tuples from {a.pairs} ({n_all}, sources {srcs}); usable % {info}", flush=True)
        inner = self.optimizer.extra
        self.optimizer.extra = lambda: (self._pair_backward(), inner())[1]
        return {"pair_init_tuples": n_all, "pair_buffer": size, "pair_init_usable_pct": info, **extra}

    @torch.enable_grad()
    def _probe_medians(self, n):
        """Median retarget residual (mm) of every direction s -> t on n uniform poses of s (current retarget code)."""
        a, dev, H, med, t0 = self.pa, self.config.device, self.hand_names, {}, time.time()
        for s in H:
            qs = torch.empty(n, self.hand_models[s].dof_count(), device=dev).uniform_(-1, 1)
            for t in H:
                if t != s:
                    _, rms, _ = UP.retarget(self.hand_models[s], self.hand_models[t], qs, a.refresh_iters, a.refresh_starts,
                                            0.05, 1e-6, DP)
                    med[(s, t)] = float(rms.median())
        print(f"[pairs] probe medians ({n} poses / direction, {time.time() - t0:.0f}s): " +
              ", ".join(f"{s}>{t} {v:.0f}" for (s, t), v in med.items()), flush=True)
        return med

    def _set_thr(self, med):
        a = self.pa
        for k, v in med.items():
            self.thr[k] = max(a.pair_max_res_mm, min(v, a.pair_res_cap_mm))

    def _ok(self, h, idx):
        """Tuple usable for hand h: residual of (tuple source -> h) <= that direction's threshold (source itself: 0)."""
        H = self.hand_names
        thr = torch.tensor([self.thr[(s, h)] if s != h else float("inf") for s in H], device=self.buf_src.device)
        return self.buf_res[h][idx] <= thr[self.buf_src[idx]]

    def _pair_mask(self, x, y, idx):
        m = self._ok(x, idx) & self._ok(y, idx)
        if self.dir_src is not None:
            m &= self.buf_src[idx] == self.hand_names.index(self.dir_src[(x, y)])
        return m

    def _usable(self):
        idx = torch.arange(self.buf_size, device=self.buf_src.device)
        if self.dir_src is None:
            return {h: round(100 * float(self._ok(h, idx).float().mean()), 1) for h in self.hand_names}
        return {f"{x}-{y}": round(100 * float(self._pair_mask(x, y, idx).float().mean()), 1) for (x, y) in self.dir_src}

    @torch.enable_grad()
    def _refresh(self):
        a, dev, t0 = self.pa, self.config.device, time.time()
        s = self.hand_names[self.refresh_src % len(self.hand_names)]
        self.refresh_src += 1
        n = a.refresh_n
        ms = self.hand_models[s]
        A, K = (a.hn_anchors, a.hn_k) if a.hn_anchors else (0, 0)
        n_rand = n - A * (K + 1)
        qs = torch.empty(n_rand + A, ms.dof_count(), device=dev).uniform_(-1, 1)         # random poses + anchors
        if A:                                                                          # E-h: neighbours of the anchors
            half = (0.5 * (ms._upper - ms._lower)).to(dev).clamp_min(1e-6)
            qa = qs[n_rand:]
            d = torch.randn(A, K, ms.dof_count(), device=dev)
            deg = torch.empty(A, K, 1, device=dev).uniform_(a.hn_deg_lo, a.hn_deg_hi)
            d = d / d.pow(2).mean(-1, keepdim=True).sqrt() * torch.deg2rad(deg) / half
            qn = (qa[:, None] + d).clamp(-1, 1)                                         # (A, K, dof)
            grp_src = torch.cat([qa[:, None], qn], 1).reshape(A * (K + 1), -1)           # anchor, its neighbours, ...
            src_all = torch.cat([qs[:n_rand], grp_src])
        else:
            src_all = qs
        new_q, new_res = {s: src_all}, {s: torch.zeros(n, device=dev)}
        for t in self.hand_names:
            if t == s:
                continue
            qt, rms, _ = UP.retarget(ms, self.hand_models[t], qs, a.refresh_iters, a.refresh_starts, 0.05, 1e-6, DP)
            qt, rms = qt.detach(), rms.float()
            if A:
                qta, rma = qt[n_rand:], rms[n_rand:]
                qtn, rmn = self._retarget_from(s, t, qn.reshape(A * K, -1), qta.repeat_interleave(K, 0), a.hn_iters)
                gq = torch.cat([qta[:, None], qtn.reshape(A, K, -1)], 1).reshape(A * (K + 1), -1)
                gr = torch.cat([rma[:, None], rmn.reshape(A, K)], 1).reshape(-1)
                qt, rms = torch.cat([qt[:n_rand], gq]), torch.cat([rms[:n_rand], gr])
            new_q[t], new_res[t] = qt, rms
            if a.pair_res_adaptive:                                   # threshold follows this direction's latest block
                self._set_thr({(s, t): float(rms.median())})
        idx = (torch.arange(n, device=dev) + self.buf_ptr) % self.buf_size
        for h in self.hand_names:
            self.buf_q[h][idx], self.buf_res[h][idx] = new_q[h], new_res[h]
        self.buf_src[idx] = self.hand_names.index(s)
        self.buf_grp[idx], self.buf_role[idx] = -1, -1
        if A:
            gidx = idx[n_rand:]
            self.buf_grp[gidx] = self.grp_next + torch.arange(A, device=dev).repeat_interleave(K + 1)
            self.buf_role[gidx] = torch.arange(K + 1, device=dev).repeat(A)
            self.grp_next += A
        self.buf_ptr = int((self.buf_ptr + n) % self.buf_size)
        if a.hn_anchors:                                              # anchors whose whole group is still in the ring
            anc = torch.nonzero(self.buf_role == 0).flatten()
            rows = (anc[:, None] + torch.arange(K + 1, device=dev)) % self.buf_size
            ok = (self.buf_grp[rows] == self.buf_grp[anc][:, None]).all(1)
            self.grp_anchor_rows = anc[ok]
        self.refresh_sec += time.time() - t0
        print(f"[refresh] call {self.pair_calls}: {n} new {s} poses -> buffer, {time.time() - t0:.0f}s, "
              f"usable % {self._usable()}" + (f", hard-negative groups {len(self.grp_anchor_rows)}" if a.hn_anchors else ""),
              flush=True)

    @torch.enable_grad()
    def _retarget_from(self, s, t, qs, qt0, iters):
        """E-h: DexPilot's objective (uniform_pairs.retarget's cost) solved from the partner qt0 of the anchor
        (tanh-parametrised joints, Adam) -> consistent partners for close source poses. Returns (q_t, rms mm)."""
        ms, mt = self.hand_models[s], self.hand_models[t]
        ts, ws = UP.keyvecs(ms, qs)
        k = min(ts.shape[1], mt.tip_count())
        r_i = (UP.open_lengths(mt)[:k] / UP.open_lengths(ms)[:k]).to(qs)
        sv, n_ff = UP.dexpilot_vectors(ts, ws, k)
        ref, w, _ = UP.dexpilot_reference(sv, n_ff, k, r_i, r_i.mean(), DP["project_dist"], DP["eta1"], DP["eta2"])
        hub = torch.nn.SmoothL1Loss(beta=DP["huber_delta"], reduction="none")
        nv = ref.shape[1]

        def cost(q):
            tt, wt = UP.keyvecs(mt, q)
            tv, _ = UP.dexpilot_vectors(tt, wt, k)
            d = (tv - ref).norm(dim=-1)
            return (hub(d, torch.zeros_like(d)) * w).sum(-1) / nv, d
        u = torch.atanh(qt0.clamp(-0.999, 0.999)).clone().requires_grad_(True)
        opt = torch.optim.Adam([u], lr=0.05)
        for _ in range(iters):
            opt.zero_grad()
            q = torch.tanh(u)
            c, _ = cost(q)
            (c.sum() + 1e-6 * q.pow(2).sum()).backward()
            opt.step()
        with torch.no_grad():
            q = torch.tanh(u)
            _, d = cost(q)
        return q.detach(), d.pow(2).mean(-1).sqrt() * 1000

    def _pair_backward(self, nce_only: bool = False, tau: float | None = None):
        if not self.w_pair:
            return
        a = self.pa
        tau = a.pair_tau if tau is None else tau
        self.pair_calls += 1
        if a.refresh_every and self.pair_calls % a.refresh_every == 0:
            self._refresh()
        idx = torch.randint(0, self.buf_size, (a.pair_batch,), device=self.config.device)
        if a.hn_anchors and len(self.grp_anchor_rows):
            K = a.hn_k                                                 # E-h: whole groups in the batch
            ng = int(a.hn_frac * a.pair_batch) // (K + 1)
            anc = self.grp_anchor_rows[torch.randint(0, len(self.grp_anchor_rows), (ng,), device=idx.device)]
            rows = ((anc[:, None] + torch.arange(K + 1, device=idx.device)) % self.buf_size).reshape(-1)
            idx = torch.cat([idx[:a.pair_batch - len(rows)], rows])
        names = self.hand_names
        hp = [(x, y) for i, x in enumerate(names) for y in names[i + 1:]]
        enc = lambda h, m: self.autoencoders[h].encode(self.buf_q[h][idx][m])[1]
        dec = lambda h, z: self.autoencoders[h].decode_from_latents(z[:, :0], z)[1]
        tot_cr, tot_nce, tot_l2, used = 0.0, 0.0, 0.0, 0
        for x, y in hp:
            m = self._pair_mask(x, y, idx)
            if int(m.sum()) < 2:
                continue
            zx, zy = enc(x, m), enc(y, m)
            rw = None
            if self.tgt_thr:                                           # E-g: rows passing only via a relaxed threshold
                relaxed = (self.buf_res[x][idx][m] > a.pair_max_res_mm) | (self.buf_res[y][idx][m] > a.pair_max_res_mm)
                rw = torch.where(relaxed, a.relaxed_w, 1.0).float()
                rw = rw / rw.sum()
            wmean = (lambda v: v.mean()) if rw is None else (lambda v: (v * rw).sum())
            cr = (zx.new_zeros(()) if nce_only else
                  0.5 * (wmean((dec(y, zx) - self.buf_q[y][idx][m]).pow(2).mean(-1))
                         + wmean((dec(x, zy) - self.buf_q[x][idx][m]).pow(2).mean(-1))))
            if a.pair_align == "l2sg" and not nce_only:
                zs, zt = (zx, zy) if self.dir_src[(x, y)] == x else (zy, zx)
                nce = F.mse_loss(zt, zs.detach())                     # E-c: source-anchored latent L2
            else:
                logits = F.normalize(zx, dim=-1) @ F.normalize(zy, dim=-1).T / tau
                lab = torch.arange(len(zx), device=zx.device)
                nce = 0.5 * (wmean(F.cross_entropy(logits, lab, reduction="none"))
                             + wmean(F.cross_entropy(logits.T, lab, reduction="none")))
            l2 = zx.new_zeros(())
            if a.w_pair_l2 and not nce_only:                           # E-f: L2 (norm + direction) pull of the pair
                var = 0.5 * (zx.detach().var(0).mean() + zy.detach().var(0).mean()).clamp_min(1e-8)
                l2 = (zx - zy).pow(2).mean() / var
            (self.w_pair * (cr + nce + a.w_pair_l2 * l2) / len(hp)).backward()
            tot_cr, tot_nce, tot_l2, used = tot_cr + float(cr), tot_nce + float(nce), tot_l2 + float(l2), used + 1
        self.extra_metrics = {**getattr(self, "extra_metrics", {}), "pair_cr": tot_cr / max(used, 1),
                              ("pair_l2" if a.pair_align == "l2sg" else "pair_nce"): tot_nce / max(used, 1), "pair_hand_pairs": used,
                              "pair_refresh_sec": self.refresh_sec,
                              **({"pair_l2n": tot_l2 / max(used, 1)} if a.w_pair_l2 else {})}


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
    ap.add_argument("--pair_buffer", type=int, default=240000,
                    help="multiple of 6 x refresh_n so every source hand always has tuples in the ring")
    ap.add_argument("--pair_direction", choices=("all", "best"), default="all",
                    help="best (E-b) = per hand pair only tuples sourced from the lower-residual direction's hand")
    ap.add_argument("--pair_res_adaptive", type=int, default=0,
                    help="1 (E-b) = threshold per direction max(pair_max_res_mm, min(median, pair_res_cap_mm))")
    ap.add_argument("--pair_res_cap_mm", type=float, default=30.0)
    ap.add_argument("--pair_align", choices=("nce", "l2sg"), default="nce",
                    help="l2sg (E-c) = source-anchored latent L2 instead of InfoNCE (needs --pair_direction best)")
    ap.add_argument("--hn_anchors", type=int, default=0, help="E-h: anchors per refresh (each with --hn_k neighbours)")
    ap.add_argument("--hn_k", type=int, default=4)
    ap.add_argument("--hn_deg_lo", type=float, default=5.0)
    ap.add_argument("--hn_deg_hi", type=float, default=10.0)
    ap.add_argument("--hn_iters", type=int, default=100, help="E-h: neighbour solve iterations from the anchor's partner")
    ap.add_argument("--hn_frac", type=float, default=0.25, help="E-h: fraction of the pair batch filled with whole groups")
    ap.add_argument("--target_thr", default="",
                    help='E-g: "hand=mm[,hand=mm]" residual threshold of the directions into that hand (e.g. inspire=60)')
    ap.add_argument("--relaxed_w", type=float, default=1.0,
                    help="E-g: weight of rows that pass only through a relaxed --target_thr (residual > pair_max_res_mm)")
    ap.add_argument("--w_pair_l2", type=float, default=0.0,
                    help="E-f: + w * |z_A - z_B|^2 / var on the pairs next to the cosine InfoNCE (aligns the latent norm)")
    ap.add_argument("--dir_probe_n", type=int, default=4000, help="poses per source hand to measure the direction medians")
    ap.add_argument("--refresh_every", type=int, default=500)
    ap.add_argument("--refresh_n", type=int, default=40000)
    ap.add_argument("--refresh_iters", type=int, default=300)
    ap.add_argument("--refresh_starts", type=int, default=1)
    ap.add_argument("--fast", type=int, default=1, help="1 = cross_dex.ours.fast_xl (batched FK / decoding, same maths)")
    ap.add_argument("--stage1_steps", type=int, default=0,
                    help="E2: >0 = LAD-style stage 1 (encoders, pair InfoNCE only) for this many steps, then all losses")
    ap.add_argument("--stage1_tau0", type=float, default=0.25, help="E2 stage 1 start temperature (LAD 0.25)")
    ap.add_argument("--stage2_only", type=int, default=0,
                    help="1 = start directly in E2's stage 2 (all losses, encoder lr x enc_lr_scale); with --resume of an E2 "
                         "checkpoint this continues its stage 2 (2026-11-19, user: run E2 longer until it converges)")
    ap.add_argument("--enc_lr_scale", type=float, default=0.1, help="E2 stage 2 encoder lr factor (LAD FT 1e-5 vs 1e-4)")

    def setup(tr, args):
        if args.fast:                                      # cross_dex.ours.fast_xl: same maths, batched FK / decoding
            FX.make_fast(tr)
            tr.fast = True
            tr._local_loss = lambda: FX.fast_local_loss(tr)
        tr.enable_local(args.w_local, args.local_deg, args.local_batch)
        if args.pair_align == "l2sg" and args.pair_direction != "best":
            raise SystemExit("--pair_align l2sg needs --pair_direction best (the source hand is the anchor)")
        info = tr.enable_pair(args)
        if args.stage1_steps or args.stage2_only:
            tr.enable_stages(args)
        if args.stage2_only:              # stage-2 optimizer (encoder / other groups) before --resume loads its state
            tr._enter_stage2()
        keys = ("w_local", "local_deg", "local_batch", "pairs", "w_pair", "pair_batch", "pair_tau", "pair_max_res_mm",
                "pair_buffer", "pair_direction", "pair_res_adaptive", "pair_res_cap_mm", "pair_align", "hn_anchors", "hn_k", "hn_deg_lo", "hn_deg_hi", "hn_iters", "hn_frac", "target_thr", "relaxed_w", "w_pair_l2", "dir_probe_n", "refresh_every", "refresh_n", "refresh_iters", "refresh_starts", "stage1_steps",
                "stage1_tau0", "enc_lr_scale", "stage2_only", "fast")
        return {"method": "ours: XL-VLA + local consistency + pair alignment, online pair buffer "
                          "(cross_dex.ours.xlvla_local_pair)" + (", LAD two-stage split (E2)" if args.stage1_steps else ""),
                "dexpilot": DP, **{k: getattr(args, k) for k in keys}, **info}

    args, _ = ap.parse_known_args(argv)
    TL.main(argv, parser=ap, trainer_cls=StagedPairTrainer if (args.stage1_steps or args.stage2_only) else PairTrainer, setup=setup)


if __name__ == "__main__":
    main()
