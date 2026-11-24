"""Ours (2026-11-19, user: "xhand v2 ... 우리 세팅 6손, 균일 샘플링 방식으로 적용해서 학습해보고 싶은데"): the losses of the
xhand finger-latent codec v2 (Dexverse-CSI/xhand-finger-latent `v2/cyc05clf05f002`, teammate code
eunyoung030927/DexVerse dev `experiments/cross_hand_latent_codec/code/{arm_train,cv2_train}.py`), re-implemented on the
XL-VLA hand VAE (same size: MLP 64-128-64, latent 32) for the six DexVerse right hands with XL-VLA's uniform pose
generator (fresh uniform joints in [-1, 1] every step) instead of the codec's "play" posture data.

  L = rec + beta KL                                    (XL-VLA trainer, pinch losses switched off: E2* has none)
    + lam   * L_align   (E2*)    lam 2.8333, linear warm-up over --align_warmup steps
    + lam_cyc  * L_cyc  / V      (v2) 0.05      from --v2_start, warm-up --v2_warmup
    + lam_pair * L_pair / V      (v2) 0.05
    + lam_fisher * L_fisher      (v2) 0.002
  L_align  for every ordered hand pair s -> t: z_s = mu_s(q_s) of uniform poses, q_t = Dec_t(z_s); point sets = the k
           common fingertips + the wrist point (canonical FK, cross_dex.ours.uniform_pairs.keyvecs), the source set scaled by
           the hand-size ratio; rigid Kabsch fit (no gradient, envelope theorem) of target onto source; residual^2 / k+1 /
           scale_t^2 (codec: weighted Kabsch residual^2 between pad correspondence sets / scale_src^2).
  L_cyc    ordered pairs A -> B with fingertips(A) <= fingertips(B) (5->4 skipped, as cv2_train.cycle_pairs);
           q~ = stopgrad(Dec_B(mu_A)) + N(0, 0.02^2) clamped; L = mean ||mu_B(q~) - mu_A||^2;  V = mean ||z - mean z||^2 of the
           step's native z (not detached).
  L_pair   paired scenes: S tuples of the DexPilot uniform pair file (cross_dex.ours.uniform_pairs, wrist-fixed) instead of
           the codec's kinematic closure scenes (their metadata is not available); a hand counts in a scene when its
           retarget residual <= --pair_res_mm; L = mean_s mean_h ||z_hs - mean_h z_hs||^2.
  L_fisher tr(Sw^-1 Sb) of hand labels on the same scenes' z (cv2_train.fisher_loss).
Differences from the codec (recorded in the run README): data = uniform poses (+ DexPilot pair scenes), six hands incl.
Inspire, fingertips + wrist instead of pads / palm anchors, align on posterior means, one run from scratch (v2 terms start
at --v2_start instead of warm-starting from E2*).

  cd /workspace/codes/dexverse/DexLatent && PYTHONPATH=/workspace/codes/cross_dex \
      python -m cross_dex.ours.xhand_v2_uniform --variant paper --run_dir <run> --ckpt_dir <dir> --num_steps 30000 \
      --pairs <dexpilot_right6_60k_wristfix.npz>
"""
from __future__ import annotations

import numpy as np
import torch

from cross_dex.ours import fast_xl as FX
from cross_dex.ours import uniform_pairs as UP
from cross_dex.ours import xlvla_local as XLL
from dexverse_xlvla import train_latent as TL


def kabsch(P, Q):
    """Rigid R, t minimising sum ||R P_i + t - Q_i||^2 (batched (B, n, 3))."""
    pc, qc = P.mean(1, keepdim=True), Q.mean(1, keepdim=True)
    Hm = (P - pc).transpose(1, 2) @ (Q - qc)
    U, _, Vt = torch.linalg.svd(Hm)
    d = torch.sign(torch.det(Vt.transpose(1, 2) @ U.transpose(1, 2)))
    D = torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1))
    R = Vt.transpose(1, 2) @ D @ U.transpose(1, 2)
    t = qc - pc @ R.transpose(1, 2)
    return R, t


class XhandV2Trainer(XLL.LocalTrainer):
    def enable_v2(self, a) -> dict:
        self.va, dev, H = a, self.config.device, self.hand_names
        self.scale = {h: float(UP.open_lengths(self.hand_models[h]).max()) for h in H}      # hand size (open wrist->tip)
        self.ntip = {h: self.hand_models[h].tip_count() for h in H}
        self.cyc_pairs = [(x, y) for x in H for y in H if x != y and self.ntip[x] <= self.ntip[y]]
        d = np.load(a.pairs)
        q, res = {h: [] for h in H}, {h: [] for h in H}
        for s in H:
            n = len(d[f"{s}__{s}"])
            for h in H:
                q[h].append(torch.from_numpy(d[f"{s}__{h}"]))
                res[h].append(torch.zeros(n) if h == s else torch.from_numpy(d[f"{s}__{h}__res_mm"]))
        self.pq = {h: torch.cat(v).float().to(dev) for h, v in q.items()}
        self.pok = {h: (torch.cat(v) <= a.pair_res_mm).to(dev) for h, v in res.items()}
        self._v2_n = 0
        self.optimizer = XLL.LocalLossOptimizer(self.optimizer, self._v2_loss)
        return {"hand_scale_m": {h: round(v, 4) for h, v in self.scale.items()},
                "cycle_pairs": [f"{x}->{y}" for x, y in self.cyc_pairs],
                "pair_scene_valid_pct": {h: round(100 * float(v.float().mean()), 1) for h, v in self.pok.items()}}

    def _v2_loss(self):
        a, dev, H = self.va, self.config.device, self.hand_names
        self._v2_n += 1
        n = self._v2_n
        enc = lambda h, x: self.autoencoders[h].encode(x)[1]
        dec = lambda h, z: self.autoencoders[h].decode_from_latents(z[:, :0], z)[1]
        q, mu, kv = {}, {}, {}
        for h in H:
            q[h] = torch.empty(a.v2_batch, self.hand_models[h].dof_count(), device=dev).uniform_(-1, 1)
            mu[h] = enc(h, q[h])
            with torch.no_grad():
                kv[h] = UP.keyvecs(self.hand_models[h], q[h])
        # E2* alignment: Kabsch residual of cross-decoded fingertips + wrist
        align, cnt = mu[H[0]].new_zeros(()), 0
        for t in H:
            srcs = [s for s in H if s != t]
            qd = dec(t, torch.cat([mu[s] for s in srcs])).clamp(-1, 1)
            tt, wt = UP.keyvecs(self.hand_models[t], qd)
            for i, s in enumerate(srcs):
                sl = slice(i * a.v2_batch, (i + 1) * a.v2_batch)
                k = min(self.ntip[s], self.ntip[t])
                ts, ws = kv[s]
                Ps = torch.cat([ts[:, :k], (ts[:, :1] - ws[:, :1])], 1) * (self.scale[t] / self.scale[s])
                Pt = torch.cat([tt[sl, :k], (tt[sl, :1] - wt[sl, :1])], 1)
                with torch.no_grad():
                    R, tr = kabsch(Pt.detach(), Ps)
                r2 = ((Pt @ R.transpose(1, 2) + tr - Ps).pow(2).sum(-1).mean(-1)) / self.scale[t] ** 2
                align, cnt = align + r2.mean(), cnt + 1
        align = align / cnt
        lam = a.lam * min(1.0, n / max(a.align_warmup, 1))
        total = lam * align
        m = {"v2_align": float(align), "v2_lam": lam}
        if n >= a.v2_start and (a.lam_cyc or a.lam_pair or a.lam_fisher):
            w = min(1.0, (n - a.v2_start + 1) / max(a.v2_warmup, 1))
            Zall = torch.cat([mu[h] for h in H])
            V = (Zall - Zall.mean(0)).pow(2).sum(-1).mean()
            cyc = Zall.new_zeros(())
            if a.lam_cyc:
                for x, y in self.cyc_pairs:
                    with torch.no_grad():
                        qt = (dec(y, mu[x]) + a.cyc_noise * torch.randn(a.v2_batch, self.hand_models[y].dof_count(), device=dev)).clamp(-1, 1)
                    cyc = cyc + (enc(y, qt) - mu[x]).pow(2).sum(-1).mean()
                cyc = cyc / len(self.cyc_pairs)
            idx = torch.randint(0, len(self.pq[H[0]]), (a.pair_scenes,), device=dev)
            Z = torch.stack([enc(h, self.pq[h][idx]) for h in H], 1)                        # (S, H, D)
            M = torch.stack([self.pok[h][idx] for h in H], 1).float()                        # valid hands per scene
            keep = M.sum(1) >= 2
            Z, M = Z[keep], M[keep]
            zc = (Z * M[..., None]).sum(1, keepdim=True) / M.sum(1)[:, None, None]
            pr = (((Z - zc).pow(2).sum(-1) * M).sum(1) / M.sum(1)).mean()
            fis = Zall.new_zeros(())
            if a.lam_fisher:
                Zf, Yf = Z[M.bool()], torch.arange(len(H), device=dev).expand(len(Z), -1)[M.bool()]
                mu0 = Zf.mean(0)
                Sw = Zf.new_zeros(Zf.shape[1], Zf.shape[1]); Sb = torch.zeros_like(Sw)
                for kk in range(len(H)):
                    Zk = Zf[Yf == kk]
                    if len(Zk) < 2:
                        continue
                    mk = Zk.mean(0)
                    Sw = Sw + (Zk - mk).T @ (Zk - mk)
                    Sb = Sb + len(Zk) * torch.outer(mk - mu0, mk - mu0)
                Sw, Sb = Sw / len(Zf), Sb / len(Zf)
                Sw = Sw + 1e-3 * Sw.diagonal().mean() * torch.eye(len(Sw), device=dev)
                fis = torch.trace(torch.linalg.solve(Sw, Sb))
            total = total + w * (a.lam_cyc * cyc / V + a.lam_pair * pr / V + a.lam_fisher * fis)
            m.update(v2_cyc=float(cyc / V), v2_pair=float(pr / V), v2_fisher=float(fis), v2_V=float(V), v2_w=w)
        self.extra_metrics = {**getattr(self, "extra_metrics", {}), **m}
        return total


def main(argv=None) -> None:
    ap = TL.build_parser()
    ap.add_argument("--pairs", required=True, help="cross_dex.ours.uniform_pairs npz (paired scenes for L_pair / L_fisher)")
    ap.add_argument("--v2_batch", type=int, default=1024, help="uniform poses per hand per step for align / cycle")
    ap.add_argument("--lam", type=float, default=2.8333, help="E2* alignment weight")
    ap.add_argument("--align_warmup", type=int, default=1500)
    ap.add_argument("--lam_cyc", type=float, default=0.05)
    ap.add_argument("--cyc_noise", type=float, default=0.02)
    ap.add_argument("--lam_pair", type=float, default=0.05)
    ap.add_argument("--pair_scenes", type=int, default=256)
    ap.add_argument("--pair_res_mm", type=float, default=15.0)
    ap.add_argument("--lam_fisher", type=float, default=0.002)
    ap.add_argument("--v2_start", type=int, default=10000, help="step from which the v2 terms are on (codec: after E2*)")
    ap.add_argument("--v2_warmup", type=int, default=300)
    ap.add_argument("--keep_pinch", type=int, default=0, help="1 = keep XL-VLA's pinch losses (codec: none)")
    ap.add_argument("--fast", type=int, default=1)

    def setup(tr, args):
        if args.fast:
            FX.make_fast(tr)
            tr.fast = True
        if not args.keep_pinch:
            tr.config.lambda_dis, tr.config.lambda_dir = 0.0, 0.0
        tr.enable_local(0.0, 1.0, 1024)
        info = tr.enable_v2(args)
        keys = ("pairs", "v2_batch", "lam", "align_warmup", "lam_cyc", "cyc_noise", "lam_pair", "pair_scenes", "pair_res_mm",
                "lam_fisher", "v2_start", "v2_warmup", "keep_pinch", "fast")
        return {"method": "ours: xhand codec v2 losses (E2* Kabsch alignment + cycle + paired scenes + Fisher) on the XL-VLA "
                          "hand VAE, six hands, uniform poses (cross_dex.ours.xhand_v2_uniform)",
                **{k: getattr(args, k) for k in keys}, **info}

    TL.main(argv, parser=ap, trainer_cls=XhandV2Trainer, setup=setup)


if __name__ == "__main__":
    main()
