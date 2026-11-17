"""Faster XL-VLA latent training for ours (2026-11-17, user: "1~3번을 확인하고 ... 적용해서 다시 올려봐").

The step time is set by the number of small GPU kernels Python launches (one CPU core at 100 %, GPU ~20 %), not by
VRAM or core count. Same maths, fewer launches:

  (3) FastCanonicalHandModel: DexLatent FK (HandLatent.kinematics, one 4x4 chain product per joint, origin / axis
      tensors copied host -> device on every call) rewritten as cached device tensors + all joint rotations in one
      batched Rodrigues + one batched product per tree depth (~6 levels instead of ~25 joints).
  (1) FastXLMixin.step: CrossEmbodimentTrainer.step with, per target hand, the five source latents decoded and passed
      through FK as ONE batch (30 decoder + 30 FK calls -> 6 + 6); per-pair losses identical.
      fast_local_loss: same for the local-consistency loss.
  (2) CUDA graphs / torch.compile: see check_fast.py results in the run README (only used if it gives the same values).

DexLatent's original modules are not modified: these are subclasses, swapped in after the trainer is built.
``check_fast.py`` compares every term with the original code on the same inputs.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import torch
from torch.nn import functional as F

import HandLatent.model as M
from dexverse_xlvla import hands as H


class FastCanonicalHandModel(H.CanonicalHandModel):
    """Same outputs as CanonicalHandModel (fingertips, wrist in the canonical frame), batched FK."""

    def _plan(self, device, dtype):
        key = (str(device), dtype)
        cache = self.__dict__.setdefault("_fast_cache", {})
        if key in cache:
            return cache[key]
        dof = {j: i for i, j in enumerate(self.dof_joints)}
        order = list(self.traversal_order)
        # revolute angles = A @ dof_angles + b (mimic), then clamp to each joint's own limits (as the original)
        rev = [j for j in order if self.joint_specs[j].kind == "revolute"]
        A = torch.zeros(len(rev), len(self.dof_joints), dtype=torch.float64)
        b = torch.zeros(len(rev), dtype=torch.float64)
        lo, hi = [], []
        for r, j in enumerate(rev):
            s = self.joint_specs[j]
            if j in dof:
                A[r, dof[j]] = 1.0
            elif s.mimic_parent is not None and s.mimic_parent in dof:
                A[r, dof[s.mimic_parent]] = float(s.mimic_multiplier)
                b[r] = float(s.mimic_offset)
            else:
                raise RuntimeError(f"{self.hand_name}: revolute joint {j} is neither a DoF nor a mimic of one")
            lo.append(float(s.lower) if s.lower is not None else float("-inf"))
            hi.append(float(s.upper) if s.upper is not None else float("inf"))
        axes = torch.stack([self.joint_specs[j].axis.to(torch.float64) for j in rev]) if rev else torch.zeros(0, 3)
        axes = F.normalize(axes, dim=-1)
        # tree levels: depth(child) = depth(parent) + 1, root depth 0
        depth = {self.root_link: 0}
        pos = {self.root_link: (0, 0)}                       # link -> (level, index in level)
        levels: List[List[str]] = []
        for j in order:
            s = self.joint_specs[j]
            d = depth[s.parent] + 1
            depth[s.child] = d
            while len(levels) < d:
                levels.append([])
            levels[d - 1].append(j)
            pos[s.child] = (d, len(levels[d - 1]) - 1)
        rev_idx = {j: i for i, j in enumerate(rev)}
        plan_levels = []
        for k, js in enumerate(levels):
            par = torch.tensor([pos[self.joint_specs[j].parent][1] for j in js], dtype=torch.long)
            org = torch.stack([self.joint_specs[j].origin_transform.to(torch.float64) for j in js])
            ridx = torch.tensor([rev_idx.get(j, -1) for j in js], dtype=torch.long)
            plan_levels.append((par.to(device), org.to(device=device, dtype=dtype), ridx.to(device)))
        tip_pos = [pos[t] for t in self.tip_links]
        wrist_pos = pos[self.wrist_link]
        lower = self._lower.to(device=device, dtype=dtype)
        upper = self._upper.to(device=device, dtype=dtype)
        plan = dict(A=A.to(device=device, dtype=dtype), b=b.to(device=device, dtype=dtype),
                    lo=torch.tensor(lo, device=device, dtype=dtype), hi=torch.tensor(hi, device=device, dtype=dtype),
                    axes=axes.to(device=device, dtype=dtype), levels=plan_levels, tip_pos=tip_pos,
                    wrist_pos=wrist_pos, lower=lower, upper=upper, eye=torch.eye(4, device=device, dtype=dtype))
        cache[key] = plan
        return plan

    @staticmethod
    def _rodrigues(axes, ang):
        """(R,3) unit axes, (B,R) angles -> (B,R,3,3); same formula as HandLatent.kinematics.axis_angle_to_matrix."""
        c, s = torch.cos(ang)[..., None, None], torch.sin(ang)[..., None, None]
        x, y, z = axes[:, 0], axes[:, 1], axes[:, 2]
        zero = torch.zeros_like(x)
        Kx = torch.stack([torch.stack([zero, -z, y], -1), torch.stack([z, zero, -x], -1),
                          torch.stack([-y, x, zero], -1)], -2)                       # (R,3,3) cross-product matrix
        outer = axes[:, :, None] * axes[:, None, :]
        eye = torch.eye(3, device=axes.device, dtype=axes.dtype)
        return c * eye + (1 - c) * outer + s * Kx

    def _raw(self, normalized: torch.Tensor):
        squeeze = normalized.ndim == 1
        q = normalized.unsqueeze(0) if squeeze else normalized
        p = self._plan(q.device, q.dtype)
        B = q.shape[0]
        ang = (q.clamp(-1.0, 1.0) + 1.0) * 0.5 * (p["upper"] - p["lower"]) + p["lower"]
        rev = torch.maximum(torch.minimum(ang @ p["A"].T + p["b"], p["hi"]), p["lo"])          # (B,R)
        rot = self._rodrigues(p["axes"], rev) if rev.shape[1] else None
        level_T = [p["eye"].expand(B, 1, 4, 4)]
        for par, org, ridx in p["levels"]:
            jt = org.expand(B, -1, -1, -1)
            m = ridx >= 0
            if bool(m.any()):
                R = torch.zeros(B, len(ridx), 4, 4, device=q.device, dtype=q.dtype)
                R[..., 3, 3] = 1.0
                R[:, ~m] = p["eye"]
                R[:, m, :3, :3] = rot[:, ridx[m]]
                jt = jt @ R
            level_T.append(level_T[-1][:, par] @ jt)
        tips = torch.stack([level_T[lv][:, i, :3, 3] for lv, i in p["tip_pos"]], dim=1)
        wrist = level_T[p["wrist_pos"][0]][:, p["wrist_pos"][1]]
        return tips, wrist, squeeze


def make_fast(trainer) -> None:
    """Swap the trainer's FK models for the fast ones (same URDF, canonical frame measured again -> identical)."""
    fast = {h: FastCanonicalHandModel(h) for h in trainer.hand_names}
    trainer.registry.models = fast
    trainer.hand_models = fast


class FastXLMixin:
    """CrossEmbodimentTrainer.step with the decoder + FK of the cross-hand pinch loss batched per target hand."""

    fast = False

    def step(self) -> Dict[str, float]:
        if not self.fast:
            return super().step()
        cfg = self.config
        self.optimizer.zero_grad()
        batch_qpos = self._sample_training_batch()
        latents, latent_stats, source_tips = {}, {}, {}
        hand_recon_loss = torch.tensor(0.0, device=cfg.device)
        for name in self.hand_names:
            hand_input = batch_qpos[name]
            _, qpos_hand_gt = self._split_qpos(name, hand_input)
            latent_arm, latent_hand, _, qpos_hand_pred, hand_stats = self.autoencoders[name](hand_input)
            latents[name] = (latent_arm, latent_hand)
            latent_stats[name] = {"hand": hand_stats}
            hand_recon_loss = hand_recon_loss + F.mse_loss(qpos_hand_pred, qpos_hand_gt)
            source_tips[name] = self.hand_models[name].forward(hand_input)
        hand_recon_loss = hand_recon_loss / float(len(self.hand_names))
        reconstruction_loss = hand_recon_loss * cfg.rec_hand_weight

        # per target: decode all source latents at once (original order of the (source, target) loop kept for sums)
        fk_of: Dict[Tuple[str, str], torch.Tensor] = {}
        for t in self.hand_names:
            srcs = [s for s in self.hand_names if s != t and len(self.shared_pinch_pairs(s, t))]
            if not srcs:
                continue
            la = torch.cat([latents[s][0] for s in srcs])
            lh = torch.cat([latents[s][1] for s in srcs])
            arm_pred, hand_pred = self.autoencoders[t].decode_from_latents(la, lh)
            fk = self.hand_models[t].forward(self._merge_qpos(arm_pred.detach(), hand_pred))
            for s, chunk in zip(srcs, fk.split([latents[s][1].shape[0] for s in srcs])):
                fk_of[(s, t)] = chunk
        pd_tot = torch.tensor(0.0, device=cfg.device)
        pr_tot = torch.tensor(0.0, device=cfg.device)
        ew_tot = torch.tensor(0.0, device=cfg.device)
        n = 0
        for s in self.hand_names:
            for t in self.hand_names:
                if t == s or (s, t) not in fk_of:
                    continue
                d, r, w = M.compute_pinch_loss(source_tips[s], fk_of[(s, t)], self.shared_pinch_pairs(s, t),
                                               cfg.lambda_dis_exp)
                n += 1
                pd_tot = pd_tot + (d * w).mean()
                pr_tot = pr_tot + (r * w).mean()
                ew_tot = ew_tot + w.mean()
        zero = torch.tensor(0.0, device=cfg.device)
        pinch_distance, pinch_direction, exp_weight_mean = (pd_tot / n, pr_tot / n, ew_tot / n) if n else (zero, zero, zero)
        kl_total, comp = torch.tensor(0.0, device=cfg.device), 0
        for stats in latent_stats.values():
            for mean, logvar in stats.values():
                kl_total = kl_total + (-0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)).mean()
                comp += 1
        kl_loss = kl_total / float(comp) if comp else zero
        total_loss = (reconstruction_loss + cfg.lambda_dis * pinch_distance + cfg.lambda_dir * pinch_direction
                      + cfg.lambda_kl * kl_loss)
        total_loss.backward()
        self.optimizer.step()
        return {
            "loss_total": float(total_loss.detach().cpu()),
            "loss_rec_total": float(reconstruction_loss.detach().cpu()),
            "loss_rec_hand": float((hand_recon_loss * cfg.rec_hand_weight).detach().cpu()),
            "loss_pinch_dis": float(pinch_distance.detach().cpu()) * cfg.lambda_dis,
            "loss_pinch_dir": float(pinch_direction.detach().cpu()) * cfg.lambda_dir,
            "loss_kl": float(kl_loss.detach().cpu()) * cfg.lambda_kl,
            "exp_dis": float(exp_weight_mean.detach().cpu()),
        }


def fast_local_loss(tr, x_of=None, d_of=None):
    """cross_dex.ours.xlvla_local.LocalTrainer._local_loss with batched FK / decoding (same terms, same order).
    x_of / d_of: optional fixed inputs per hand (for the equality check)."""
    if not tr.w_local:
        return None
    dev = tr.config.device
    enc = lambda h, x: tr.autoencoders[h].encode(x)[1]
    dec = lambda h, z: tr.autoencoders[h].decode_from_latents(z[:, :0], z)[1]
    z, z2, lines, rel_self = {}, {}, {}, []
    for h in tr.hand_names:
        m = tr.hand_models[h]
        if x_of is None:
            half = (0.5 * (m._upper - m._lower)).to(dev).clamp_min(1e-6)
            x = torch.empty(tr.local_batch, m.dof_count(), device=dev).uniform_(-1, 1)
            d = torch.randn_like(x)
            d = d / d.pow(2).mean(-1, keepdim=True).sqrt() * torch.deg2rad(torch.tensor(tr.local_deg, device=dev)) / half
        else:
            x, d = x_of[h], d_of[h]
        x2 = (x + d).clamp(-1, 1)
        zz = enc(h, torch.cat([x, x2]))
        z[h], z2[h] = zz.split(len(x))
        o = dec(h, zz)
        o1, o2 = o.split(len(x))
        dout, din = o2 - o1, x2 - x
        rel_self.append((dout - din).pow(2).sum(-1).mean() / din.pow(2).sum(-1).mean().clamp_min(1e-12))
        t = m.forward(torch.cat([x, x2]))
        t1, t2 = t.split(len(x))
        lines[h] = (t2[:, 1:] - t2[:, :1]) - (t1[:, 1:] - t1[:, :1])
    rel = {}
    for t_ in tr.hand_names:
        srcs = [s for s in tr.hand_names if s != t_]
        mt = tr.hand_models[t_]
        zs = torch.cat([torch.cat([z[s], z2[s]]) for s in srcs])
        a = mt.forward(dec(t_, zs).clamp(-1, 1))
        for s, ch in zip(srcs, a.split([2 * len(z[s]) for s in srcs])):
            a1, a2 = ch.split(len(z[s]))
            dt = (a2[:, 1:] - a2[:, :1]) - (a1[:, 1:] - a1[:, :1])
            k = min(dt.shape[1], lines[s].shape[1])
            ds = lines[s][:, :k].detach()
            rel[(s, t_)] = (dt[:, :k] - ds).pow(2).sum(-1).mean() / ds.pow(2).sum(-1).mean().clamp_min(1e-12)
    rel_cross = [rel[(s, t_)] for s in tr.hand_names for t_ in tr.hand_names if s != t_]
    ls, lc = sum(rel_self) / len(rel_self), sum(rel_cross) / len(rel_cross)
    tr.extra_metrics = {**getattr(tr, "extra_metrics", {}), "local_self": float(ls), "local_cross": float(lc)}
    return tr.w_local * (ls + lc)
