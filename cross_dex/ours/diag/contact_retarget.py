"""Oracle-intent test (RSS2027, 2026-11-20): retarget hand s -> hand t by matching WORLD fingertip positions.

Hand s's episode is replayed on hand t in the same scene, so world fingertip positions = object-frame contact targets
(the hand-agnostic "intent": where on the object each finger goes). DexPilot keeps t's canonical wrist pose = s's and
matches wrist-relative finger vectors (scaled by hand size), so t's fingertips land elsewhere when the hands differ in
size / shape. Here the wrist is free too: per frame, t's canonical frame = s's canonical frame x D (D = small rigid
transform) and t's fingers q_t minimise

  sum_i |D . tip_t,i(q_t) - tip_s,i(q_s)|^2  (s's canonical frame, fingertips t lacks drop out)
  + w_trans |D_trans|^2 + w_rot |D_rot|^2   (stay near s's wrist pose)
  + w_smooth (|dD/dt|^2 + |dq/dt|^2)         (whole episode solved at once)

initialised from DexPilot (q) and D = identity.
"""
from __future__ import annotations

import numpy as np
import torch

W = {"trans": 0.05, "rot": 0.01, "smooth_d": 1.0, "smooth_q": 1e-4, "reg_q": 1e-6}


def so3_exp(w):
    th = w.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    k = w / th
    K = torch.zeros(*w.shape[:-1], 3, 3, device=w.device, dtype=w.dtype)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 2] = -k[..., 2], k[..., 1], -k[..., 0]
    K = K - K.transpose(-1, -2)
    th = th[..., None]
    return torch.eye(3, device=w.device, dtype=w.dtype) + th.sin() * K + (1 - th.cos()) * K @ K


def tip_match(ma, mb, qa, q0, iters=600, lr=0.01, w=W):
    """qa (T, dof_s), q0 (T, dof_t) normalized (DexPilot init). Returns q_t (T, dof_t), D (T, 4, 4) t-canon -> s-canon,
    stats (fingertip error mm before / after, wrist shift mm, wrist rotation deg)."""
    with torch.no_grad():
        ts = ma._forward_internal(qa)[0]
    k = min(ts.shape[1], mb.tip_count())
    ts = ts[:, :k]
    u = torch.atanh(q0.clamp(-0.999, 0.999)).clone().requires_grad_(True)
    d = torch.zeros(len(qa), 6, device=qa.device, requires_grad=True)       # rotvec, translation (m)
    opt = torch.optim.Adam([u, d], lr=lr)

    def tips_t(q, d):
        tt = mb._forward_internal(q)[0][:, :k]
        return tt @ so3_exp(d[:, :3]).transpose(-1, -2) + d[:, None, 3:]

    with torch.no_grad():
        e0 = (tips_t(q0, torch.zeros_like(d)) - ts).norm(dim=-1)
    for _ in range(iters):
        opt.zero_grad()
        q = torch.tanh(u)
        err = (tips_t(q, d) - ts).pow(2).sum(-1).mean(-1)
        loss = (err.mean() + w["trans"] * d[:, 3:].pow(2).sum(-1).mean() + w["rot"] * d[:, :3].pow(2).sum(-1).mean()
                + w["smooth_d"] * d.diff(dim=0).pow(2).sum(-1).mean() + w["smooth_q"] * q.diff(dim=0).pow(2).sum(-1).mean()
                + w["reg_q"] * q.pow(2).sum(-1).mean())
        loss.backward()
        opt.step()
    with torch.no_grad():
        q = torch.tanh(u)
        e1 = (tips_t(q, d) - ts).norm(dim=-1)
        D = torch.eye(4, device=qa.device).repeat(len(qa), 1, 1)
        D[:, :3, :3], D[:, :3, 3] = so3_exp(d[:, :3]), d[:, 3:]
    stats = {"tip_mm_dexpilot": round(float(e0.mean() * 1000), 1), "tip_mm_contact": round(float(e1.mean() * 1000), 1),
             "wrist_shift_mm_mean": round(float(d[:, 3:].norm(dim=-1).mean() * 1000), 1),
             "wrist_rot_deg_mean": round(float(np.degrees(d[:, :3].norm(dim=-1).mean().item())), 1)}
    return q.detach().cpu(), D.detach().cpu().numpy().astype(np.float64), stats


def compose_wrist(vec_s, D):
    """s's canonical pose vectors (T, 9) x D (T, 4, 4) -> t's canonical pose vectors (T, 9)."""
    from dexverse_xlvla.wrist import from_rot6d, rot6d
    Ws = np.tile(np.eye(4), (len(vec_s), 1, 1))
    Ws[:, :3, :3], Ws[:, :3, 3] = from_rot6d(vec_s[:, 3:9]), vec_s[:, :3]
    Wt = Ws @ D
    return np.concatenate([Wt[:, :3, 3], rot6d(Wt[:, :3, :3])], -1)
