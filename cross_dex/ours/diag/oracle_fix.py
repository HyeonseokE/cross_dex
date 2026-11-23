"""Oracle functional correction for cross-hand replay (RSS2027, 2026-11-23): the upper bound of module (2).

Hand s's episode replayed on hand t: fingers = DexPilot(s -> t) + Delta_t(x), where x is the situation of each frame
(object pose in the canonical wrist frame + lift, delta_knn.py features) and
  Delta_t(x) = mean q_t over t's OWN demo frames nearest to x  (t's other episodes; the same episode index is excluded
                                                               so a same-layout scene cannot be copied)
             - mean DexPilot(q_o -> t) over the nearest frames of the other hands o != t
i.e. what t's operator did differently from the kinematic retargeting of the other hands in that situation. It uses
t's task data, so it is an oracle (module (2) must predict it without them). Wrist unchanged (s's canonical pose).
Delta is computed every `sub` frames, linearly interpolated, then smoothed over `smooth` frames.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import torch

from cross_dex.ours.diag.delta_knn import HANDS, SKIP, quat_order, quat_to_R
from cross_dex.ours.diag.delta_knn_joint import DP, task_data
from cross_dex.ours.uniform_pairs import retarget
from dexverse_xlvla import hands as H
from dexverse_xlvla.replay import pkl_path
from dexverse_xlvla.wrist import from_rot6d, load_wrists, rot6d

_BANK = {}


def bank(task, wrist_dir, sub):
    if task not in _BANK:
        models = {h: H.CanonicalHandModel(h) for h in HANDS}
        X, Q, hand, ep = task_data(task, models, wrist_dir, sub)
        _BANK[task] = (models, torch.from_numpy(X), Q, hand, ep)
    return _BANK[task]


def episode_x(task, src, k, wrist_file, model):
    epi = pickle.load(open(pkl_path(task, src), "rb"))["episodes"][k]
    act, st = np.asarray(epi["actions"], dtype=np.float64), epi["states"]
    objs = sorted(o for o in st[0]["rigid_object"] if not o.startswith(SKIP))
    order = quat_order(st)
    wr = load_wrists(wrist_file, {src: model})[src]
    idx = np.arange(len(act))
    vec = wr.vec_from_actions(act[:, :6])
    Rw = from_rot6d(vec[:, 3:9])
    feats = []
    for o in objs:
        P = np.stack([np.asarray(st[min(i, len(st) - 1)]["rigid_object"][o]["root_pose"]).reshape(-1) for i in idx])
        P0 = np.asarray(st[0]["rigid_object"][o]["root_pose"]).reshape(-1)
        rel_p = np.einsum("nji,nj->ni", Rw, P[:, :3] - vec[:, :3])
        rel_R = np.einsum("nji,njk->nik", Rw, quat_to_R(P[:, 3:7], order))
        feats += [rel_p / 0.03, rot6d(rel_R) / 0.3, (P[:, 2:3] - P0[2]) / 0.03]
    return np.concatenate(feats, -1).astype(np.float32)


def correct(task, src, dst, k, qb, wrist_file, dev, sub=3, knn=8, smooth=31, iters=300):
    """qb (T, dof_t) DexPilot result for episode k of src -> corrected (T, dof_t), stats."""
    models, X, Q, hand, ep = bank(task, os.path.dirname(wrist_file), sub)
    hs, ht = HANDS.index(src), HANDS.index(dst)
    off = {h: int(np.flatnonzero(hand == h)[0]) for h in Q}
    x = torch.from_numpy(episode_x(task, src, k, wrist_file, models[src]))
    T = len(qb)
    qi = np.arange(0, T, sub)
    D = torch.cdist(x[qi], X)
    own = torch.from_numpy((hand == ht) & (ep != k))
    oth = torch.from_numpy(hand != ht)
    d_own, nb_own = D.masked_fill(~own, float("inf")).topk(knn, largest=False)
    d_oth, nb_oth = D.masked_fill(~oth, float("inf")).topk(knn, largest=False)
    nb_oth = nb_oth.numpy()
    rt = {}
    for o in np.unique(hand[nb_oth]):
        rows = sorted(set(int(r) for r in nb_oth.reshape(-1) if hand[r] == o))
        q, _, _ = retarget(models[HANDS[o]], models[dst], Q[o][np.array(rows) - off[o]].to(dev), iters, 1, 0.05, 1e-6, DP)
        for r, v in zip(rows, q.detach().cpu().clamp(-1, 1)):
            rt[r] = v
    mean_own = Q[ht][nb_own.numpy() - off[ht]].mean(1)                                  # (n, dof_t)
    mean_oth = torch.stack([torch.stack([rt[int(r)] for r in row]).mean(0) for row in nb_oth])
    delta = (mean_own - mean_oth).numpy()
    full = np.stack([np.interp(np.arange(T), qi, delta[:, j]) for j in range(delta.shape[1])], 1)
    if smooth > 1:
        ker = np.ones(smooth) / smooth
        full = np.stack([np.convolve(np.pad(full[:, j], smooth // 2, mode="edge"), ker, "valid") for j in range(full.shape[1])], 1)
    out = (qb + torch.from_numpy(full).float()).clamp(-1, 1)
    with torch.no_grad():
        tb = models[dst]._forward_internal(qb.to(dev))[0][:, :4]
        to = models[dst]._forward_internal(out.to(dev))[0][:, :4]
    stats = {"delta_norm_mean": round(float(np.abs(full).mean()), 4),
             "tip_shift_mm_mean": round(float((to - tb).norm(dim=-1).mean() * 1000), 1),
             "knn_dist_own_med": round(float(d_own.mean(1).median()), 3),
             "knn_dist_oth_med": round(float(d_oth.mean(1).median()), 3)}
    return out, stats
