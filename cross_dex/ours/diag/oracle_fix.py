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


# ---------------------------------------------------------------------------------------------------------------------
# v2 (2026-11-23): wrist + fingers, gated. The v1 situation (object pose in the wrist frame) already fixes the
# wrist-object relation, so it cannot tell where t places its wrist. v2 matches on the object's own state and the
# episode progress, and corrects both
#   fingers  Delta_q = mean q_t(own nb) - mean DexPilot(q_o -> t)(other nb)
#   wrist    Delta_p = mean wrist position in the object frame (own nb) - same (other nb), applied in s's object frame
# Frames whose own neighbours are far (no comparable t demo) get no correction (linear gate between TAU).
TAU = (1.0, 2.0)
_BANK2 = {}


def _ep_feats(epi, wr, sub, objs, order):
    act, st = np.asarray(epi["actions"], dtype=np.float64), epi["states"]
    T = min(len(act), len(st))
    idx = np.arange(0, T, sub)
    vec = wr.vec_from_actions(act[idx, :6])
    x, wpo = [], None
    for j, o in enumerate(objs):
        P = np.stack([np.asarray(st[i]["rigid_object"][o]["root_pose"]).reshape(-1) for i in idx])
        P0 = np.asarray(st[0]["rigid_object"][o]["root_pose"]).reshape(-1)
        Ro, R0 = quat_to_R(P[:, 3:7], order), quat_to_R(P0[3:7], order)
        x += [(P[:, :3] - P0[:3]) / 0.03, rot6d(np.einsum("ji,njk->nik", R0, Ro)) / 0.3]
        if j == 0:
            wpo = np.einsum("nji,nj->ni", Ro, vec[:, :3] - P[:, :3])          # wrist position in object frame (m)
            Rmain = Ro
    x.append((idx / max(T - 1, 1))[:, None] / 0.1)
    return np.concatenate(x, -1).astype(np.float32), wpo, Rmain, idx, act


def bank2(task, wrist_dir, sub):
    if task in _BANK2:
        return _BANK2[task]
    models = {h: H.CanonicalHandModel(h) for h in HANDS}
    X, W, Q, hand, ep = [], [], {}, [], []
    for hi, h in enumerate(HANDS):
        try:
            d = pickle.load(open(pkl_path(task, h), "rb"))
        except FileNotFoundError:
            continue
        from dexverse_xlvla.replay import action_names
        m = models[h]
        ia = m.data_index(action_names(task, h))
        wr = load_wrists(f"{wrist_dir}/wrist_calib_{task}.json", {h: m})[h]
        st0 = d["episodes"][0]["states"]
        objs = sorted(o for o in st0[0]["rigid_object"] if not o.startswith(SKIP))
        order = quat_order(st0)
        qs = []
        for e, epi in enumerate(d["episodes"]):
            x, wpo, _, idx, act = _ep_feats(epi, wr, sub, objs, order)
            X.append(x); W.append(wpo)
            qs.append(m.angles_to_normalized(torch.tensor(act[idx][:, ia], dtype=torch.float32)).clamp(-1, 1))
            hand += [hi] * len(idx); ep += [e] * len(idx)
        Q[hi] = torch.cat(qs)
    _BANK2[task] = (models, torch.from_numpy(np.concatenate(X)), np.concatenate(W), Q, np.array(hand), np.array(ep))
    return _BANK2[task]


def _smooth(full, smooth):
    ker = np.ones(smooth) / smooth
    return np.stack([np.convolve(np.pad(full[:, j], smooth // 2, mode="edge"), ker, "valid") for j in range(full.shape[1])], 1)


def correct2(task, src, dst, k, qb, wrist_file, dev, sub=3, knn=8, smooth=31, iters=300):
    """-> corrected fingers (T, dof_t), wrist position offset in world (T, 3), stats."""
    models, X, W, Q, hand, ep = bank2(task, os.path.dirname(wrist_file), sub)
    ht = HANDS.index(dst)
    off = {h: int(np.flatnonzero(hand == h)[0]) for h in Q}
    epi = pickle.load(open(pkl_path(task, src), "rb"))["episodes"][k]
    st0 = epi["states"]
    objs = sorted(o for o in st0[0]["rigid_object"] if not o.startswith(SKIP))
    wr = load_wrists(wrist_file, {src: models[src]})[src]
    x, _, Rmain, qi, _ = _ep_feats(epi, wr, sub, objs, quat_order(st0))
    T = len(qb)
    qi, x, Rmain = qi[qi < T], x[qi < T], Rmain[qi < T]
    D = torch.cdist(torch.from_numpy(x), X)
    own = torch.from_numpy((hand == ht) & (ep != k))
    oth = torch.from_numpy(hand != ht)
    d_own, nb_own = D.masked_fill(~own, float("inf")).topk(knn, largest=False)
    d_oth, nb_oth = D.masked_fill(~oth, float("inf")).topk(knn, largest=False)
    nb_own, nb_oth = nb_own.numpy(), nb_oth.numpy()
    rt = {}
    for o in np.unique(hand[nb_oth]):
        rows = sorted(set(int(r) for r in nb_oth.reshape(-1) if hand[r] == o))
        q, _, _ = retarget(models[HANDS[o]], models[dst], Q[o][np.array(rows) - off[o]].to(dev), iters, 1, 0.05, 1e-6, DP)
        for r, v in zip(rows, q.detach().cpu().clamp(-1, 1)):
            rt[r] = v
    dq = (Q[ht][nb_own - off[ht]].mean(1) - torch.stack([torch.stack([rt[int(r)] for r in row]).mean(0) for row in nb_oth])).numpy()
    dp_obj = W[nb_own].mean(1) - W[nb_oth].mean(1)                                     # (n, 3) object frame
    dp = np.einsum("nij,nj->ni", Rmain, dp_obj)                                         # world
    g = np.clip((TAU[1] - d_own.mean(1).numpy()) / (TAU[1] - TAU[0]), 0, 1)[:, None]
    dq, dp = dq * g, dp * g
    interp = lambda a: np.stack([np.interp(np.arange(T), qi, a[:, j]) for j in range(a.shape[1])], 1)
    fq, fp = _smooth(interp(dq), smooth), _smooth(interp(dp), smooth)
    out = (qb + torch.from_numpy(fq).float()).clamp(-1, 1)
    with torch.no_grad():
        tb = models[dst]._forward_internal(qb.to(dev))[0][:, :4]
        to = models[dst]._forward_internal(out.to(dev))[0][:, :4]
    stats = {"tip_shift_mm_mean": round(float((to - tb).norm(dim=-1).mean() * 1000), 1),
             "wrist_shift_mm_mean": round(float(np.linalg.norm(fp, axis=-1).mean() * 1000), 1),
             "gate_mean": round(float(g.mean()), 3),
             "knn_dist_own_med": round(float(d_own.mean(1).median()), 3),
             "knn_dist_oth_med": round(float(d_oth.mean(1).median()), 3)}
    return out, fp, stats
