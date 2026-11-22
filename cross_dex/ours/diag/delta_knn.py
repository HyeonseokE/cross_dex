"""Functional-residual signal check without training (RSS2027, 2026-11-23; dexlatent venv, CPU).

Question: does hand h's teleop action deviate from what OTHER hands did in the same situation, beyond the normal
episode-to-episode variation of hand h itself? If yes, a correction module (module 2) has something to learn.

Per task, every demo step (subsampled) becomes a query with
  x = object pose(s) in the canonical wrist frame (pos /3 cm, rot6d /0.3) + object lift above its start (/3 cm)
  y = shared-space finger latent E_h(q_h) of the given checkpoint (standardised per dim over all hands)
kNN (k neighbours in x, same task):
  self  = hand h, other episodes      -> r_self  = |y - mean(y_nb)|   (noise: operator / episode variation)
  cross = all other hands             -> r_cross = |y - mean(y_nb)|   (noise + hand-specific deviation)
Reported: median residuals and neighbour distances, the cross effect after controlling for neighbour distance
(least squares r ~ a + b d + c [cross]), hand-ID accuracy of the residual vectors (chance = 1/#hands), and whether a
hand's mean cross residual points the same way across tasks (cosine same hand vs different hands).

  cd /workspace/codes/dexverse/DexLatent && PYTHONPATH=/workspace/codes/cross_dex python -m cross_dex.ours.diag.delta_knn \
      --ckpt <latent .pt> --wrist_dir <wrist_calib dir> --out <json>
"""
import argparse
import json
import pickle

import numpy as np
import torch

from dexverse_xlvla.eval_latent import enc, load
from dexverse_xlvla.replay import action_names, pkl_path
from dexverse_xlvla.wrist import from_rot6d, load_wrists, rot6d

HANDS = ["shadow", "allegro", "inspire", "leap", "sharpa", "wuji"]
TASKS = ["Dexverse-GraspPan-v0", "Dexverse-GraspBleach-v0", "Dexverse-GraspKettle-v0", "Dexverse-FunctionalHammerStrike-v0"]
SKIP = ("table", "success_marker")


def quat_to_R(q, order):
    if order == "xyzw":
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    else:
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def quat_order(states):
    """Table / markers rest upright with identity rotation: the slot holding ~1 is w."""
    q = np.asarray(states[0]["rigid_object"]["table"]["root_pose"]).reshape(-1)[3:7]
    return "xyzw" if abs(q[3]) > abs(q[0]) else "wxyz"


def task_data(task, tr, wrist_dir, sub):
    X, Y, hand, ep = [], [], [], []
    models = tr.hand_models
    for hi, h in enumerate(HANDS):
        try:
            p = pkl_path(task, h)
        except FileNotFoundError:
            continue
        d = pickle.load(open(p, "rb"))
        names = action_names(task, h)
        m = models[h]
        ia = m.data_index(names)
        wr = load_wrists(f"{wrist_dir}/wrist_calib_{task}.json", {h: m})
        if h not in wr:
            continue
        objs = None
        for e, epi in enumerate(d["episodes"]):
            act = np.asarray(epi["actions"], dtype=np.float64)
            st = epi["states"]
            if objs is None:
                objs = sorted(k for k in st[0]["rigid_object"] if not k.startswith(SKIP))
                order = quat_order(st)
            T = min(len(act), len(st))
            idx = np.arange(0, T, sub)
            vec = wr[h].vec_from_actions(act[idx, :6])                       # (n, 9) canonical wrist pose, world
            Rw = from_rot6d(vec[:, 3:9])
            feats = []
            for o in objs:
                P = np.stack([np.asarray(st[i]["rigid_object"][o]["root_pose"]).reshape(-1) for i in idx])
                P0 = np.asarray(st[0]["rigid_object"][o]["root_pose"]).reshape(-1)
                Ro = quat_to_R(P[:, 3:7], order)
                rel_p = np.einsum("nji,nj->ni", Rw, P[:, :3] - vec[:, :3])
                rel_R = np.einsum("nji,njk->nik", Rw, Ro)
                feats += [rel_p / 0.03, rot6d(rel_R) / 0.3, (P[:, 2:3] - P0[2]) / 0.03]
            X.append(np.concatenate(feats, -1))
            q = m.angles_to_normalized(torch.tensor(act[idx][:, ia], dtype=torch.float32)).clamp(-1, 1)
            with torch.no_grad():
                Y.append(enc(tr, h, q).numpy())
            hand += [hi] * len(idx)
            ep += [e] * len(idx)
    X, Y = np.concatenate(X), np.concatenate(Y)
    Y = (Y - Y.mean(0)) / (Y.std(0) + 1e-6)
    return X.astype(np.float32), Y.astype(np.float32), np.array(hand), np.array(ep)


def knn(X, Y, hand, ep, k, chunk=2048):
    Xt, n = torch.from_numpy(X), len(X)
    ht, et = torch.from_numpy(hand), torch.from_numpy(ep)
    res = {s: (np.zeros(n), np.zeros(n), np.zeros_like(Y)) for s in ("self", "cross")}
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        D = torch.cdist(Xt[a:b], Xt)
        same_h = ht[a:b, None] == ht[None]
        same_e = same_h & (et[a:b, None] == et[None])
        for s, mask in (("self", same_h & ~same_e), ("cross", ~same_h)):
            Dm = D.masked_fill(~mask, float("inf"))
            dist, nb = Dm.topk(k, largest=False)
            mean = torch.from_numpy(Y)[nb].mean(1).numpy()
            res[s][2][a:b] = Y[a:b] - mean
            res[s][0][a:b] = np.linalg.norm(res[s][2][a:b], axis=-1)
            res[s][1][a:b] = dist.mean(1).numpy()
    return res


def hand_acc(V, hand, ep, folds=5):
    """Softmax regression hand-ID accuracy, split by episode (no episode in both train and test)."""
    V = torch.from_numpy(V.astype(np.float32))
    hs = np.unique(hand)
    lab = torch.from_numpy(np.searchsorted(hs, hand))
    accs = []
    for f in range(folds):
        te = torch.from_numpy(ep % folds == f)
        W = torch.zeros(V.shape[1], len(hs), requires_grad=True)
        bb = torch.zeros(len(hs), requires_grad=True)
        opt = torch.optim.Adam([W, bb], lr=0.05)
        for _ in range(300):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(V[~te] @ W + bb, lab[~te]) + 1e-3 * W.pow(2).sum()
            loss.backward()
            opt.step()
        with torch.no_grad():
            accs.append(float(((V[te] @ W + bb).argmax(1) == lab[te]).float().mean()))
    return round(float(np.mean(accs)), 3), round(1 / len(hs), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wrist_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--sub", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(8)
    tr = load(a.ckpt)
    for m in tr.autoencoders.values():
        m.eval()
    out, mean_delta = {}, {}
    for task in TASKS:
        X, Y, hand, ep = task_data(task, tr, a.wrist_dir, a.sub)
        r = knn(X, Y, hand, ep, a.k)
        held = X[:, [c for c in range(X.shape[1]) if (c + 1) % 10 == 0]].max(1) > 1 / 3.0   # object lifted > 1 cm
        rows = {}
        for ph, sel in (("all", np.ones(len(X), bool)), ("lifted", held)):
            if sel.sum() < 50:
                continue
            rs, ds = r["self"][0][sel], r["self"][1][sel]
            rc, dc = r["cross"][0][sel], r["cross"][1][sel]
            A = np.stack([np.ones(2 * sel.sum()), np.concatenate([ds, dc]), np.r_[np.zeros(sel.sum()), np.ones(sel.sum())]], 1)
            coef = np.linalg.lstsq(A, np.concatenate([rs, rc]), rcond=None)[0]
            rows[ph] = {"n": int(sel.sum()), "r_self_med": round(float(np.median(rs)), 3),
                        "r_cross_med": round(float(np.median(rc)), 3), "d_self_med": round(float(np.median(ds)), 3),
                        "d_cross_med": round(float(np.median(dc)), 3), "cross_effect_ctrl_dist": round(float(coef[2]), 3),
                        "hand_acc_cross_resid": hand_acc(r["cross"][2][sel], hand[sel], ep[sel]),
                        "hand_acc_self_resid": hand_acc(r["self"][2][sel], hand[sel], ep[sel]),
                        "per_hand_r_cross_over_self": {HANDS[h]: round(float(np.median(rc[hand[sel] == h]) /
                                                                       np.median(rs[hand[sel] == h])), 3)
                                                       for h in np.unique(hand[sel])}}
        out[task] = rows
        for h in np.unique(hand):
            sel = (hand == h) & held
            if sel.sum() >= 20:
                mean_delta[(HANDS[h], task)] = r["cross"][2][sel].mean(0)
        print(task, json.dumps(rows), flush=True)
    keys = list(mean_delta)
    same, diff = [], []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            (h1, t1), (h2, t2) = keys[i], keys[j]
            if t1 == t2:
                continue
            u, v = mean_delta[keys[i]], mean_delta[keys[j]]
            c = float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9))
            (same if h1 == h2 else diff).append(c)
    out["task_consistency_cos"] = {"same_hand_diff_task": round(float(np.mean(same)), 3) if same else None,
                                   "diff_hand_diff_task": round(float(np.mean(diff)), 3) if diff else None,
                                   "n_same": len(same), "n_diff": len(diff)}
    print(json.dumps(out["task_consistency_cos"]), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
