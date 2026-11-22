"""Functional-residual signal check against the kinematic ceiling (RSS2027, 2026-11-23; dexlatent venv, GPU optional).

Same question as delta_knn.py, but measured without any learned shared space, so the space's alignment error cannot
masquerade as a functional difference. Situation x = object pose(s) in the canonical wrist frame (+ lift), as in
delta_knn.py. For sampled query frames of hand h:
  self   neighbours = hand h, other episodes                -> predicted fingertips tips_h(q_nb)
  cross  neighbours = other hands s, retargeted s -> h with DexPilot (cross_dex.ours.uniform_pairs.retarget, the
                      kinematic ceiling)                    -> predicted fingertips tips_h(DexPilot(q_nb))
Residual = actual fingertips tips_h(q_h) - mean predicted fingertips, canonical frame, mm (first 4 fingertips so 4- and
5-finger hands share one vector). Cross - self beyond neighbour distance = what kinematic retargeting misses.

  cd /workspace/codes/dexverse/DexLatent && PYTHONPATH=/workspace/codes/cross_dex python -m cross_dex.ours.diag.delta_knn_joint \
      --wrist_dir <wrist_calib dir> --out <json>
"""
import argparse
import json
import pickle

import numpy as np
import torch

from cross_dex.ours.diag.delta_knn import HANDS, SKIP, TASKS, hand_acc, quat_order, quat_to_R
from cross_dex.ours.uniform_pairs import retarget
from dexverse_xlvla import hands as H
from dexverse_xlvla.replay import action_names, pkl_path
from dexverse_xlvla.wrist import from_rot6d, load_wrists, rot6d

DP = {"huber_delta": 0.03, "project_dist": 0.03, "eta1": 1e-4, "eta2": 3e-2}
NT = 4      # fingertips compared (thumb + 3; present on every hand)


def task_data(task, models, wrist_dir, sub):
    X, Q, hand, ep = [], {}, [], []
    for hi, h in enumerate(HANDS):
        try:
            p = pkl_path(task, h)
        except FileNotFoundError:
            continue
        d = pickle.load(open(p, "rb"))
        m = models[h]
        ia = m.data_index(action_names(task, h))
        wr = load_wrists(f"{wrist_dir}/wrist_calib_{task}.json", {h: m})
        objs, qs = None, []
        for e, epi in enumerate(d["episodes"]):
            act = np.asarray(epi["actions"], dtype=np.float64)
            st = epi["states"]
            if objs is None:
                objs = sorted(k for k in st[0]["rigid_object"] if not k.startswith(SKIP))
                order = quat_order(st)
            idx = np.arange(0, min(len(act), len(st)), sub)
            vec = wr[h].vec_from_actions(act[idx, :6])
            Rw = from_rot6d(vec[:, 3:9])
            feats = []
            for o in objs:
                P = np.stack([np.asarray(st[i]["rigid_object"][o]["root_pose"]).reshape(-1) for i in idx])
                P0 = np.asarray(st[0]["rigid_object"][o]["root_pose"]).reshape(-1)
                rel_p = np.einsum("nji,nj->ni", Rw, P[:, :3] - vec[:, :3])
                rel_R = np.einsum("nji,njk->nik", Rw, quat_to_R(P[:, 3:7], order))
                feats += [rel_p / 0.03, rot6d(rel_R) / 0.3, (P[:, 2:3] - P0[2]) / 0.03]
            X.append(np.concatenate(feats, -1))
            qs.append(m.angles_to_normalized(torch.tensor(act[idx][:, ia], dtype=torch.float32)).clamp(-1, 1))
            hand += [hi] * len(idx)
            ep += [e] * len(idx)
        Q[hi] = torch.cat(qs)
    return np.concatenate(X).astype(np.float32), Q, np.array(hand), np.array(ep)


def tips(m, q, dev):
    with torch.no_grad():
        return m._forward_internal(q.to(dev))[0][:, :NT].cpu() * 1000.0       # mm, canonical frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrist_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--sub", type=int, default=3)
    ap.add_argument("--queries", type=int, default=300, help="per hand per task")
    ap.add_argument("--iters", type=int, default=300)
    a = ap.parse_args()
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models = {h: H.CanonicalHandModel(h) for h in HANDS}
    for m in models.values():
        if hasattr(m, "to"):
            m.to(dev)
    out, mean_delta = {}, {}
    for task in TASKS:
        X, Q, hand, ep = task_data(task, models, a.wrist_dir, a.sub)
        off = {h: np.flatnonzero(hand == h)[0] for h in Q}                   # global row -> per-hand row
        lift = X[:, [c for c in range(X.shape[1]) if (c + 1) % 10 == 0]].max(1) > 1 / 3.0
        Xt = torch.from_numpy(X)
        qry = np.concatenate([rng.choice(np.flatnonzero(hand == h), min(a.queries, (hand == h).sum()), replace=False)
                              for h in Q])
        D = torch.cdist(Xt[qry], Xt)
        same_h = torch.from_numpy(hand[qry][:, None] == hand[None])
        same_e = same_h & torch.from_numpy(ep[qry][:, None] == ep[None])
        nb, dist = {}, {}
        for s, mask in (("self", same_h & ~same_e), ("cross", ~same_h)):
            dd, ii = D.masked_fill(~mask, float("inf")).topk(a.k, largest=False)
            nb[s], dist[s] = ii.numpy(), dd.mean(1).numpy()
        # retarget every needed cross neighbour frame s -> h (one batch per hand pair)
        need = {}
        for qi, g in enumerate(qry):
            for n in nb["cross"][qi]:
                need.setdefault((hand[n], hand[g]), set()).add(n)
        rt, rt_res = {}, []
        for (s, h), rows in need.items():
            rows = sorted(rows)
            qs = Q[s][np.array(rows) - off[s]].to(dev)
            q, rms, _ = retarget(models[HANDS[s]], models[HANDS[h]], qs, a.iters, 1, 0.05, 1e-6, DP)
            tt = tips(models[HANDS[h]], q.detach().clamp(-1, 1), dev)
            for r, t in zip(rows, tt):
                rt[(r, h)] = t
            rt_res.append(rms.detach().cpu().numpy())
        R = {"self": [], "cross": []}
        for qi, g in enumerate(qry):
            h = hand[g]
            act = tips(models[HANDS[h]], Q[h][[g - off[h]]], dev)[0]
            ps = tips(models[HANDS[h]], Q[h][nb["self"][qi] - off[h]], dev).mean(0)
            pc = torch.stack([rt[(n, h)] for n in nb["cross"][qi]]).mean(0)
            R["self"].append((act - ps).reshape(-1).numpy())
            R["cross"].append((act - pc).reshape(-1).numpy())
        R = {s: np.stack(v) for s, v in R.items()}
        hq, eq, lq = hand[qry], ep[qry], lift[qry]
        rows = {}
        for ph, sel in (("all", np.ones(len(qry), bool)), ("lifted", lq)):
            if sel.sum() < 50:
                continue
            ns = np.linalg.norm(R["self"][sel].reshape(sel.sum(), NT, 3), axis=-1).mean(1)
            nc = np.linalg.norm(R["cross"][sel].reshape(sel.sum(), NT, 3), axis=-1).mean(1)
            A = np.stack([np.ones(2 * sel.sum()), np.r_[dist["self"][sel], dist["cross"][sel]],
                          np.r_[np.zeros(sel.sum()), np.ones(sel.sum())]], 1)
            coef = np.linalg.lstsq(A, np.r_[ns, nc], rcond=None)[0]
            rows[ph] = {"n": int(sel.sum()), "tip_mm_self_med": round(float(np.median(ns)), 1),
                        "tip_mm_cross_med": round(float(np.median(nc)), 1),
                        "d_self_med": round(float(np.median(dist["self"][sel])), 3),
                        "d_cross_med": round(float(np.median(dist["cross"][sel])), 3),
                        "cross_effect_mm_ctrl_dist": round(float(coef[2]), 1),
                        "hand_acc_cross_resid": hand_acc(R["cross"][sel], hq[sel], eq[sel]),
                        "hand_acc_self_resid": hand_acc(R["self"][sel], hq[sel], eq[sel]),
                        "per_hand_mm_cross_self": {HANDS[h]: [round(float(np.median(nc[hq[sel] == h])), 1),
                                                              round(float(np.median(ns[hq[sel] == h])), 1)]
                                                   for h in np.unique(hq[sel])}}
        rows["dexpilot_res_mm_med"] = round(float(np.median(np.concatenate(rt_res))), 2)
        out[task] = rows
        for h in np.unique(hq):
            sel = (hq == h) & lq
            if sel.sum() >= 20:
                mean_delta[(HANDS[h], task)] = R["cross"][sel].mean(0)
        print(task, json.dumps(rows), flush=True)
    keys = list(mean_delta)
    same, diff = [], []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            (h1, t1), (h2, t2) = keys[i], keys[j]
            if t1 != t2:
                u, v = mean_delta[keys[i]], mean_delta[keys[j]]
                (same if h1 == h2 else diff).append(float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-9)))
    out["task_consistency_cos"] = {"same_hand_diff_task": round(float(np.mean(same)), 3) if same else None,
                                   "diff_hand_diff_task": round(float(np.mean(diff)), 3) if diff else None,
                                   "n_same": len(same), "n_diff": len(diff)}
    out["mean_cross_resid_mm_lifted"] = {f"{h}|{t}": [round(float(x), 1) for x in v] for (h, t), v in mean_delta.items()}
    print(json.dumps(out["task_consistency_cos"]), flush=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
