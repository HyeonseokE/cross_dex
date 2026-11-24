"""Training targets of the functional-correction module C for an unseen hand (RSS2027, 2026-11-23; dexlatent venv).

Zero-shot target hand (default shadow) is never used. For every frame of every SOURCE hand h's demos of one task:
  x      situation = object pose(s) in the canonical wrist frame + lift (cross_dex.ours.diag.delta_knn features)
  k_ref  = mean_n E_h(DexPilot(q_n -> h)) over the k nearest frames n (in x) of the OTHER source hands
           = the shared-space action the other hands would give in this situation, as a kinematic retarget onto h
  delta  = E_h(q_h) - k_ref     (what h's operator did beyond the kinematic retarget, in the policy's latent space)
Also the hand descriptors (URDF only) of all hands, target included. One process per source hand (--hand), so the
DexPilot retargets of the five hands run in parallel.
-> <out>/<task>/<hand>.npz: x, k_ref, k_self, delta, ep, knn_dist  and  <out>/descriptors.npz

  cd /workspace/codes/dexverse/DexLatent && PYTHONPATH=/workspace/codes/cross_dex python -m cross_dex.ours.fc.delta_data \
      --ckpt <space .pt> --wrist_dir <dir> --hand allegro --out <dir> [--target shadow]
"""
import argparse
import os

import numpy as np
import torch

from cross_dex.ours.diag.delta_knn import HANDS
from cross_dex.ours.diag.delta_knn_joint import DP, task_data
from cross_dex.ours.uniform_pairs import retarget
from dexverse_xlvla.eval_latent import enc, load

NF = 5


def descriptor(m):
    """URDF-only hand descriptor: fingertips (canonical frame, m) at normalized q = -1, 0, +1 and the open pose, little
    finger zero-padded on 4-finger hands, + a present-finger mask and the DoF count."""
    poses = [-torch.ones(1, m.dof_count()), torch.zeros(1, m.dof_count()), torch.ones(1, m.dof_count()),
             m.open_qpos().unsqueeze(0)]
    with torch.no_grad():
        tips = [m._forward_internal(p)[0][0].numpy() for p in poses]
    n = tips[0].shape[0]
    pad = np.zeros((len(poses), NF, 3), np.float32)
    pad[:, :n] = np.stack(tips)[:, :NF]
    mask = np.zeros(NF, np.float32)
    mask[:n] = 1
    return np.concatenate([pad.reshape(-1) * 10, mask, [m.dof_count() / 22.0]]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wrist_dir", required=True)
    ap.add_argument("--hand", required=True)
    ap.add_argument("--target", default="shadow")
    ap.add_argument("--task", default="Dexverse-GraspKettle-v0")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--sub", type=int, default=3)
    ap.add_argument("--iters", type=int, default=300)
    a = ap.parse_args()
    assert a.hand != a.target
    torch.manual_seed(42)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = load(a.ckpt)
    for m in tr.autoencoders.values():
        m.eval()
    models = tr.hand_models
    os.makedirs(f"{a.out}/{a.task}", exist_ok=True)
    if not os.path.exists(f"{a.out}/descriptors.npz"):
        np.savez(f"{a.out}/descriptors.npz", **{h: descriptor(models[h]) for h in HANDS})
    X, Q, hand, ep = task_data(a.task, models, a.wrist_dir, a.sub)
    hi, ti = HANDS.index(a.hand), HANDS.index(a.target)
    keep = hand != ti                                                    # the target's demos are never used
    X, hand, ep = X[keep], hand[keep], ep[keep]
    off, cur = {}, 0
    for h in Q:
        if h != ti:
            off[h] = cur
            cur += len(Q[h])
    qry = np.flatnonzero(hand == hi)
    oth = torch.from_numpy(hand != hi)
    Xt = torch.from_numpy(X)
    nbs, dists = [], []
    for s in range(0, len(qry), 2048):
        D = torch.cdist(Xt[qry[s:s + 2048]], Xt).masked_fill(~oth[None], float("inf"))
        d, n = D.topk(a.k, largest=False)
        nbs.append(n.numpy()); dists.append(d.mean(1).numpy())
    nb, dist = np.concatenate(nbs), np.concatenate(dists)
    lat = {}
    for o in np.unique(hand[nb]):
        rows = np.array(sorted(set(int(r) for r in nb.reshape(-1) if hand[r] == o)))
        zs = []
        for b in range(0, len(rows), 4096):
            q, _, _ = retarget(models[HANDS[o]], models[a.hand], Q[o][rows[b:b + 4096] - off[o]].to(dev), a.iters, 1,
                               0.05, 1e-6, DP)
            with torch.no_grad():
                zs.append(enc(tr, a.hand, q.detach().cpu().clamp(-1, 1)).numpy())
        for r, z in zip(rows, np.concatenate(zs)):
            lat[r] = z
        print(a.hand, "<-", HANDS[o], len(rows), "retargeted", flush=True)
    k_ref = np.stack([np.stack([lat[int(r)] for r in row]).mean(0) for row in nb])
    with torch.no_grad():
        k_self = enc(tr, a.hand, Q[hi]).numpy()
    np.savez_compressed(f"{a.out}/{a.task}/{a.hand}.npz", x=X[qry], k_ref=k_ref.astype(np.float32),
                        k_self=k_self.astype(np.float32), delta=(k_self - k_ref).astype(np.float32), ep=ep[qry],
                        knn_dist=dist.astype(np.float32))
    print(a.hand, "done", len(qry), "frames, |delta| mean", float(np.abs(k_self - k_ref).mean()), flush=True)


if __name__ == "__main__":
    main()
