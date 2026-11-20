"""Hand-agnostic decoder correction g(z) (RSS2027 problem 2; dexlatent venv).

One network for all hands: g(z) -> fingertip offsets (5 x 3, thumb..little) in the hand-common canonical frame. A hand t
applies it through its own kinematics: q' = IK_t(FK_t(D_t(z)) + g(z)), IK by gradient from D_t(z) (allegro / LEAP use
the first 4 tips). Trained on task-agnostic poses only: for every hand h and pose q of the pair set (h's own draws +
retargets onto h), z = E_h(q), target = FK_h(q) - FK_h(D_h(z)). The shared space (E, D) stays fixed.

Zero-shot check: for each target hand t, g_{-t} is trained without hand t (leave one hand out) and evaluated on t;
g_all (all hands) is the seen-hand reference.
  self   t's teleop finger actions (4 tasks): fingertip error of D_t(E_t(q*)) before / after (mm)
  cross  z = E_s(q_s) of every other hand s decoded on t:
           pair  s's uniform draw and its retarget onto t: fingertip error to the retarget (mm), thumb-finger line error
                 (direction deg, length mm) to s's fingertips
           demo  s's teleop actions: line error to s's fingertips

  python -m cross_dex.ours.diag.agnostic_fix --latent_ckpt <pt> --json <out> [--pairs <npz>]
"""
import argparse
import json

import numpy as np
import torch

from cross_dex.ours.diag.decoder_fix import TASKS, demo_q, encdec

HANDS = "shadow,inspire,sharpa,wuji,allegro,leap"


def net():
    g = torch.nn.Sequential(torch.nn.Linear(32, 512), torch.nn.SiLU(), torch.nn.Linear(512, 512), torch.nn.SiLU(),
                            torch.nn.Linear(512, 15))
    torch.nn.init.zeros_(g[-1].weight); torch.nn.init.zeros_(g[-1].bias)
    return g


def train_g(Z, D, Mk, hands, steps, dev):
    """Z[h] (N,32), D[h] (N,5,3) target offsets (m), Mk[h] (5,) tip mask; trained on the given hands."""
    z = torch.cat([Z[h] for h in hands]).to(dev); d = torch.cat([D[h] for h in hands]).to(dev)
    m = torch.cat([Mk[h].expand(len(Z[h]), 5) for h in hands]).to(dev)
    g = net().to(dev); opt = torch.optim.Adam(g.parameters(), 1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for it in range(steps):
        b = torch.randint(0, len(z), (4096,), device=dev)
        err = ((g(z[b]).view(-1, 5, 3) - d[b]) ** 2).sum(-1) * m[b]
        loss = err.sum() / m[b].sum(); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return g.cpu().eval()


def apply(M, g, d0, z, iters=100, lam=1e-3):
    """q' = IK(FK(d0) + g(z)) by Adam from d0 (normalized joints, kept in [-1, 1])."""
    k = M.tip_count()
    with torch.no_grad():
        tgt = M.forward(d0) + g(z).view(-1, 5, 3)[:, :k]
    q = d0.clone().requires_grad_(True); opt = torch.optim.Adam([q], lr=0.02)
    for _ in range(iters):
        opt.zero_grad()
        loss = ((M.forward(q) - tgt) ** 2).sum((-1, -2)).sum() + lam * ((q - d0) ** 2).sum()
        loss.backward(); opt.step()
        with torch.no_grad():
            q.clamp_(-1, 1)
    return q.detach()


def tip_mm(M, a, b):
    with torch.no_grad():
        return float((M.forward(a) - M.forward(b)).norm(dim=-1).mean() * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_ckpt", required=True)
    ap.add_argument("--pairs", default="/workspace/datasets/dexverse/uniform_pairs/dexpilot_right6_60k_wristfix.npz")
    ap.add_argument("--hands", default=HANDS)
    ap.add_argument("--tasks", default=TASKS)
    ap.add_argument("--per_hand", type=int, default=120000)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--json", required=True)
    a = ap.parse_args()
    from dexverse_xlvla.eval_latent import dec, enc, line_err, load
    tr = load(a.latent_ckpt)
    for m in tr.autoencoders.values():
        m.eval()
    hands = a.hands.split(","); P = np.load(a.pairs); M = tr.hand_models
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); g0 = torch.Generator().manual_seed(42)
    torch.manual_seed(42)
    Z, D, Mk = {}, {}, {}
    for h in hands:                                     # training targets: task-agnostic poses of each hand
        q = torch.tensor(np.concatenate([P[f"{s}__{h}"] for s in hands]))
        q = q[torch.randperm(len(q), generator=g0)[: a.per_hand]]
        z, s = encdec(tr, h, q); k = M[h].tip_count()
        with torch.no_grad():
            off = M[h].forward(q) - M[h].forward(s.clamp(-1, 1))
        D[h] = torch.zeros(len(q), 5, 3); D[h][:, :k] = off; Z[h] = z
        Mk[h] = torch.tensor([1.0] * k + [0.0] * (5 - k))
        print(h, "train offset mm", round(float(off.norm(dim=-1).mean() * 1000), 2), flush=True)
    G = {"all": train_g(Z, D, Mk, hands, a.steps, dev)}
    for t in hands:
        G[f"-{t}"] = train_g(Z, D, Mk, [h for h in hands if h != t], a.steps, dev)
        print("trained g without", t, flush=True)
    # evaluation rows
    dq = {}
    for h in hands:
        x = torch.cat([torch.cat(list(demo_q(tr, t, h)[1].values())) for t in a.tasks.split(",")])
        dq[h] = x[torch.randperm(len(x), generator=g0)[: a.rows]]
    pidx = torch.randperm(60000, generator=g0)[: a.rows]
    out = {}
    for t in hands:
        r = {}
        z, s = encdec(tr, t, dq[t]); s = s.clamp(-1, 1)
        r["self_demo_tip_mm_before"] = tip_mm(M[t], dq[t], s)
        for gn in ["all", f"-{t}"]:
            r[f"self_demo_tip_mm_after_{'all' if gn == 'all' else 'loo'}"] = tip_mm(M[t], dq[t], apply(M[t], G[gn], s, z))
        acc = {}
        for src in hands:
            if src == t:
                continue
            k = min(M[src].tip_count(), M[t].tip_count()); pr = [(0, i) for i in range(1, k)]
            qs = torch.tensor(P[f"{src}__{src}"])[pidx]; qt = torch.tensor(P[f"{src}__{t}"])[pidx]
            for kind, x in [("pair", qs), ("demo", dq[src])]:
                with torch.no_grad():
                    zz = enc(tr, src, x); d0 = dec(tr, t, zz).clamp(-1, 1); ts = M[src].forward(x)
                for nm, qq in [("before", d0), ("after_all", apply(M[t], G["all"], d0, zz)),
                               ("after_loo", apply(M[t], G[f"-{t}"], d0, zz))]:
                    with torch.no_grad():
                        dd, ll = line_err(ts, M[t].forward(qq), pr)
                    acc.setdefault(f"{kind}_dir_deg_{nm}", []).append(dd)
                    acc.setdefault(f"{kind}_len_mm_{nm}", []).append(ll * 1e3)
                    if kind == "pair":
                        acc.setdefault(f"pair_tip_mm_{nm}", []).append(tip_mm(M[t], qt, qq))
        r.update({k: float(np.mean(v)) for k, v in acc.items()})
        out[t] = r
        print(t, json.dumps({k: round(v, 2) for k, v in r.items()}), flush=True)
    mean = {k: float(np.mean([out[t][k] for t in hands])) for k in out[hands[0]]}
    print("mean", json.dumps({k: round(v, 2) for k, v in mean.items()}), flush=True)
    json.dump({"latent_ckpt": a.latent_ckpt, "per_target": out, "mean": mean}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
