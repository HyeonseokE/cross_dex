"""Composed residual module f(z, hand, task context) for compositional generalization (RSS2027, 2026-11-23; dexlatent venv).

Setting: 4 tasks x 6 hands; one (GraspKettle, h) cell is held out (h = each hand in turn, 6 folds). The self residual
r = q* - D_h(E_h(q*)) of the held-out cell must be predicted from the other cells only:
  [1] other hands x same task  -> what the task needs (hand-agnostic context c / task id)
  [2] same hand x other tasks  -> how this hand is used (hand id)
Variants (all predict r in normalized joints, zero-padded to 22, loss masked to the hand's dof):
  V0 per_hand    f_h(z)                    same hand, other tasks only (= residual_fix --train_tasks, step 4)
  V1 hand        f(z, hand)                all other cells, no task information
  V2 hand_task   f(z, hand, task one-hot)  + task identity (learned from the other hands on GraspKettle)
  V3 hand_ctx    f(z, hand, c)             + hand-agnostic object context c (no task id)
  V4 all         f(z, hand, task, c)
Factorized (separate hand / task modules, composed bilinearly; --kinds fact_task,fact_ctx,fact_both):
  r = a(z, hand) + W(z, hand) e          hand module a, W (22 x K) sees only z and the hand id
  e = emb[task] (fact_task) | g(c) (fact_ctx) | both summed (fact_both)   task module sees only task / c
  For the held-out (GraspKettle, h): a, W of h come from h's other tasks, e of GraspKettle from the other hands.
c per step (privileged sim state from the teleop pickle, available in simulation): object position relative to the
wrist translation (world, 3), object linear velocity (3), object height above its start (1), episode phase t/T (1).
Metric: fingertip error (mm, canonical frame) of D_h(z) + r_hat vs q* on the held-out cell (all 50 episodes).

  cd /workspace/codes/dexverse/DexLatent && PYTHONPATH=/workspace/codes/cross_dex python -m \
     cross_dex.ours.crosshand_dp.cond_fix --latent_ckpt <space model.pt> --cache <npz> --json <out.json>
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch

from cross_dex.ours.diag.decoder_fix import encdec, own_pkl, tip_mm

TASKS = ["Dexverse-GraspKettle-v0", "Dexverse-GraspBleach-v0", "Dexverse-GraspPan-v0", "Dexverse-FunctionalHammerStrike-v0"]
HANDS = ["shadow", "inspire", "sharpa", "wuji", "allegro", "leap"]
PAD = 22


def quat_rot(q):   # (w, x, y, z) -> 3x3
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def build_cache(tr, path):
    from dexverse_xlvla.replay import action_names
    out = {}
    for t in TASKS:
        for h in HANDS:
            M = tr.hand_models[h]; ia = M.data_index(action_names(t, h))
            d = pickle.load(open(own_pkl(t, h), "rb"))
            Q, C = [], []
            for e in d["episodes"]:
                a = np.asarray(e["actions"], np.float64); T = len(a)
                q = M.angles_to_normalized(torch.tensor(a[:, ia], dtype=torch.float32)).clamp(-1, 1).numpy()
                st = e["states"][:T]
                rp = np.stack([s["articulation"]["robot"]["root_pose"][0] for s in st])
                jp = np.stack([s["articulation"]["robot"]["joint_position"][0, :3] for s in st])
                op = np.stack([s["rigid_object"]["object"]["root_pose"][0, :3] for s in st])
                ov = np.stack([s["rigid_object"]["object"]["root_velocity"][0, :3] for s in st])
                wrist = rp[:, :3] + np.einsum("ij,tj->ti", quat_rot(rp[0, 3:]), jp)
                c = np.concatenate([op - wrist, ov, (op[:, 2] - op[0, 2])[:, None], (np.arange(T) / T)[:, None]], 1)
                Q.append(q); C.append(c.astype(np.float32))
            out[f"{t}|{h}|q"] = np.concatenate(Q); out[f"{t}|{h}|c"] = np.concatenate(C)
            print("cache", t, h, out[f"{t}|{h}|q"].shape, flush=True)
    np.savez(path, **out)


def mlp(n_in):
    net = torch.nn.Sequential(torch.nn.Linear(n_in, 512), torch.nn.SiLU(), torch.nn.Linear(512, 512), torch.nn.SiLU(),
                              torch.nn.Linear(512, PAD))
    torch.nn.init.zeros_(net[-1].weight); torch.nn.init.zeros_(net[-1].bias)
    return net


class Fact(torch.nn.Module):
    def __init__(self, dz, mode, k=8):
        super().__init__()
        self.dz, self.mode, self.k = dz, mode, k
        def head(n_out):
            net = torch.nn.Sequential(torch.nn.Linear(dz + 6, 512), torch.nn.SiLU(), torch.nn.Linear(512, 512),
                                      torch.nn.SiLU(), torch.nn.Linear(512, n_out))
            torch.nn.init.zeros_(net[-1].weight); torch.nn.init.zeros_(net[-1].bias)
            return net
        self.a, self.W = head(PAD), head(PAD * k)
        if mode in ("fact_task", "fact_both"):
            self.emb = torch.nn.Parameter(torch.randn(4, k) * 0.5)
        if mode in ("fact_ctx", "fact_both"):
            self.g = torch.nn.Sequential(torch.nn.Linear(8, 128), torch.nn.SiLU(), torch.nn.Linear(128, k))

    def forward(self, X):
        zh, t, c = X[:, :self.dz + 6], X[:, self.dz + 6:self.dz + 10], X[:, self.dz + 10:]
        e = 0
        if self.mode in ("fact_task", "fact_both"):
            e = e + t @ self.emb
        if self.mode in ("fact_ctx", "fact_both"):
            e = e + self.g(c)
        return self.a(zh) + (self.W(zh).view(-1, PAD, self.k) @ e.unsqueeze(-1)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent_ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--steps", type=int, default=10000)
    ap.add_argument("--folds", default=",".join(HANDS))
    ap.add_argument("--kinds", default="per_hand,hand,hand_task,hand_ctx,all")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    from dexverse_xlvla.eval_latent import load
    tr = load(a.latent_ckpt)
    for m in tr.autoencoders.values():
        m.eval()
    if not os.path.exists(a.cache):
        build_cache(tr, a.cache)
    K = np.load(a.cache)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cells = {}
    for ti, t in enumerate(TASKS):
        for hi, h in enumerate(HANDS):
            q = torch.tensor(K[f"{t}|{h}|q"]); z, s = encdec(tr, h, q); n = q.shape[1]
            r = torch.zeros(len(q), PAD); r[:, :n] = q - s
            mask = torch.zeros(len(q), PAD); mask[:, :n] = 1
            cells[(t, h)] = {"q": q, "z": z, "s": s, "r": r, "m": mask, "c": torch.tensor(K[f"{t}|{h}|c"]),
                             "h": torch.nn.functional.one_hot(torch.tensor(hi), 6).float().expand(len(q), 6),
                             "t": torch.nn.functional.one_hot(torch.tensor(ti), 4).float().expand(len(q), 4)}
    allc = torch.cat([v["c"] for v in cells.values()]); cmu, csd = allc.mean(0), allc.std(0) + 1e-6
    for v in cells.values():
        v["c"] = (v["c"] - cmu) / csd

    def inputs(v, kind):
        if kind.startswith("fact"):
            return torch.cat([v["z"], v["h"], v["t"], v["c"]], 1)
        parts = [v["z"]]
        if kind != "per_hand":
            parts.append(v["h"])
        if kind in ("hand_task", "all"):
            parts.append(v["t"])
        if kind in ("hand_ctx", "all"):
            parts.append(v["c"])
        return torch.cat(parts, 1)

    res = {}
    for h in a.folds.split(","):
        held = (TASKS[0], h); M = tr.hand_models[h]; v = cells[held]
        res[h] = {"before": float(tip_mm(M, v["q"], v["s"]).mean())}
        for kind in a.kinds.split(","):
            train = [k for k in cells if k != held and (kind != "per_hand" or k[1] == h)]
            X = torch.cat([inputs(cells[k], kind) for k in train]).to(dev)
            R = torch.cat([cells[k]["r"] for k in train]).to(dev); Mk = torch.cat([cells[k]["m"] for k in train]).to(dev)
            torch.manual_seed(42)
            net = (Fact(v["z"].shape[1], kind) if kind.startswith("fact") else mlp(X.shape[1])).to(dev)
            opt = torch.optim.Adam(net.parameters(), 1e-3); sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
            for _ in range(a.steps):
                b = torch.randint(0, len(X), (4096,), device=dev)
                opt.zero_grad(); loss = (((net(X[b]) - R[b]) ** 2) * Mk[b]).sum() / Mk[b].sum(); loss.backward(); opt.step(); sch.step()
            net = net.cpu().eval()
            with torch.no_grad():
                n = v["q"].shape[1]; pred = net(inputs(v, kind))[:, :n]
                res[h][kind] = float(tip_mm(M, v["q"], (v["s"] + pred).clamp(-1, 1)).mean())
        print(h, {k: round(x, 2) for k, x in res[h].items()}, flush=True)
    mean = {k: float(np.mean([res[h][k] for h in res])) for k in next(iter(res.values()))}
    print("mean", {k: round(x, 2) for k, x in mean.items()}, flush=True)
    if a.json:
        json.dump({"folds": res, "mean": mean, "latent_ckpt": a.latent_ckpt, "steps": a.steps}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
