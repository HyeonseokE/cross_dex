"""Do hands share latent regions or sit in separate clusters? (dexlatent venv, CPU)

Demo finger actions of every hand (joint pi0.5 set) -> z = E_h(q*). Per task and over all tasks:
  probe   linear classifier z -> hand, trained / tested on different episodes (accuracy; chance = 1 / #hands)
  nn      for each z: distance to the nearest z of the same hand from another episode vs to the nearest z of any other
          hand (same task); share of frames whose nearest neighbour (other episodes) is another hand
  spread  distance between hand centroids vs the mean distance of a hand's z to its own centroid

  python -m cross_dex.ours.diag.hand_clusters --root <xlvla_joint set>/all --latent_ckpt <vae .pt> [--json out]
"""
import argparse, glob, json

import numpy as np
import pandas as pd
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--latent_ckpt", required=True)
    ap.add_argument("--per_cell", type=int, default=1500)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    from dexverse_xlvla.eval_latent import load, enc
    tr = load(a.latent_ckpt)
    for m in tr.autoencoders.values():
        m.eval()
    cells = json.load(open(f"{a.root}/meta/xlvla_cells.json"))["cells"]
    df = pd.concat([pd.read_parquet(f, columns=["episode_index", "action"]) for f in sorted(glob.glob(f"{a.root}/data/*/*.parquet"))])
    act = np.stack(df["action"].to_numpy()).astype(np.float32); ep = df["episode_index"].to_numpy()
    per = len(np.unique(ep)) // len(cells); rng = np.random.default_rng(0)
    hands = list(dict.fromkeys(c[1] for c in cells)); tasks = list(dict.fromkeys(c[0] for c in cells))
    Z, H, T, E = [], [], [], []
    for c, (t, h) in enumerate(cells):
        i = np.where(ep // per == c)[0]; i = rng.choice(i, min(a.per_cell, len(i)), replace=False)
        n = tr.hand_models[h].dof_count()
        with torch.no_grad():
            Z.append(enc(tr, h, torch.tensor(act[i, 9:9 + n])).numpy())
        H += [hands.index(h)] * len(i); T += [t] * len(i); E.append(ep[i])
    Z, H, T, E = np.concatenate(Z), np.array(H), np.array(T), np.concatenate(E)

    def probe(m):
        te = (E % 5 == 0) & m; trn = (E % 5 != 0) & m
        X = torch.tensor(Z); y = torch.tensor(H)
        mu, sd = X[trn].mean(0), X[trn].std(0) + 1e-6; X = (X - mu) / sd
        lin = torch.nn.Linear(32, len(hands)); opt = torch.optim.Adam(lin.parameters(), 1e-2)
        idx = torch.tensor(np.where(trn)[0])
        for _ in range(1500):
            opt.zero_grad(); torch.nn.functional.cross_entropy(lin(X[idx]), y[idx]).backward(); opt.step()
        with torch.no_grad():
            return float((lin(X[torch.tensor(np.where(te)[0])]).argmax(1) == y[torch.tensor(np.where(te)[0])]).float().mean())

    def nn_stats(m):
        X = torch.tensor(Z[m]); h, e = H[m], E[m]
        D = torch.cdist(X, X)
        D[torch.tensor(e[:, None] == e[None, :])] = float("inf")              # other episodes only
        same = D.clone(); same[torch.tensor(h[:, None] != h[None, :])] = float("inf")
        other = D.clone(); other[torch.tensor(h[:, None] == h[None, :])] = float("inf")
        ds, do = same.min(1).values, other.min(1).values
        cen = np.stack([Z[m][h == k].mean(0) for k in np.unique(h)])
        within = np.mean([np.linalg.norm(Z[m][h == k] - cen[i], axis=1).mean() for i, k in enumerate(np.unique(h))])
        cd = np.linalg.norm(cen[:, None] - cen[None], axis=-1); cd = cd[~np.eye(len(cen), dtype=bool)].mean()
        return {"nn_same_hand": float(ds.median()), "nn_other_hand": float(do.median()),
                "share_nn_is_other_hand_pct": float(100 * (do < ds).float().mean()),
                "centroid_dist": float(cd), "within_hand_spread": float(within)}
    out = {"chance_pct": 100 / len(hands), "all_tasks": {"probe_acc_pct": 100 * probe(np.ones(len(H), bool)), **nn_stats(np.ones(len(H), bool))}}
    for t in tasks:
        m = T == t
        out[t] = {"probe_acc_pct": 100 * probe(m), **nn_stats(m)}
    print(json.dumps(out, indent=1))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
