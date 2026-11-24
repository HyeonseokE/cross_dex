"""Does f need to see the object?  f(hand geometry) vs f(hand geometry + object points) on Dexonomy pairs (RSS2027, 2026-11-23).

Step 1 of the object-input study: IDEAL object points (uniform samples of the object's mesh surface, no occlusion, no noise).
If this does not help, a wrist-camera point cloud cannot help either. Same data, hands, loss, optimiser and noise as
train_f_cmp.py (the colleague's pd2 code); the only change is the model input:
  hand      pd2.train_f.F unchanged
  hand+pc   F plus a PointNet branch (per-point MLP 3-64-128-128, max pool) on N object surface points expressed in the
            hand's canonical palm frame; the N points nearest to the centroid of the hand's finger pads are used.
The palm frame of the input is the hand's raw wrist (known when a dataset is converted; at run time it has to be iterated).
Splits: random 80/20 scenes, and whole grasp types held out (every third Feix type) to test unseen kinds of grasps.

  python train_f_pc.py --xwo <repo> --dexo 'pairs*.npz' --urdf <dir> --out <dir>
"""
import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

from train_f_cmp import load_dexo, load_pd2


def surface_points(obj_dir, n, rng):     # uniform samples of <object>/mesh/simplified.obj (unit scale)
    f = os.path.join(os.path.dirname(os.path.dirname(obj_dir)), "mesh", "simplified.obj"); V, F = [], []
    for l in open(f):
        if l.startswith("v "):
            V.append(l.split()[1:4])
        elif l.startswith("f "):
            F.append([x.split("/")[0] for x in l.split()[1:4]])
    V, F = np.array(V, np.float64), np.array(F, np.int64) - 1; a, b, c = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1); k = rng.choice(len(F), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n); m = u + v > 1; u[m], v[m] = 1 - u[m], 1 - v[m]
    return (a[k] + u[:, None] * (b[k] - a[k]) + v[:, None] * (c[k] - a[k])).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xwo", required=True); ap.add_argument("--dexo", required=True); ap.add_argument("--urdf", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--points", type=int, default=128); ap.add_argument("--variants", type=int, default=16); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    a.xwo = os.path.abspath(a.xwo); os.makedirs(a.out, exist_ok=True); logf = open(f"{a.out}/train_log.txt", "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    C, tf = load_pd2(a.xwo); dev = a.device; hands = [C.Hand(h, dev) for h in C.HANDS]
    D = load_dexo(a.dexo, hands, C, a.urdf, dev, a.seed, 0); D.cache(tf.SIGMA, a.variants)
    rng = np.random.default_rng(a.seed); cache = {}; P = np.zeros((D.S, 1024, 3), np.float32); ok = np.ones(D.S, bool)
    for i, (o, s) in enumerate(zip(D.obj, D.scale)):
        if o not in cache:
            try:
                cache[o] = surface_points(o, 1024, rng)
            except Exception as e:   # noqa: BLE001
                cache[o] = None; log(f"no mesh {o} {str(e)[:60]}")
        if cache[o] is None:
            ok[i] = False
        else:
            P[i] = cache[o] * s
    P = torch.tensor(P, device=dev); PC = torch.zeros(D.S, 5, a.points, 3, device=dev)
    with torch.no_grad():
        for i in range(5):
            Wi = tf.inv(D.W[:, i]); p = torch.einsum("sij,snj->sni", Wi[:, :3, :3], P) + Wi[:, None, :3, 3]     # palm frame
            fing, mask, _ = D.fc[i][0]; cen = (fing[:, :, :3] * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
            k = (p - cen[:, None]).norm(dim=-1).topk(a.points, largest=False).indices
            PC[:, i] = torch.gather(p, 1, k[..., None].expand(-1, -1, 3))
    log(f"scenes {D.S} (with mesh {int(ok.sum())})  points per hand {a.points}  mean point distance to the pad centroid "
        f"{float((PC - 0).norm(dim=-1).mean() * 1e3):.0f} mm from the palm origin")

    class FPC(tf.F):
        def __init__(self, h=128):
            super().__init__(h)
            self.pe = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 128), nn.SiLU(), nn.Linear(128, 128))
            self.head = nn.Sequential(nn.Linear(2 * h + 6 + 128, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 9))
            nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

        def forward(self, fing, mask, palm, pc):
            B = fing.shape[0]; oh = torch.eye(5, device=fing.device)[None].expand(B, 5, 5)
            e = self.enc(torch.cat([fing * self.fscale, oh], -1)); m = mask[..., None]
            mean = (e * m).sum(1) / m.sum(1).clamp_min(1); mx = e.masked_fill(m == 0, -1e4).max(1).values
            o = self.head(torch.cat([mean, mx, palm * self.pscale, self.pe(pc * 10).max(1).values], -1))
            return tf.make_T(tf.rot6d_to_R(o[:, 3:] + self.r6), o[:, :3] * 0.1)

    def offs(f, kind, idx, sigma):
        return torch.stack([f(*D.feats(i, idx, sigma), PC[idx, i]) if kind == "hand+pc" else f(*D.feats(i, idx, sigma)) for i in range(5)], 1)

    def train(kind, tr):
        torch.manual_seed(a.seed); f = (FPC() if kind == "hand+pc" else tf.F()).to(dev)
        opt = torch.optim.AdamW(f.parameters(), lr=1e-3, weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
        idx_t = torch.as_tensor(tr, device=dev)
        for s in range(a.steps):        # = pd2.train_f.train
            sb = idx_t[torch.randint(len(idx_t), (128,), device=dev)]; T = offs(f, kind, sb, tf.SIGMA); E = D.W[sb] @ tf.inv(T); pair = 0
            for x in range(5):
                for y in range(x + 1, 5):
                    pair = pair + (E[:, x, :3, 3] - E[:, y, :3, 3]).norm(dim=-1) + tf.W_ROT * (E[:, x, :3, :3] - E[:, y, :3, :3]).flatten(1).norm(dim=-1) / math.sqrt(2)
            pair = pair.mean() / 10
            zero = T[..., :3, 3].mean(1).norm(dim=-1).mean() + tf.W_ROT * (T[..., :3, :3].mean(1) - torch.eye(3, device=dev)).flatten(1).norm(dim=-1).mean() / math.sqrt(2)
            loss = pair + tf.LAMBDA * zero; opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        return f.eval()

    @torch.no_grad()
    def evaluate(models, tr, te):
        te_t = torch.as_tensor(te, device=dev); W = D.W[te_t]; E = {"raw": W}
        Wtr = D.W[torch.as_tensor(tr, device=dev)]; mean_tr = tf.se3_mean(Wtr, 1); Cc = torch.stack([tf.se3_mean(tf.inv(mean_tr) @ Wtr[:, i], 0) for i in range(5)])
        E["per_hand_const"] = W @ tf.inv(Cc)[None]
        for kind, f in models.items():
            E[kind] = W @ tf.inv(offs(f, kind, te_t, 0.0))
        out = {}
        for k, v in E.items():
            p, r = tf.dispersion(v); out[k] = {"pos_mm": float(p.mean()), "rot_deg": float(r.mean())}
        return out

    tid = np.array([int(t.split("_")[0]) for t in D.tag]); types = sorted(set(tid.tolist())); held = types[::3]
    splits = {"random": (D.tr[ok[D.tr]], D.te[ok[D.te]]), "types_held_out": (np.nonzero(ok & ~np.isin(tid, held))[0], np.nonzero(ok & np.isin(tid, held))[0])}
    R = {"config": {"steps": a.steps, "points": a.points, "held_types": held, "scenes": D.S, "hands": list(C.HANDS)}, "splits": {}}
    for name, (tr, te) in splits.items():
        models = {k: train(k, tr) for k in ("hand", "hand+pc")}
        for k, f in models.items():
            torch.save(f.state_dict(), f"{a.out}/f_{name}_{k.replace('+', '_')}.pt")
        R["splits"][name] = {"n_train": int(len(tr)), "n_test": int(len(te)), "test": evaluate(models, tr, te), "train": evaluate(models, tr, tr[:2000])}
        log(f"[{name}] train {len(tr)} test {len(te)}  " + "  ".join(f"{k} {v['pos_mm']:.1f}mm/{v['rot_deg']:.1f}deg" for k, v in R["splits"][name]["test"].items())
            + "   (train: " + "  ".join(f"{k} {v['pos_mm']:.1f}" for k, v in R["splits"][name]["train"].items()) + ")")
        json.dump(R, open(f"{a.out}/metrics.json", "w"), indent=1)
    log("DONE")


if __name__ == "__main__":
    main()
