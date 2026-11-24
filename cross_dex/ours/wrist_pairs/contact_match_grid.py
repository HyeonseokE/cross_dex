"""Cross-hand contact matching at dataset scale: batched GPU solver with per-object distance grids (RSS2027, 2026-11-23).

Same objective and initialisation as contact_match.py / contact_match_gpu.py. The penetration term no longer evaluates all
face planes of the object for every finger point (the bottleneck: ~90 points x ~4000 planes x restarts x iterations per
grasp and hand); instead each object gets one distance grid (RES^3, object at unit scale, plane distance of the CoACD convex
pieces: exact inside a piece, a lower bound outside) that is looked up trilinearly and multiplied by the grasp's scale.
Grids are cached in --sdf. No physics: a touching, non-penetrating pose EXISTS; it is not checked that the grasp holds.

  python contact_match_grid.py --grasps 'grasps_*.pkl' | --data 'shard_*.npz'  --urdf <dir> --sdf <dir> --out pairs.npz
Input npz (dexo_extract_full.py) / output npz: per grasp type, src (grasp file), k, scale, obj (pieces folder), palm_T, shadow_q, knuckles,
tips, pc/pm (closest distal contact per finger + mask), sc/sm (<= SEC other contacts per finger); output adds per hand
q_<h>, T_<h> (root link in the object frame), tip_<h> (worst targeted fingertip error, mm), pen_<h> (mm), ok_<h>.
"""
import argparse
import glob
import json
import os
import pickle
import time

import numpy as np
import torch
from scipy.spatial import ConvexHull

from contact_match_gpu import SPEC, expmap, kabsch, seg_dist
from contact_match_grid_pack import contacts_arrays
from urdf_fk import Hand

SEC, RES, FT = 8, 80, torch.float32
HANDS = {}


def pack(G):
    n = len(G); D = {"type": np.array([g["type"] for g in G]), "src": np.array([g["file"] for g in G]), "k": np.array([g["k"] for g in G]),
                     "scale": np.array([g["scale"][0] for g in G], np.float32), "obj": np.array([os.path.normpath(g["pieces"]) for g in G]),
                     "palm_T": np.stack([g["palm_T"] for g in G]).astype(np.float32), "shadow_q": np.stack([g["shadow_q"] for g in G]).astype(np.float32),
                     "knuckles": np.stack([g["knuckles"] for g in G]).astype(np.float32), "tips": np.stack([g["tips"] for g in G]).astype(np.float32),
                     "pc": np.zeros((n, 5, 3), np.float32), "pm": np.zeros((n, 5), np.float32), "sc": np.zeros((n, 5, SEC, 3), np.float32),
                     "sm": np.zeros((n, 5, SEC), np.float32)}
    for i, g in enumerate(G):
        D["pc"][i], D["pm"][i], D["sc"][i], D["sm"][i] = contacts_arrays(g["contacts"])
    return D


def build_grid(obj, dev):    # -> grid [RES,RES,RES] (z,y,x) float16, lo [3], hi [3]   (unit scale)
    V, E = [], []
    for f in sorted(glob.glob(os.path.join(obj, "*.obj"))):
        v = np.array([l.split()[1:4] for l in open(f) if l.startswith("v ")], np.float64)
        try:
            E.append(ConvexHull(v).equations.astype(np.float32)); V.append(v)
        except Exception:   # noqa: BLE001
            continue
    if not E:
        return None
    V = np.concatenate(V); lo, hi = V.min(0), V.max(0); m = 0.3 * (hi - lo).max(); lo, hi = (lo - m).astype(np.float32), (hi + m).astype(np.float32)
    ax = [torch.linspace(float(lo[i]), float(hi[i]), RES, device=dev) for i in range(3)]
    zz, yy, xx = torch.meshgrid(ax[2], ax[1], ax[0], indexing="ij"); pts = torch.stack([xx, yy, zz], -1).reshape(-1, 3)
    sdf = torch.full((len(pts),), 1e3, device=dev)
    for e in E:
        e = torch.from_numpy(e).to(dev); st = max(1, int(1e8 // len(e)))
        for i in range(0, len(pts), st):
            sdf[i:i + st] = torch.minimum(sdf[i:i + st], (pts[i:i + st] @ e[:, :3].T + e[:, 3]).max(1).values)
    return sdf.view(RES, RES, RES).half().cpu().numpy(), lo, hi


def grids(objs, sdf_dir, dev, verbose=True):
    os.makedirs(sdf_dir, exist_ok=True); out = {}; t0 = time.time()
    for i, o in enumerate(objs):
        f = os.path.join(sdf_dir, o.split("/processed_data/")[-1].split("/")[0] + ".npz")
        if os.path.exists(f):
            d = np.load(f); out[o] = (d["g"], d["lo"], d["hi"])
        else:
            r = build_grid(o, dev)
            if r is not None:
                np.savez(f + ".tmp.npz", g=r[0], lo=r[1], hi=r[2]); os.replace(f + ".tmp.npz", f); out[o] = r
        if verbose and i % 500 == 0:
            print(f"grid {i}/{len(objs)} {time.time() - t0:.0f}s", flush=True)
    return out


def solve(H, r, D, idx, GR, glo, ghi, iters, rst, dev, prune=0):
    """prune > 0: after `prune` iterations only the best restart of every grasp is optimised further."""
    nf = len(H.chains); ds = H.distal_start(); n = len(idx)
    lo, up = H.lower.to(dev, FT), H.upper.to(dev, FT)
    fr = torch.tensor([0.3, 0.5, 0.1, 0.7], device=dev, dtype=FT)[:rst, None]; q0 = lo + (up - lo) * fr
    with torch.no_grad():
        kn = np.stack([c[0, 0].cpu().numpy() for c in H.chain_points(q0[:1])])
    T0_ = torch.as_tensor(np.stack([kabsch(kn, k[:nf]) for k in D["knuckles"][idx]]).astype(np.float32), device=dev)
    cen_ = torch.as_tensor(D["knuckles"][idx, :nf].mean(1), device=dev)
    base = [torch.as_tensor(D[k][idx, :nf], device=dev) for k in ("pc", "pm", "sc", "sm")]
    # grasps grouped by object: one volume per object, its grasps' points padded to the largest group
    ug, inv_, cnt = np.unique(D["gi"][idx], return_inverse=True, return_counts=True); U, c = len(ug), int(cnt.max())
    o = np.argsort(inv_, kind="stable"); pos = np.empty(n, np.int64); pos[o] = np.arange(n) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    slot = np.full((U, c), n, np.int64); slot[inv_, pos] = np.arange(n)
    slot, inv_t, pos_t = (torch.as_tensor(x, device=dev) for x in (slot, inv_, pos))
    vol = GR[torch.as_tensor(ug, device=dev)].float()[:, None]                                        # [U,1,R,R,R]
    s = torch.as_tensor(D["scale"][idx], device=dev)[:, None, None]; blo, bhi = glo[ug][:, None], ghi[ug][:, None]
    lin = torch.linspace(0, 1, 5, device=dev)[None, :, None]

    def stage(k, u, dt, dr, n_it):          # k restarts per grasp (grasp-major), parameters [n k, ...]
        B = n * k; pc, pm, sc, sm = (x.repeat_interleave(k, 0) for x in base); T0, cen = T0_.repeat_interleave(k, 0), cen_.repeat_interleave(k, 0)
        u, dt, dr = (x.clone().requires_grad_(True) for x in (u, dt, dr))
        opt = torch.optim.Adam([{"params": [u], "lr": 0.05}, {"params": [dt], "lr": 0.002}, {"params": [dr], "lr": 0.02}])

        def terms(q):
            E = expmap(dr); R = E @ T0[:, :3, :3]; t = (E @ (T0[:, :3, 3] - cen)[:, :, None]).squeeze(-1) + cen + dt
            ch = [torch.einsum("bij,bnj->bni", R, x) + t[:, None] for x in H.chain_points(q)]
            e_prim = torch.stack([seg_dist(pc[:, f], ch[f][:, ds[f]], ch[f][:, -1]) - r for f in range(nf)], 1)
            e_sec = torch.stack([torch.stack([seg_dist(sc[:, f], ch[f][:, i, None], ch[f][:, i + 1, None])
                                              for i in range(ch[f].shape[1] - 1)], 0).min(0).values - r for f in range(nf)], 1)
            pts = torch.cat([torch.cat([x[:, i, None] + (x[:, i + 1, None] - x[:, i, None]) * lin for i in range(x.shape[1] - 1)], 1) for x in ch], 1)
            pn = pts.reshape(n, -1, 3) / s; M = pn.shape[1]                                             # unit-scale object frame
            pu = torch.cat([pn, pn.new_zeros(1, M, 3)])[slot].reshape(U, c * M, 3); pcl = torch.maximum(torch.minimum(pu, bhi), blo)
            g = ((pcl - blo) / (bhi - blo) * 2 - 1)[:, :, None, None]                                   # [U, c M, 1, 1, 3] (x, y, z)
            val = torch.nn.functional.grid_sample(vol, g, mode="bilinear", padding_mode="border", align_corners=True)[:, 0, :, 0, 0]
            sdf = ((val + (pu - pcl).norm(dim=-1)).view(U, c, M)[inv_t, pos_t] * s[:, :, 0]).reshape(B, -1)
            return e_prim, e_sec, torch.relu(r - sdf), R, t

        for _ in range(n_it):
            opt.zero_grad()
            e_prim, e_sec, pen, _, _ = terms(lo + torch.sigmoid(u) * (up - lo))
            loss = (((e_prim * 1e3) ** 2) * pm).sum(1) + 0.3 * ((((e_sec * 1e3) ** 2) * sm).sum(2) / sm.sum(2).clamp_min(1)).sum(1) \
                + 3.0 * ((pen * 1e3) ** 2).sum(1) + 1e-2 * ((dt * 1e3) ** 2).sum(1)
            loss.sum().backward(); opt.step()
        with torch.no_grad():
            q = lo + torch.sigmoid(u) * (up - lo); e_prim, _, pen, R, t = terms(q)
            tip = (e_prim.abs() * pm).max(1).values; penm = pen.max(1).values
            b = (tip + penm).view(n, k).argmin(1) + torch.arange(n, device=dev) * k
        return u.detach()[b], dt.detach()[b], dr.detach()[b], q[b], R[b], t[b], tip[b], penm[b]

    u0 = torch.logit(((q0 - lo) / (up - lo)).clamp(0.02, 0.98)).repeat(n, 1); z = torch.zeros(n * rst, 3, device=dev)
    if prune and rst > 1:
        u1, dt1, dr1 = stage(rst, u0, z, z, prune)[:3]; out = stage(1, u1, dt1, dr1, iters - prune)
    else:
        out = stage(rst, u0, z, z, iters)
    q, R, t, tip, penm = out[3:]
    T = torch.eye(4, device=dev).repeat(n, 1, 1); T[:, :3, :3] = R; T[:, :3, 3] = t
    return q.cpu().numpy(), T.cpu().numpy(), (tip * 1e3).cpu().numpy(), (penm * 1e3).cpu().numpy()


def match(D, a, dev, verbose=True):
    t0 = time.time(); objs = sorted(set(D["obj"].tolist())); GD = grids(objs, a.sdf, dev, verbose)
    keep = np.array([o in GD for o in D["obj"]]); D = {k: v[keep] for k, v in D.items()}
    objs = [o for o in objs if o in GD]; oi = {o: i for i, o in enumerate(objs)}; D["gi"] = np.array([oi[o] for o in D["obj"]])
    GR = torch.from_numpy(np.stack([GD[o][0] for o in objs])).to(dev)
    glo = torch.tensor(np.stack([GD[o][1] for o in objs]), device=dev); ghi = torch.tensor(np.stack([GD[o][2] for o in objs]), device=dev)
    n = len(D["k"]); print(f"grasps {n} objects {len(objs)} grids {time.time() - t0:.0f}s", flush=True)
    for h in a.hands.split(","):
        s = SPEC[h]
        if h not in HANDS:
            HANDS[h] = Hand(os.path.join(a.urdf, s[0]), s[1], s[2])
        H = HANDS[h]; nf = len(H.chains)
        q = np.zeros((n, len(H.names)), np.float32); T = np.zeros((n, 4, 4), np.float32); tip = np.zeros(n, np.float32); pen = np.zeros(n, np.float32)
        for i in range(0, n, a.chunk):
            idx = np.arange(i, min(i + a.chunk, n))
            q[idx], T[idx], tip[idx], pen[idx] = solve(H, s[3], D, idx, GR, glo, ghi, a.iters, a.restarts, dev, a.prune)
            if verbose:
                print(f"{h} {idx[-1] + 1}/{n} {time.time() - t0:.0f}s", flush=True)
        ok = (D["pm"][:, :nf].sum(1) > 0) & (tip < 5) & (pen < 2)
        D[f"q_{h}"], D[f"T_{h}"], D[f"tip_{h}"], D[f"pen_{h}"], D[f"ok_{h}"] = q, T, tip, pen, ok
        print(f"{h} yield {ok.mean():.3f}  tip p50 {np.median(tip):.2f} pen p50 {np.median(pen):.2f}  {time.time() - t0:.0f}s", flush=True)
    allok = np.all([D[f"ok_{h}"] for h in a.hands.split(",")], 0)
    print(f"all-hands yield {allok.mean():.3f} ({int(allok.sum())} scenes)  total {time.time() - t0:.0f}s", flush=True)
    D["names"] = np.array([json.dumps({h: HANDS[h].names for h in a.hands.split(",")})]); del D["gi"]
    return D


def load(f):
    D = dict(np.load(f))
    for k in ("type", "src", "obj"):
        D[k] = D[k].astype(str)
    return D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasps", default=""); ap.add_argument("--data", default="")
    ap.add_argument("--urdf", required=True); ap.add_argument("--sdf", required=True)
    ap.add_argument("--iters", type=int, default=200); ap.add_argument("--restarts", type=int, default=4); ap.add_argument("--prune", type=int, default=0)
    ap.add_argument("--hands", default=",".join(SPEC)); ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--out", required=True, help="npz, or (with --data) a folder: one output per input shard, done shards skipped")
    ap.add_argument("--until", default="", help="with --data: keep polling for new shards until this file exists")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.grasps:
        np.savez(a.out, **match(pack([g for f in sorted(glob.glob(a.grasps)) for g in pickle.load(open(f, "rb"))]), a, dev)); return
    os.makedirs(a.out, exist_ok=True)
    while True:
        last = bool(a.until) and os.path.exists(a.until)
        todo = [f for f in sorted(glob.glob(a.data)) if not f.endswith(".tmp.npz") and not os.path.exists(os.path.join(a.out, os.path.basename(f)))]
        for f in todo:
            print("shard", f, flush=True); tmp = os.path.join(a.out, os.path.basename(f)[:-4] + ".tmp.npz")
            np.savez(tmp, **match(load(f), a, dev, verbose=False)); os.replace(tmp, os.path.join(a.out, os.path.basename(f)))
        if not todo:
            if last or not a.until:
                break
            time.sleep(60)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
