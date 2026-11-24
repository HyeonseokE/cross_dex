"""Batched (GPU) cross-hand contact matching on Dexonomy grasps (RSS2027 wrist-pair data, 2026-11-23).

Same objective as contact_match.py (pilot, one grasp at a time on CPU), solved for many grasps at once:
  primary    distal segment of finger f touches Shadow's distal contact point of f
  secondary  some segment of finger f touches Shadow's proximal / middle contact points of f (weight 0.3, <= SEC per finger)
  penetration no finger sphere inside the object (object = union of CoACD convex pieces, face planes)
Differences from the pilot: float32 on GPU; the reported penetration uses the plane distance (exact inside a piece, a
lower bound of the distance outside -> penetration is over-estimated, i.e. conservative) instead of trimesh.
No physics: a touching, non-penetrating pose EXISTS; it is not checked that the grasp holds.

  python contact_match_gpu.py --grasps 'grasps_*.pkl' --urdf <dir> --out pairs.pkl     (torch + scipy, no trimesh)
"""
import argparse
import glob
import os
import pickle
import time
from multiprocessing import Pool

import numpy as np
import torch
from scipy.spatial import ConvexHull

from urdf_fk import Hand

SPEC = {   # urdf (under --urdf), root link, tips thumb..little, sphere radius (m)   (= contact_match.py)
    "inspire": ("inspire/retarget/inspire_hand_right/inspire_hand_right.urdf", "base",
                ("thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"), 0.008),
    "sharpa": ("sharpa/retarget/sharpa_wave_right/sharpa_wave_right.urdf", "right_hand_C_MC",
               ("right_thumb_fingertip", "right_index_fingertip", "right_middle_fingertip", "right_ring_fingertip", "right_pinky_fingertip"), 0.009),
    "wuji": ("wuji/retarget/wuji_hand_right/wuji_hand_right.urdf", "right_palm_link",
             ("right_finger1_tip_link", "right_finger2_tip_link", "right_finger3_tip_link", "right_finger4_tip_link", "right_finger5_tip_link"), 0.008),
    "allegro": ("allegro/retarget/allegro_hand_right/allegro_hand_right.urdf", "base_link",
                ("link_15.0_tip", "link_3.0_tip", "link_7.0_tip", "link_11.0_tip"), 0.012),
    "leap": ("leap/retarget/leap_hand_right_fixed.urdf", "base_link",
             ("thumb_tip_head", "index_tip_head", "middle_tip_head", "ring_tip_head"), 0.012),
}
SEC, RST = 8, 4
FT = torch.float32


NEAR = 0.08


def planes(g):             # face planes of the scaled convex pieces near the hand: list of [F, 4] (n.x + d <= 0 inside)
    key = np.concatenate([g["knuckles"], g["tips"], np.stack([c[2] for c in g["contacts"]])]) if g["contacts"] else None
    out = []
    if key is None:
        return out
    for f in sorted(glob.glob(os.path.join(g["pieces"], "*.obj"))):
        v = np.array([l.split()[1:4] for l in open(f) if l.startswith("v ")], np.float64) * g["scale"]
        if np.linalg.norm(v[:, None] - key[None], axis=-1).min() > NEAR:       # piece far from Shadow's fingers: ignored
            continue
        try:
            out.append(ConvexHull(v).equations.astype(np.float32))
        except Exception:   # noqa: BLE001
            continue
    return out


def kabsch(A, B):
    ca, cb = A.mean(0), B.mean(0); U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]); R = Vt.T @ D @ U.T
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = cb - R @ ca
    return T


def expmap(w):
    th = w.norm(dim=-1).clamp_min(1e-9); k = w / th[:, None]
    K = torch.zeros(len(w), 3, 3, dtype=w.dtype, device=w.device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0], K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 2], k[:, 1], k[:, 2], -k[:, 0], -k[:, 1], k[:, 0]
    s, c = torch.sin(th)[:, None, None], torch.cos(th)[:, None, None]
    return torch.eye(3, dtype=w.dtype, device=w.device) + s * K + (1 - c) * (K @ K)


def seg_dist(p, a, b):
    ab = b - a; t = (((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)).clamp(0, 1)
    return (p - (a + t[..., None] * ab)).norm(dim=-1)


def solve(H, r, G, PL, iters, dev):
    """G: grasps (dicts), PL: their plane lists. -> per grasp None or dict (best restart)."""
    nf = len(H.chains); ds = H.distal_start(); n = len(G)
    lo, up = H.lower.to(dev, FT), H.upper.to(dev, FT)
    pc = np.zeros((n, nf, 3), np.float32); pm = np.zeros((n, nf), np.float32)
    sc = np.zeros((n, nf, SEC, 3), np.float32); sm = np.zeros((n, nf, SEC), np.float32)
    for i, g in enumerate(G):
        best = {}; sec = {f: [] for f in range(nf)}
        for fi, distal, pos, dist, _ in g["contacts"]:
            if fi >= nf:
                continue
            if distal:
                if fi not in best or dist < best[fi][1]:
                    best[fi] = (pos, dist)
            else:
                sec[fi].append(pos)
        for f, (p, _) in best.items():
            pc[i, f] = p; pm[i, f] = 1
        for f, v in sec.items():
            v = v[::max(1, len(v) // SEC)][:SEC]
            for k, p in enumerate(v):
                sc[i, f, k] = p; sm[i, f, k] = 1
    q0 = lo + (up - lo) * torch.tensor([0.1, 0.3, 0.5, 0.7], device=dev, dtype=FT)[:RST, None]
    with torch.no_grad():
        kn = np.stack([c[0, 0].cpu().numpy() for c in H.chain_points(q0[1:2])])
    T0 = torch.tensor(np.stack([kabsch(kn, g["knuckles"][:nf]) for g in G]), dtype=FT, device=dev).repeat_interleave(RST, 0)
    cen = torch.tensor(np.stack([g["knuckles"][:nf].mean(0) for g in G]), dtype=FT, device=dev).repeat_interleave(RST, 0)
    Pm, Tm = max(len(p) for p in PL), max(sum(len(e) for e in p) for p in PL)
    P = torch.zeros(n, Tm, 4); P[:, :, 3] = -1e3; pid = torch.full((n, Tm), Pm, dtype=torch.long)   # padding -> dummy piece Pm
    for i, p in enumerate(PL):
        e = np.concatenate(p); P[i, :len(e)] = torch.from_numpy(e)
        pid[i, :len(e)] = torch.from_numpy(np.concatenate([np.full(len(x), j) for j, x in enumerate(p)]))
    P = P.to(dev).repeat_interleave(RST, 0); pid = pid.to(dev).repeat_interleave(RST, 0)
    pc, pm, sc, sm = (torch.tensor(x, device=dev).repeat_interleave(RST, 0) for x in (pc, pm, sc, sm))
    B = n * RST
    u = torch.logit(((q0 - lo) / (up - lo)).clamp(0.02, 0.98)).repeat(n, 1).clone().requires_grad_(True)
    dt = torch.zeros(B, 3, device=dev, requires_grad=True); dr = torch.zeros(B, 3, device=dev, requires_grad=True)
    opt = torch.optim.Adam([{"params": [u], "lr": 0.05}, {"params": [dt], "lr": 0.002}, {"params": [dr], "lr": 0.02}])
    lin = torch.linspace(0, 1, 5, device=dev)[None, :, None]

    def terms(q):
        E = expmap(dr); R = E @ T0[:, :3, :3]; t = (E @ (T0[:, :3, 3] - cen)[:, :, None]).squeeze(-1) + cen + dt
        ch = [torch.einsum("bij,bnj->bni", R, c) + t[:, None] for c in H.chain_points(q)]
        e_prim = torch.stack([seg_dist(pc[:, f], ch[f][:, ds[f]], ch[f][:, -1]) - r for f in range(nf)], 1)          # [B, nf]
        e_sec = torch.stack([torch.stack([seg_dist(sc[:, f], ch[f][:, i, None], ch[f][:, i + 1, None])
                                          for i in range(ch[f].shape[1] - 1)], 0).min(0).values - r for f in range(nf)], 1)   # [B, nf, SEC]
        pts = torch.cat([torch.cat([c[:, i, None] + (c[:, i + 1, None] - c[:, i, None]) * lin for i in range(c.shape[1] - 1)], 1) for c in ch], 1)
        v = torch.einsum("btk,bnk->bnt", P[..., :3], pts) + P[:, None, :, 3]                                    # [B, N, T]
        sdf = torch.full((B, pts.shape[1], Pm + 1), 1e3, device=dev).scatter_reduce(
            2, pid[:, None].expand(-1, pts.shape[1], -1), v, "amax", include_self=False)[..., :Pm].min(-1).values   # [B, N]
        return e_prim, e_sec, torch.relu(r - sdf), R, t

    for _ in range(iters):
        opt.zero_grad()
        q = lo + torch.sigmoid(u) * (up - lo)
        e_prim, e_sec, pen, _, _ = terms(q)
        loss = (((e_prim * 1e3) ** 2) * pm).sum(1) + 0.3 * ((((e_sec * 1e3) ** 2) * sm).sum(2) / sm.sum(2).clamp_min(1)).sum(1) \
            + 3.0 * ((pen * 1e3) ** 2).sum(1) + 1e-2 * ((dt * 1e3) ** 2).sum(1)
        loss.sum().backward(); opt.step()
    with torch.no_grad():
        q = lo + torch.sigmoid(u) * (up - lo)
        e_prim, _, pen, R, t = terms(q)
        tip = (e_prim.abs() * pm).max(1).values; penm = pen.max(1).values
        b = (tip + penm).view(n, RST).argmin(1) + torch.arange(n, device=dev) * RST
        q, R, t, e_prim, penm = (x[b].cpu().numpy() for x in (q, R, t, e_prim, penm))
    pmn = pm.view(n, RST, nf)[:, 0].cpu().numpy() > 0
    out = []
    for i in range(n):
        if not pmn[i].any():
            out.append(None); continue
        T = np.eye(4); T[:3, :3] = R[i]; T[:3, 3] = t[i]; fs = np.nonzero(pmn[i])[0]
        out.append({"q": q[i].astype(np.float32), "T": T, "tip_err_mm": (np.abs(e_prim[i, fs]) * 1e3).tolist(), "pen_mm": float(penm[i] * 1e3),
                    "fingers": fs.tolist()})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasps", required=True)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--hands", default=",".join(SPEC))
    ap.add_argument("--budget", type=float, default=1e9, help="max B * N * planes per chunk (memory)")
    ap.add_argument("--max_chunk", type=int, default=4096)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    G = [g for f in sorted(glob.glob(a.grasps)) for g in pickle.load(open(f, "rb"))]
    t0 = time.time()
    with Pool(a.workers) as pool:
        PL = pool.map(planes, G, chunksize=16)
    keep = [i for i, p in enumerate(PL) if p]
    G, PL = [G[i] for i in keep], [PL[i] for i in keep]
    size = np.array([sum(len(e) for e in p) for p in PL]); order = np.argsort(size)
    print(f"grasps {len(G)} planes {time.time() - t0:.0f}s  padded planes p50 {np.median(size):.0f} max {size.max()}", flush=True)
    rows = [{"gi": i, "type": g["type"], "file": g["file"], "k": g["k"], "palm_T": g["palm_T"], "shadow_q": g["shadow_q"], "hands": {}}
            for i, g in enumerate(G)]
    for h in a.hands.split(","):
        s = SPEC[h]; H = Hand(os.path.join(a.urdf, s[0]), s[1], s[2])
        npts = sum(5 * (len(c) - 1) for c in H.chains)      # upper bound of sphere centres
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and j - i < a.max_chunk:
                if (j + 1 - i) * RST * npts * size[order[j]] > a.budget:
                    break
                j += 1
            idx = order[i:j]
            res = solve(H, s[3], [G[k] for k in idx], [PL[k] for k in idx], a.iters, dev)
            for k, r in zip(idx, res):
                if r is not None:
                    r["names"] = H.names; rows[k]["hands"][h] = r
            i = j
            print(f"{h} {i}/{len(order)} {time.time() - t0:.0f}s", flush=True)
        ok = [max(r["hands"][h]["tip_err_mm"]) < 5 and r["hands"][h]["pen_mm"] < 2 for r in rows if h in r["hands"]]
        print(f"{h} yield {np.mean(ok):.3f} of {len(ok)}", flush=True)
    hs = a.hands.split(",")
    allok = [all(h in r["hands"] and max(r["hands"][h]["tip_err_mm"]) < 5 and r["hands"][h]["pen_mm"] < 2 for h in hs) for r in rows]
    print(f"all-hands yield {np.mean(allok):.3f} ({int(np.sum(allok))} scenes)", flush=True)
    pickle.dump(rows, open(a.out, "wb"))


if __name__ == "__main__":
    main()
