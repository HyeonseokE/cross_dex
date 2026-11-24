"""Cross-hand contact matching on Dexonomy grasps: kinematic pilot (RSS2027 wrist-pair data, 2026-11-23).

For every Shadow grasp of dexo_extract.py and every target hand, optimise the wrist pose (6D, object frame) and the
finger joints so that the target hand touches the object where Shadow does:
  primary    the distal segment of finger f (last movable joint -> tip) touches Shadow's distal contact point of f
  secondary  some segment of finger f touches Shadow's proximal / middle contact points of f (weight 0.3)
  penetration no finger sphere inside the object
The hand surface is approximated by spheres of radius R[hand] along the finger bones (link origins of the retarget
URDFs; hand meshes are not used). Object = union of its CoACD convex pieces: during optimisation the distance is the
max over face planes per piece (exact inside, a lower bound outside); the reported penetration uses the true signed
distance (trimesh). Wrist init = Kabsch of the target knuckles onto Shadow's knuckles; 4 restarts of the finger curl.
No physics: the result says whether a touching, non-penetrating pose EXISTS, not that the grasp holds.

  python contact_match.py --grasps pilot_grasps.pkl --urdf <dir> --part 0/8 --out res_0.pkl
  python contact_match.py --report 'res_*.pkl' --json pilot_report.json
"""
import argparse
import glob
import json
import os
import pickle

import numpy as np
import torch
import trimesh
from scipy.spatial import ConvexHull

from urdf_fk import Hand

SPEC = {   # urdf (under --urdf), root link, tips thumb..little, sphere radius (m)
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
POWER = {1, 2, 3, 4, 5, 10, 11, 15, 17, 26, 28, 30, 31}   # Feix power grasps (palm / pad opposition, power)
DT = torch.double


def pieces(g):
    out = []
    for f in sorted(glob.glob(os.path.join(g["pieces"], "*.obj"))):
        m = trimesh.load(f, force="mesh", process=False)
        v = np.asarray(m.vertices) * g["scale"]
        out.append(trimesh.Trimesh(v, m.faces, process=False))
    return out


def planes(ps):
    eq = []
    for m in ps:
        try:
            e = ConvexHull(m.vertices).equations            # n.x + d <= 0 inside
        except Exception:   # noqa: BLE001
            continue
        eq.append(torch.tensor(e, dtype=DT))
    F = max(len(e) for e in eq); P = torch.zeros(len(eq), F, 4, dtype=DT); P[:, :, 3] = -1e3
    for i, e in enumerate(eq):
        P[i, :len(e)] = e
    return P


def sdf_planes(P, x):      # x [..., 3] -> [...]: min over pieces of max over faces (n.x + d)
    v = torch.einsum("pfk,...k->...pf", P[:, :, :3], x) + P[:, :, 3]
    return v.max(-1).values.min(-1).values


def sdf_true(ps, x):       # numpy [N,3] -> signed distance (negative inside)
    d = np.full(len(x), 1e3)
    for m in ps:
        d = np.minimum(d, -trimesh.proximity.signed_distance(m, x))
    return d


def seg_dist(p, a, b):     # point [...,3] to segment a-b [...,3]
    ab = b - a; t = (((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)).clamp(0, 1)
    return (p - (a + t[..., None] * ab)).norm(dim=-1)


def kabsch(A, B):          # rigid T with T A ~ B (numpy [N,3])
    ca, cb = A.mean(0), B.mean(0); U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]); R = Vt.T @ D @ U.T
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = cb - R @ ca
    return T


def expmap(w):             # [B,3] -> [B,3,3]
    th = w.norm(dim=-1).clamp_min(1e-9); k = w / th[:, None]
    K = torch.zeros(len(w), 3, 3, dtype=w.dtype)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 0], K[:, 1, 2], K[:, 2, 0], K[:, 2, 1] = -k[:, 2], k[:, 1], k[:, 2], -k[:, 0], -k[:, 1], k[:, 0]
    s, c = torch.sin(th)[:, None, None], torch.cos(th)[:, None, None]
    return torch.eye(3, dtype=w.dtype) + s * K + (1 - c) * (K @ K)


def solve(H, r, g, P, iters, restarts=4):
    nf = len(H.chains); ds = H.distal_start()
    prim = {}; sec = {f: [] for f in range(nf)}
    for fi, distal, pos, dist, _ in g["contacts"]:
        if fi >= nf:
            continue
        if distal:
            if fi not in prim or dist < prim[fi][1]:
                prim[fi] = (pos, dist)
        else:
            sec[fi].append(pos)
    if not prim:
        return None
    q0 = H.lower + (H.upper - H.lower) * torch.tensor([0.1, 0.3, 0.5, 0.7], dtype=DT)[:restarts, None]
    kn = np.stack([c[:, 0].mean(0).numpy() for c in H.chain_points(q0[1:2])])
    T0 = torch.tensor(kabsch(kn, g["knuckles"][:nf]), dtype=DT)
    cen = torch.tensor(g["knuckles"][:nf].mean(0), dtype=DT)
    u = torch.logit(((q0 - H.lower) / (H.upper - H.lower)).clamp(0.02, 0.98)).clone().requires_grad_(True)
    dt = torch.zeros(restarts, 3, dtype=DT, requires_grad=True); dr = torch.zeros(restarts, 3, dtype=DT, requires_grad=True)
    opt = torch.optim.Adam([{"params": [u], "lr": 0.05}, {"params": [dt], "lr": 0.002}, {"params": [dr], "lr": 0.02}])
    pc = {f: torch.tensor(p[0], dtype=DT) for f, p in prim.items()}
    sc = {f: torch.tensor(np.stack(v), dtype=DT) for f, v in sec.items() if v}

    def world(q):
        R = expmap(dr) @ T0[:3, :3]; t = (expmap(dr) @ (T0[:3, 3] - cen)[:, None]).squeeze(-1) + cen + dt
        return [torch.einsum("bij,bnj->bni", R, c) + t[:, None] for c in H.chain_points(q)], R, t

    def terms(q):
        ch, R, t = world(q)
        e_prim = torch.stack([seg_dist(pc[f], ch[f][:, ds[f]], ch[f][:, -1]) - r for f in pc], 1)          # [B, n_prim]
        e_sec = [(torch.stack([seg_dist(sc[f][None, :, :], ch[f][:, i, None], ch[f][:, i + 1, None]) for i in range(ch[f].shape[1] - 1)], 0)
                  .min(0).values - r) for f in sc]
        pts = torch.cat([torch.cat([c[:, i, None] + (c[:, i + 1, None] - c[:, i, None]) * torch.linspace(0, 1, 5, dtype=DT)[None, :, None]
                                    for i in range(c.shape[1] - 1)], 1) for c in ch], 1)                     # [B, N, 3]
        return e_prim, e_sec, pts, R, t

    for it in range(iters):
        opt.zero_grad()
        q = H.lower + torch.sigmoid(u) * (H.upper - H.lower)
        e_prim, e_sec, pts, _, _ = terms(q)
        pen = torch.relu(r - sdf_planes(P, pts))
        loss = ((e_prim * 1e3) ** 2).sum(1) + 0.3 * sum(((e * 1e3) ** 2).mean(1) for e in e_sec) + 3.0 * ((pen * 1e3) ** 2).sum(1) \
            + 1e-2 * ((dt * 1e3) ** 2).sum(1)
        loss.sum().backward(); opt.step()
    with torch.no_grad():
        q = H.lower + torch.sigmoid(u) * (H.upper - H.lower)
        e_prim, _, pts, R, t = terms(q)
    return {"q": q.numpy(), "R": R.numpy(), "t": t.numpy(), "e_prim": e_prim.numpy(), "pts": pts.numpy(), "fingers": list(pc)}


def run(a):
    G = pickle.load(open(a.grasps, "rb")); i0, n = (int(x) for x in a.part.split("/"))
    hands = {h: Hand(os.path.join(a.urdf, s[0]), s[1], s[2]) for h, s in SPEC.items() if h in a.hands.split(",")}
    res = []
    for gi in range(i0, len(G), n):
        g = G[gi]; ps = pieces(g)
        if not ps:
            continue
        P = planes(ps); row = {"gi": gi, "type": g["type"], "palm_T": g["palm_T"], "hands": {}}
        for h, H in hands.items():
            s = solve(H, SPEC[h][3], g, P, a.iters)
            if s is None:
                continue
            B = len(s["q"]); pen = np.stack([np.maximum(SPEC[h][3] - sdf_true(ps, s["pts"][b]), 0).max() for b in range(B)])
            tip = np.abs(s["e_prim"]).max(1); score = tip + pen; b = int(np.argmin(score))
            T = np.eye(4); T[:3, :3] = s["R"][b]; T[:3, 3] = s["t"][b]
            row["hands"][h] = {"q": s["q"][b], "T": T, "tip_err_mm": (np.abs(s["e_prim"][b]) * 1e3).tolist(), "pen_mm": float(pen[b] * 1e3),
                               "fingers": s["fingers"], "names": H.names}
        res.append(row)
        print(gi, g["type"], {h: (round(max(v["tip_err_mm"]), 1), round(v["pen_mm"], 1)) for h, v in row["hands"].items()}, flush=True)
    pickle.dump(res, open(a.out, "wb"))


def report(a):
    rows = [r for f in sorted(glob.glob(a.report)) for r in pickle.load(open(f, "rb"))]
    out = {"n_grasps": len(rows), "criteria": "every targeted fingertip < 5 mm and penetration < 2 mm", "hands": {}}
    ok_all = []
    for h in SPEC:
        R = [(r, r["hands"][h]) for r in rows if h in r["hands"]]
        if not R:
            continue
        tip4 = np.array([max([e for e, f in zip(v["tip_err_mm"], v["fingers"]) if f < 4], default=0.0) for _, v in R])
        use = np.zeros(5)
        for _, v in R:
            use[list(v["fingers"])] += 1
        tip = np.array([max(v["tip_err_mm"]) for _, v in R]); pen = np.array([v["pen_mm"] for _, v in R]); ok = (tip < 5) & (pen < 2)
        pw = np.array([int(r["type"].split("_")[0]) in POWER for r, _ in R])
        rel = [np.linalg.inv(r["palm_T"]) @ v["T"] for (r, v), o in zip(R, ok) if o]
        sp = {}
        if len(rel) > 3:
            t = np.stack([x[:3, 3] for x in rel]); Rm = np.stack([x[:3, :3] for x in rel]); U, _, Vt = np.linalg.svd(Rm.mean(0)); Rbar = U @ Vt
            ang = np.degrees(np.arccos(np.clip((np.einsum("ij,nij->n", Rbar, Rm) - 1) / 2, -1, 1)))
            sp = {"pos_spread_mm": float(np.linalg.norm(t - t.mean(0), axis=1).mean() * 1e3), "rot_spread_deg": float(ang.mean()),
                  "pos_spread_p90_mm": float(np.percentile(np.linalg.norm(t - t.mean(0), axis=1), 90) * 1e3), "rot_spread_p90_deg": float(np.percentile(ang, 90))}
        out["hands"][h] = {"n": len(R), "tip_err_mm_p50": float(np.median(tip)), "tip_err_mm_p90": float(np.percentile(tip, 90)),
                           "pen_mm_p50": float(np.median(pen)), "pen_mm_p90": float(np.percentile(pen, 90)), "yield": float(ok.mean()),
                           "yield_power": float(ok[pw].mean()), "yield_precision": float(ok[~pw].mean()),
                           "yield_common4": float(((tip4 < 5) & (pen < 2)).mean()),
                           "finger_target_share_thumb_index_middle_ring_little": [round(float(x), 2) for x in use / len(R)],
                           "tip_ok": float((tip < 5).mean()), "pen_ok": float((pen < 2).mean()), **sp}
    for r in rows:
        ok_all.append(all(h in r["hands"] and max(r["hands"][h]["tip_err_mm"]) < 5 and r["hands"][h]["pen_mm"] < 2 for h in SPEC))
    out["all5_yield"] = float(np.mean(ok_all))
    out["all5_yield_common4"] = float(np.mean([all(h in r["hands"] and r["hands"][h]["pen_mm"] < 2 and max(
        [e for e, f in zip(r["hands"][h]["tip_err_mm"], r["hands"][h]["fingers"]) if f < 4], default=0.0) < 5 for h in SPEC) for r in rows]))
    bt = {}
    for r, o in zip(rows, ok_all):
        bt.setdefault(r["type"], []).append(o)
    out["all5_by_type"] = {t: round(float(np.mean(v)), 2) for t, v in sorted(bt.items(), key=lambda x: int(x[0].split("_")[0]))}
    print(json.dumps(out, indent=1))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grasps", default=""); ap.add_argument("--urdf", default=""); ap.add_argument("--part", default="0/1")
    ap.add_argument("--hands", default=",".join(SPEC)); ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--out", default=""); ap.add_argument("--report", default=""); ap.add_argument("--json", default="")
    a = ap.parse_args()
    torch.set_num_threads(1)
    report(a) if a.report else run(a)
