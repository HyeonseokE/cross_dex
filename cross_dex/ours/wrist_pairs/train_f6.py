"""Wrist offset predictor f on 6-hand contact-matched pairs (RSS2027, 2026-11-23).

Method of the colleague's G2 (xhand-wrist-offset-f, play_data2/code/pd2/train_f.py), kept: model (shared per-finger MLP ->
mean|max pool -> head -> 9D pos + rot6d, zero-init), loss (pairwise dispersion of EE_common_i = wrist_i o f(x_i)^-1 inside a
scene + mean-zero term, W_ROT 0.10 m/rad, LAMBDA 1), q noise 0.05 rad, 6000 steps AdamW 1e-3 cosine, batch 128 scenes,
80/20 scene split, in-distribution + leave-one-hand-out, baselines raw / per-hand constant.
Different (their hand metadata and tactile-pad meshes are not available to us):
  data      scenes = Dexonomy Shadow grasps whose contacts all 5 other hands reproduce (contact_match*.py; kinematic only),
            6 hands (shadow, inspire, sharpa, wuji, allegro, leap) instead of 5 planner-generated ones
  wrist     canonical palm frame built from the URDF: origin = mean non-thumb knuckle, x = extended finger direction,
            y = towards the index side, z = x cross y (dorsal for a right hand)
  features  per finger [fingertip position 3, distal bone direction 3, finger sphere radius, distal bone length] instead of
            [pad centre 3, pad normal 3, pad thickness, pad size]; palm [thumb knuckle 3, palm width, finger length, n fingers / 5]
            instead of [palm pad centre 3, normal 3]
  baselines Kabsch-E0 / generator frame need the pad model -> not computed

  python train_f6.py --pairs 'res_*.pkl' --grasps pilot_grasps.pkl --urdf <dir> --out <dir>
"""
import argparse
import glob
import json
import math
import os
import pickle

import numpy as np
import torch
import torch.nn as nn

from urdf_fk import Hand

SPEC = {
    "shadow": ("shadow/retarget/floating_shadow_right.urdf", "palm", ("thtip", "fftip", "mftip", "rftip", "lftip"), 0.009),
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
HANDS = list(SPEC)
SHADOW_MJ = ["FFJ4", "FFJ3", "FFJ2", "FFJ1", "MFJ4", "MFJ3", "MFJ2", "MFJ1", "RFJ4", "RFJ3", "RFJ2", "RFJ1", "LFJ5", "LFJ4", "LFJ3",
             "LFJ2", "LFJ1", "THJ5", "THJ4", "THJ3", "THJ2", "THJ1"]      # Dexonomy qpos[7:] order (rh_ prefix dropped)
POWER = {1, 2, 3, 4, 5, 10, 11, 15, 17, 26, 28, 30, 31}
W_ROT, LAMBDA, SIGMA, PAD = 0.10, 1.0, 0.05, 22
FT = torch.float32


def rot6d_to_R(d):
    a, b = d[..., :3], d[..., 3:6]
    x = nn.functional.normalize(a, dim=-1, eps=1e-8); b = b - (x * b).sum(-1, keepdim=True) * x
    y = nn.functional.normalize(b, dim=-1, eps=1e-8)
    return torch.stack([x, y, torch.cross(x, y, dim=-1)], -1)


def make_T(R, t):
    T = torch.zeros(R.shape[:-2] + (4, 4), device=R.device, dtype=R.dtype)
    T[..., :3, :3] = R; T[..., :3, 3] = t; T[..., 3, 3] = 1
    return T


def inv(T):
    Rt = T[..., :3, :3].transpose(-1, -2)
    return make_T(Rt, -(Rt @ T[..., :3, 3, None])[..., 0])


def proj_so3(M):
    U, _, Vh = torch.linalg.svd(M); d = torch.sign(torch.det(U @ Vh))
    return U @ torch.diag_embed(torch.stack([torch.ones_like(d), torch.ones_like(d), d], -1)) @ Vh


def se3_mean(T, dim):
    return make_T(proj_so3(T[..., :3, :3].mean(dim)), T[..., :3, 3].mean(dim))


def angle(Ra, Rb):
    return torch.acos((((Ra.transpose(-1, -2) @ Rb).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))


class F(nn.Module):     # = pd2.train_f.F
    def __init__(self, h=128):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(8 + 5, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, h))
        self.head = nn.Sequential(nn.Linear(2 * h + 6, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 9))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("fscale", torch.tensor([10.] * 3 + [1.] * 3 + [50., 50.]))
        self.register_buffer("pscale", torch.tensor([10.] * 6))
        self.register_buffer("r6", torch.tensor([1., 0, 0, 0, 1, 0]))

    def forward(self, fing, mask, palm):
        B = fing.shape[0]; oh = torch.eye(5, device=fing.device)[None].expand(B, 5, 5)
        e = self.enc(torch.cat([fing * self.fscale, oh], -1)); m = mask[..., None]
        mean = (e * m).sum(1) / m.sum(1).clamp_min(1); mx = e.masked_fill(m == 0, -1e4).max(1).values
        o = self.head(torch.cat([mean, mx, palm * self.pscale], -1))
        return make_T(rot6d_to_R(o[:, 3:] + self.r6), o[:, :3] * 0.1)


class HandF:
    """URDF hand + canonical palm frame + f input features."""
    def __init__(self, name, urdf_dir, dev):
        s = SPEC[name]; self.name, self.r, self.dev = name, s[3], dev
        self.H = H = Hand(os.path.join(urdf_dir, s[0]), s[1], s[2]); self.nf = len(H.chains); self.ds = H.distal_start()
        self.lo, self.up = H.lower.to(dev, FT), H.upper.to(dev, FT); self.dof = len(H.names)
        with torch.no_grad():
            fr = torch.linspace(0, 1, 11, device=dev)[:, None]; ch = H.chain_points(self.lo + fr * (self.up - self.lo))
            kn = torch.stack([c[0, 0] for c in ch]); ext = torch.stack([(c[:, -1] - c[:, 0]).norm(dim=-1) for c in ch[1:]]).mean(0)
            k = int(ext.argmax())                                                      # most extended of the 11 curl fractions
            x = nn.functional.normalize(torch.stack([c[k, -1] - c[k, 0] for c in ch[1:]]).mean(0), dim=0)
            y = kn[1] - kn[self.nf - 1]; y = nn.functional.normalize(y - (y @ x) * x, dim=0)
            C = torch.eye(4, device=dev); C[:3, 0], C[:3, 1], C[:3, 2], C[:3, 3] = x, y, torch.linalg.cross(x, y), kn[1:].mean(0)
            self.C, self.Ci = C, torch.linalg.inv(C)
            thk = self.Ci[:3, :3] @ kn[0] + self.Ci[:3, 3]
            self.palm = torch.cat([thk, torch.stack([(kn[1] - kn[self.nf - 1]).norm(), ext.max(), torch.tensor(self.nf / 5, device=dev)])])
        self.mask = torch.tensor([1.] * self.nf + [0.] * (5 - self.nf), device=dev)

    def features(self, q):
        B = q.shape[0]; ch = self.H.chain_points(q); fing = torch.zeros(B, 5, 8, device=self.dev)
        for f, c in enumerate(ch):
            tip = c[:, -1] @ self.Ci[:3, :3].T + self.Ci[:3, 3]; d = (c[:, -1] - c[:, self.ds[f]]) @ self.Ci[:3, :3].T
            L = d.norm(dim=-1, keepdim=True)
            fing[:, f] = torch.cat([tip, d / L.clamp_min(1e-6), torch.full((B, 1), self.r, device=self.dev), L], -1)
        return fing, self.mask[None].expand(B, 5), self.palm[None].expand(B, 6)


class Data:
    def __init__(self, a, dev):
        self.hands = [HandF(h, a.urdf, dev) for h in HANDS]; self.dev = dev; sh = self.hands[0]; si = [SHADOW_MJ.index(n) for n in sh.H.names]
        if a.npz:                                            # contact_match_grid.py output shards
            W, Q, ty, tips = [], [], [], []
            for f in sorted(glob.glob(a.npz)):
                d = np.load(f); m = np.all([d[f"ok_{h}"] for h in HANDS[1:]], 0)
                if a.tip_mm != 5.0 or a.pen_mm != 2.0:
                    m = m & np.all([(d[f"tip_{h}"] < a.tip_mm) & (d[f"pen_{h}"] < a.pen_mm) for h in HANDS[1:]], 0)
                w = np.zeros((int(m.sum()), 6, 4, 4), np.float32); q = np.zeros((int(m.sum()), 6, PAD), np.float32)
                w[:, 0] = d["palm_T"][m]; q[:, 0, :sh.dof] = d["shadow_q"][m][:, 7:][:, si]
                for j, h in enumerate(HANDS[1:], 1):
                    w[:, j] = d[f"T_{h}"][m]; q[:, j, :self.hands[j].dof] = d[f"q_{h}"][m]
                W.append(w); Q.append(q); ty.append(d["type"][m].astype(str)); tips.append(d["tips"][m])
            W, Q, ty, tips = np.concatenate(W), np.concatenate(Q), np.concatenate(ty), np.concatenate(tips)
            if a.max_scenes and len(W) > a.max_scenes:
                k = np.random.default_rng(a.seed).permutation(len(W))[:a.max_scenes]; W, Q, ty, tips = W[k], Q[k], ty[k], tips[k]
            Wp = torch.tensor(W[:, 0], device=dev)
            self.W = torch.tensor(W, device=dev) @ torch.stack([h.C for h in self.hands])[None]; self.q = torch.tensor(Q, device=dev)
            self.S = len(W); self.type = ty.tolist(); self.power = np.array([int(t.split("_")[0]) in POWER for t in self.type])
            with torch.no_grad():
                k = slice(0, 20000); ch = sh.H.chain_points(self.q[k, 0, :sh.dof]); tip = torch.stack([c[:, -1] for c in ch], 1)
                tw = torch.einsum("sij,sfj->sfi", Wp[k, :3, :3], tip) + Wp[k, None, :3, 3]
                self.shadow_check_mm = float((tw - torch.tensor(tips[k], device=dev)).norm(dim=-1).mean() * 1e3)
            return
        rows = [r for f in sorted(glob.glob(a.pairs)) for r in pickle.load(open(f, "rb"))]
        G = pickle.load(open(a.grasps, "rb")) if a.grasps else None
        ok = lambda v: max(v["tip_err_mm"]) < a.tip_mm and v["pen_mm"] < a.pen_mm   # noqa: E731
        rows = [r for r in rows if all(h in r["hands"] and ok(r["hands"][h]) for h in HANDS[1:])]
        S = len(rows); W = np.zeros((S, 6, 4, 4), np.float32); Q = np.zeros((S, 6, PAD), np.float32)
        for i, r in enumerate(rows):
            sq = r["shadow_q"] if "shadow_q" in r else G[r["gi"]]["shadow_q"]
            W[i, 0] = r["palm_T"]; Q[i, 0, :sh.dof] = np.asarray(sq)[7:][si]
            for j, h in enumerate(HANDS[1:], 1):
                W[i, j] = r["hands"][h]["T"]; Q[i, j, :self.hands[j].dof] = r["hands"][h]["q"]
        self.W = torch.tensor(W, device=dev) @ torch.stack([h.C for h in self.hands])[None]     # canonical palm in the object frame
        self.q = torch.tensor(Q, device=dev); self.S, self.dev = S, dev
        self.power = np.array([int(r["type"].split("_")[0]) in POWER for r in rows]); self.type = [r["type"] for r in rows]
        if G is not None:      # Shadow URDF vs Dexonomy MJCF: fingertips must agree
            with torch.no_grad():
                ch = sh.H.chain_points(self.q[:, 0, :sh.dof]); tip = torch.stack([c[:, -1] for c in ch], 1)
                Wp = torch.tensor(np.stack([r["palm_T"] for r in rows]), dtype=FT, device=dev)
                tw = torch.einsum("sij,sfj->sfi", Wp[:, :3, :3], tip) + Wp[:, None, :3, 3]
                ref = torch.tensor(np.stack([G[r["gi"]]["tips"] for r in rows]), dtype=FT, device=dev)
                self.shadow_check_mm = float((tw - ref).norm(dim=-1).mean() * 1e3)

    def feats(self, i, idx, sigma=0.0):
        H = self.hands[i]; q = self.q[idx, i, :H.dof]
        if sigma > 0:
            q = torch.maximum(torch.minimum(q + sigma * torch.randn_like(q), H.up), H.lo)
        return H.features(q)


def train(D, hands, idx, steps, seed, log):
    torch.manual_seed(seed)
    f = F().to(D.dev); opt = torch.optim.AdamW(f.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps); idx_t = torch.as_tensor(idx, device=D.dev); n = len(hands)
    for s in range(steps):
        sb = idx_t[torch.randint(len(idx_t), (128,), device=D.dev)]
        T = torch.stack([f(*D.feats(i, sb, SIGMA)) for i in hands], 1); E = D.W[sb][:, hands] @ inv(T)
        pair = 0
        for a in range(n):
            for b in range(a + 1, n):
                pair = pair + (E[:, a, :3, 3] - E[:, b, :3, 3]).norm(dim=-1) + \
                    W_ROT * (E[:, a, :3, :3] - E[:, b, :3, :3]).flatten(1).norm(dim=-1) / math.sqrt(2)
        pair = pair.mean() / (n * (n - 1) / 2)
        zero = T[..., :3, 3].mean(1).norm(dim=-1).mean() + \
            W_ROT * (T[..., :3, :3].mean(1) - torch.eye(3, device=D.dev)).flatten(1).norm(dim=-1).mean() / math.sqrt(2)
        loss = pair + LAMBDA * zero
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if s % 1000 == 0 or s == steps - 1:
            log(f"  step {s} pair={pair.item() * 1000:.2f}mm-eq zero={zero.item() * 1000:.2f}")
    return f.eval()


@torch.no_grad()
def evaluate(D, f, tr, te, train_hands, held=None):
    tr_t, te_t = torch.as_tensor(tr, device=D.dev), torch.as_tensor(te, device=D.dev); W = D.W[te_t]; E = {"raw": W}
    Wtr = D.W[tr_t]; mean_tr = se3_mean(Wtr[:, train_hands], 1); C = torch.zeros(6, 4, 4, device=D.dev)
    for i in train_hands:
        C[i] = se3_mean(inv(mean_tr) @ Wtr[:, i], 0)
    for i in range(6):
        if i not in train_hands:
            C[i] = se3_mean(C[train_hands], 0)
    E["per_hand_const"] = W @ inv(C)[None]
    E["f"] = W @ inv(torch.stack([f(*D.feats(i, te_t)) for i in range(6)], 1))
    res = {}; pw = D.power[te]
    for k, v in E.items():
        m = se3_mean(v, 1); p = ((v[..., :3, 3] - m[:, None, :3, 3]).norm(dim=-1).mean(1) * 1000).cpu().numpy()
        r = torch.rad2deg(angle(v[..., :3, :3], m[:, None, :3, :3])).mean(1).cpu().numpy()
        res[k] = {"pos_mm": float(p.mean()), "rot_deg": float(r.mean()), "pos_mm_power": float(p[pw].mean()), "pos_mm_precision": float(p[~pw].mean())}
        if held is not None:
            o = [i for i in range(6) if i != held]; mo = se3_mean(v[:, o], 1)
            res[k]["held_pos_mm"] = float((v[:, held, :3, 3] - mo[:, :3, 3]).norm(dim=-1).mean() * 1000)
            res[k]["held_rot_deg"] = float(torch.rad2deg(angle(v[:, held, :3, :3], mo[:, :3, :3])).mean())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default=""); ap.add_argument("--npz", default=""); ap.add_argument("--max_scenes", type=int, default=0)
    ap.add_argument("--max_test", type=int, default=50000); ap.add_argument("--grasps", default=""); ap.add_argument("--urdf", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--tip_mm", type=float, default=5.0); ap.add_argument("--pen_mm", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); logf = open(f"{a.out}/train_log.txt", "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    D = Data(a, torch.device(a.device)); rng = np.random.default_rng(a.seed); perm = rng.permutation(D.S); nte = int(round(0.2 * D.S))
    te, tr = np.sort(perm[:nte][:a.max_test]), np.sort(perm[nte:])
    log(f"scenes {D.S}: train {len(tr)} test {len(te)}  power {int(D.power.sum())}  shadow urdf-vs-mjcf tip {getattr(D, 'shadow_check_mm', -1):.1f} mm")
    for h in D.hands:
        log(f"  {h.name} palm feature {[round(float(x), 3) for x in h.palm]}")
    R = {"config": {"sigma": SIGMA, "w_rot": W_ROT, "lambda": LAMBDA, "steps": a.steps, "n_train": len(tr), "n_test": len(te), "seed": a.seed,
                    "hands": HANDS, "pairs": a.pairs or a.npz, "scenes": D.S, "tip_mm": a.tip_mm, "pen_mm": a.pen_mm}}
    f = train(D, list(range(6)), tr, a.steps, a.seed, log); R["in_distribution"] = evaluate(D, f, tr, te, list(range(6)))
    torch.save(f.state_dict(), f"{a.out}/f_id.pt"); log("ID " + json.dumps(R["in_distribution"])); R["loho"] = {}
    for h in range(6):
        hs = [i for i in range(6) if i != h]; f = train(D, hs, tr, a.steps, a.seed + 1 + h, log)
        R["loho"][HANDS[h]] = r = evaluate(D, f, tr, te, hs, held=h); torch.save(f.state_dict(), f"{a.out}/f_loho_{HANDS[h]}.pt")
        log(f"LOHO {HANDS[h]} " + "  ".join(f"{k}: {v['pos_mm']:.1f}mm/{v['rot_deg']:.1f}deg held {v['held_pos_mm']:.1f}mm/{v['held_rot_deg']:.1f}deg" for k, v in r.items()))
    R["loho_mean"] = {k: {m: float(np.mean([R["loho"][h][k][m] for h in HANDS])) for m in R["loho"][HANDS[0]][k]} for k in ("raw", "per_hand_const", "f")}
    json.dump(R, open(f"{a.out}/metrics.json", "w"), indent=1); log("LOHO mean " + json.dumps(R["loho_mean"]))


if __name__ == "__main__":
    main()
