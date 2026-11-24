"""Wrist offset predictor f: planner + simulation "play data 2" vs Dexonomy contact-matched pairs (RSS2027, 2026-11-23).

Everything about f is the colleague's code, imported unchanged from the xhand-wrist-offset-f repo (--xwo): hand model and
input features (pd2.common.Hand: PadFK pad centre / normal / thickness / size, palm centre / normal, canonical palm frame),
model F, loss and optimiser (pd2.train_f.train). Only the two container paths in pd2/common.py are redirected to the repo copy.
One speed change: the q noise (sigma 0.05) is drawn --variants times per scene up front and the features cached, instead of a
fresh sample + FK at every step (the GPU is shared; FK per step made one model take 36 min).
Compared training sets (5 common hands sharpa, allegro, leap, shadow, wuji):
  play2  play_data2/pilot_v0/dataset.npz, kept scenes (per-hand planner, Isaac Lab squeeze + lift, all 5 hands succeed)
  dexo   contact_match_grid.py output: Dexonomy Shadow grasps whose contacts the other 4 hands reproduce (kinematic only),
         optionally subsampled (--sizes) to separate "more data" from "different data"
Our wrist pose (URDF root link in the object frame) is converted to their canonical palm frame: root -> palm_body is the
identity for sharpa / allegro / shadow / wuji and a constant translation for leap (base_link -> palm_lower), checked with
fingertips (0.000 mm); joints are reordered by name.
Each model is evaluated on the test split (20 %) of BOTH sets: in-scene dispersion of the corrected wrists (raw /
per-hand constant fitted on the model's own training set / f) and, leave-one-hand-out, the held-out hand's error.

  python train_f_cmp.py --xwo <repo> --dexo 'pairs*.npz' --urdf <dir> --out <dir> [--sizes 1045,0] [--loho]
"""
import argparse
import glob
import json
import os
import sys
import types

import numpy as np
import torch

LEAP_BASE_TO_PALM = (0.1, 0.04, 0.01)      # palm_lower origin in base_link (fitted on fingertips, residual 0.000 mm)


def load_pd2(xwo):
    src = open(f"{xwo}/play_data2/code/pd2/common.py").read()
    a = 'LC = "/workspace/xhand/latent_codec_v1"'; b = 'PAD_DIR = "/workspace/DexVerse/source/dexverse/dexverse/data_collection/tactile_pads"'
    assert a in src and b in src
    src = src.replace(a, f'LC = "{xwo}"').replace(b, f'PAD_DIR = "{xwo}/tactile_pads"')
    sys.path.insert(0, f"{xwo}/play_data2/code"); import pd2  # noqa: F401
    m = types.ModuleType("pd2.common"); m.__package__ = "pd2"; sys.modules["pd2.common"] = m; exec(compile(src, "pd2/common.py", "exec"), m.__dict__)
    import pd2.train_f as tf
    return m, tf


class Src:
    """Scenes of one data source in the interface pd2.train_f.train expects (W, q, hands, feats, dev)."""
    def __init__(self, name, W, q, hands, dev, seed, tag=None):
        self.name, self.hands, self.dev, self.S = name, hands, dev, len(W)
        self.W = torch.tensor(W, dtype=torch.float32, device=dev); self.q = torch.tensor(q, dtype=torch.float32, device=dev)
        perm = np.random.default_rng(seed).permutation(self.S); nte = int(round(0.2 * self.S))
        self.te, self.tr = np.sort(perm[:nte]), np.sort(perm[nte:]); self.tag = tag

    @torch.no_grad()
    def cache(self, sigma, variants):
        """Features of every scene once: clean + `variants` noisy copies (instead of a fresh noise sample and FK every step)."""
        self.fc = []
        for i, H in enumerate(self.hands):
            out = []
            for v in range(variants + 1):
                parts = []
                for k in range(0, self.S, 65536):
                    q = self.q[k:k + 65536, i, :H.dof]
                    if v > 0:
                        q = torch.maximum(torch.minimum(q + sigma * torch.randn_like(q), H.fk.upper), H.fk.lower)
                    parts.append(H.features(q))
                out.append(tuple(torch.cat([p[j] for p in parts]) for j in range(3)))
            self.fc.append(out)
        self.variants = variants

    def feats(self, i, idx, sigma=0.0):
        if getattr(self, "fc", None) is not None:
            f = self.fc[i][0 if sigma == 0 else 1 + int(torch.randint(self.variants, (1,)))]
            return f[0][idx], f[1][idx], f[2][idx]
        H = self.hands[i]; q = self.q[idx, i, :H.dof]
        if sigma > 0:
            q = torch.maximum(torch.minimum(q + sigma * torch.randn_like(q), H.fk.upper), H.fk.lower)
        return H.features(q)


def load_dexo(pattern, hands, C, urdf, dev, seed, max_scenes):
    from contact_match_gpu import SPEC
    from train_f6 import SHADOW_MJ
    from urdf_fk import Hand as UH
    from xhand_latent.kinematics import palm_transform
    W, Q, ty, ob, sc = [], [], [], [], []
    for f in sorted(glob.glob(pattern)):
        d = np.load(f); names = json.loads(str(d["names"][0])); m = np.all([d[f"ok_{h}"] for h in C.HANDS if h != "shadow"], 0)
        w = np.zeros((int(m.sum()), 5, 4, 4), np.float32); q = np.zeros((int(m.sum()), 5, 22), np.float32)
        for j, (h, H) in enumerate(zip(C.HANDS, hands)):
            P = palm_transform(H.spec)
            if h == "leap":
                K = np.eye(4); K[:3, 3] = LEAP_BASE_TO_PALM; P = K @ P
            if h == "shadow":
                T = d["palm_T"][m]; src = d["shadow_q"][m][:, 7:]; nm = SHADOW_MJ
            else:
                T = d[f"T_{h}"][m]; src = d[f"q_{h}"][m]; nm = [n.replace(".", "_") for n in names[h]]
            w[:, j] = T @ P.astype(np.float32)
            for i, n in enumerate(nm):
                q[:, j, H.fk.qidx[n]] = src[:, i]
        W.append(w); Q.append(q); ty.append(d["type"][m].astype(str)); ob.append(d["obj"][m].astype(str)); sc.append(d["scale"][m])
    W, Q, ty, ob, sc = (np.concatenate(x) for x in (W, Q, ty, ob, sc))
    if max_scenes and len(W) > max_scenes:
        k = np.random.default_rng(seed).permutation(len(W))[:max_scenes]; W, Q, ty, ob, sc = W[k], Q[k], ty[k], ob[k], sc[k]
    src = Src("dexo", W, Q, hands, dev, seed, ty); src.obj, src.scale = ob, sc
    return src


@torch.no_grad()
def evaluate(tf, f, train_src, tr_idx, test_src, train_hands, held=None, max_test=20000):
    dev = test_src.dev; te = torch.as_tensor(test_src.te[:max_test], device=dev); W = test_src.W[te]; E = {"raw": W}
    Wtr = train_src.W[torch.as_tensor(tr_idx[:200000], device=dev)]; mean_tr = tf.se3_mean(Wtr[:, train_hands], 1); C = torch.zeros(5, 4, 4, device=dev)
    for i in train_hands:
        C[i] = tf.se3_mean(tf.inv(mean_tr) @ Wtr[:, i], 0)
    for i in range(5):
        if i not in train_hands:
            C[i] = tf.se3_mean(C[train_hands], 0)
    E["per_hand_const"] = W @ tf.inv(C)[None]
    E["f"] = W @ tf.inv(torch.stack([f(*test_src.feats(i, te)) for i in range(5)], 1))
    res = {}
    for k, v in E.items():
        p, r = tf.dispersion(v); res[k] = {"pos_mm": float(p.mean()), "rot_deg": float(r.mean())}
        if held is not None:
            hp, hr = tf.heldout_err(v, held); res[k]["held_pos_mm"] = float(hp.mean()); res[k]["held_rot_deg"] = float(hr.mean())
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xwo", required=True); ap.add_argument("--dexo", required=True); ap.add_argument("--urdf", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sizes", default="0", help="dexo training-set sizes (scenes; 0 = all), comma separated")
    ap.add_argument("--max_scenes", type=int, default=0); ap.add_argument("--variants", type=int, default=16); ap.add_argument("--loho", action="store_true"); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    a.xwo = os.path.abspath(a.xwo); os.makedirs(a.out, exist_ok=True); logf = open(f"{a.out}/train_log.txt", "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()
    C, tf = load_pd2(a.xwo); dev = a.device; hands = [C.Hand(h, dev) for h in C.HANDS]
    d = np.load(f"{a.xwo}/play_data2/pilot_v0/dataset.npz"); k = d["kept"]
    S = {"play2": Src("play2", d["T_OP"][k], d["q"][k], hands, dev, a.seed), "dexo": load_dexo(a.dexo, hands, C, a.urdf, dev, a.seed, a.max_scenes)}
    for x in S.values():
        x.cache(tf.SIGMA, a.variants)
    log(f"play2 scenes {S['play2'].S} (train {len(S['play2'].tr)})  dexo scenes {S['dexo'].S} (train {len(S['dexo'].tr)})  hands {C.HANDS}")
    runs = [("play2", S["play2"], S["play2"].tr)]
    for n in [int(x) for x in a.sizes.split(",")]:
        tr = S["dexo"].tr if n == 0 or n >= len(S["dexo"].tr) else np.sort(np.random.default_rng(a.seed).permutation(S["dexo"].tr)[:n])
        runs.append((f"dexo_{len(tr)}", S["dexo"], tr))
    runs.append((f"play2+dexo_{len(runs[-1][2])}", None, None))
    R = {"config": {"steps": a.steps, "seed": a.seed, "hands": list(C.HANDS), "dexo": a.dexo, "scenes": {k: v.S for k, v in S.items()}}, "runs": {}}
    for name, src, tr in runs:
        if src is None:        # both sets pooled: scene index = play2 train followed by the last dexo subset
            p, dx, dtr = S["play2"], S["dexo"], runs[-2][2]
            src = Src("mix", torch.cat([p.W[p.tr], dx.W[dtr]]).cpu().numpy(), torch.cat([p.q[p.tr], dx.q[dtr]]).cpu().numpy(), hands, dev, a.seed)
            tr = np.arange(src.S)
            src.cache(tf.SIGMA, a.variants)
        log(f"[{name}] train scenes {len(tr)}")
        f = tf.train(src, list(range(5)), tr, a.steps, a.seed, log); torch.save(f.state_dict(), f"{a.out}/f_{name}.pt")
        r = {"n_train": int(len(tr)), "id": {t: evaluate(tf, f, src, tr, S[t], list(range(5))) for t in S}}
        log(f"[{name}] " + "  ".join(f"test {t}: " + " ".join(f"{k} {v['pos_mm']:.1f}mm/{v['rot_deg']:.1f}deg" for k, v in r["id"][t].items()) for t in S))
        if a.loho:
            r["loho"] = {}
            for h in range(5):
                hs = [i for i in range(5) if i != h]; f = tf.train(src, hs, tr, a.steps, a.seed + 1 + h, lambda s: None)
                r["loho"][C.HANDS[h]] = {t: evaluate(tf, f, src, tr, S[t], hs, held=h) for t in S}
                log(f"[{name}] LOHO {C.HANDS[h]} " + "  ".join(f"test {t}: " + " ".join(f"{k} {v['held_pos_mm']:.1f}mm/{v['held_rot_deg']:.1f}deg"
                                                                                       for k, v in r["loho"][C.HANDS[h]][t].items()) for t in S))
            r["loho_mean"] = {t: {k: {mm: float(np.mean([r["loho"][h][t][k][mm] for h in C.HANDS])) for mm in ("pos_mm", "rot_deg", "held_pos_mm", "held_rot_deg")}
                                  for k in ("raw", "per_hand_const", "f")} for t in S}
            log(f"[{name}] LOHO mean " + json.dumps({t: {k: [round(v["held_pos_mm"], 1), round(v["held_rot_deg"], 1)] for k, v in x.items()} for t, x in r["loho_mean"].items()}))
        R["runs"][name] = r; json.dump(R, open(f"{a.out}/metrics.json", "w"), indent=1)
    log("DONE")


if __name__ == "__main__":
    main()
