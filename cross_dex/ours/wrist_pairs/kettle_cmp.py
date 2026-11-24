"""Offline wrist alignment of f models on our GraspKettle teleop demos (RSS2027, 2026-11-23).

5 hands of the colleague's hand metadata (sharpa, allegro, leap, shadow, wuji), 50 demos each. Per step: canonical palm
("wrist") pose in the kettle frame (their recipe: reference FK of the recorded joint_position on the recorded root pose,
palm_body o palm_transform) and finger q. EE_common = wrist o off^-1 with off =
  raw             identity
  per_hand_const  one constant per hand fitted on episodes 0-39 (the per-hand wrist calibration idea)
  <name>          f(q) of a checkpoint of pd2.train_f.F (--f name=path ...)
Phase: t_lift = first step (>= 15) the kettle is 1 cm above its height at step 10 for 5 steps; "pre" = the 40 steps before t_lift
(approach + grasp), "post" = t_lift .. end (lift + hold), 20 bins each. Metric on episodes 40-49 (never used for the
constant): distance between the hands' per-bin mean EE_common (mean over hand pairs, median over bins; mm / deg) -- what a
shared wrist channel must absorb -- and the spread inside one hand (mm / deg).

  python kettle_cmp.py --xwo <repo> --pkl <teleop_pkl root> --f g2=<f_id.pt> dexo=<f.pt> --json out.json
"""
import argparse
import glob
import json
import os
import pickle

import numpy as np
import torch

from train_f_cmp import load_pd2

TASK = "Dexverse-GraspKettle-v0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xwo", required=True); ap.add_argument("--pkl", required=True); ap.add_argument("--f", nargs="*", default=[])
    ap.add_argument("--json", default=""); ap.add_argument("--cache", default="kettle_wrist.npz"); ap.add_argument("--dump_const", default="")
    a = ap.parse_args()
    a.xwo = os.path.abspath(a.xwo); C, tf = load_pd2(a.xwo)
    from xhand_latent.kinematics import palm_transform
    from xhand_latent.reference_kin import make_T
    hands = [C.Hand(h, "cpu") for h in C.HANDS]
    if not os.path.exists(a.cache):
        out = {}
        for h, H in zip(C.HANDS, hands):
            f = max(glob.glob(f"{a.pkl}/*/{TASK}/{TASK}_*_floating_{h}_right.pkl"), key=os.path.getsize); d = pickle.load(open(f, "rb"))
            names = d["action_layout"]["robot_joint_names"]; fk = H.fk; P = palm_transform(fk.spec); W, Q, EP, T, Z = [], [], [], [], []
            for e, ep in enumerate(d["episodes"]):
                S = ep["states"]; st = lambda k1, k2, k3: np.stack([np.asarray(s[k1][k2][k3])[0] for s in S]).astype(np.float64)   # noqa: E731
                jp, root, obj = st("articulation", "robot", "joint_position"), st("articulation", "robot", "root_pose"), st("rigid_object", "object", "root_pose")
                q_ref = jp[:, [names.index(n) for n in fk.reference.joint_order]]
                rT = np.stack([make_T(x[:3], x[3:]) for x in root]); oT = np.stack([make_T(x[:3], x[3:]) for x in obj])
                if np.abs(root - root[:1]).max() < 1e-6:
                    palm = np.asarray(fk.reference.fk(q_ref, root_T=rT[0])[fk.spec["palm_body"]])
                else:
                    palm = np.stack([np.asarray(fk.reference.fk(q_ref[i:i + 1], root_T=rT[i])[fk.spec["palm_body"]])[0] for i in range(len(q_ref))])
                W.append(np.linalg.inv(oT) @ palm @ P); Q.append(jp[:, [names.index(n) for n in H.jn]])
                EP.append(np.full(len(S), e)); T.append(np.arange(len(S))); Z.append(obj[:, 2])
            out[f"{h}_W"], out[f"{h}_q"], out[f"{h}_ep"], out[f"{h}_t"], out[f"{h}_z"] = (np.concatenate(x) for x in (W, Q, EP, T, Z))
            print("extracted", h, len(d["episodes"]), "episodes", len(out[f"{h}_W"]), "steps", flush=True)
        np.savez(a.cache, **out)
    D = np.load(a.cache)
    # phase bins: (segment 0 pre / 1 post, bin 0..19) per step, -1 = unused
    data = []
    for h, H in zip(C.HANDS, hands):
        W, q, ep, t, z = (D[f"{h}_{k}"] for k in ("W", "q", "ep", "t", "z")); seg = np.full(len(W), -1); b = np.zeros(len(W), int); nolift = 0
        for e in np.unique(ep):
            m = np.nonzero(ep == e)[0]; up = z[m] > z[m][min(10, len(m) - 1)] + 0.01; up[:15] = False   # the kettle settles ~1 cm in the first steps
            run = np.convolve(up, np.ones(5), "valid") == 5
            if not run.any():
                nolift += 1; continue
            tl = int(np.argmax(run)); pre = np.arange(max(tl - 40, 0), tl); post = np.arange(tl, len(m))
            seg[m[pre]] = 0; b[m[pre]] = np.minimum((np.arange(len(pre)) * 20) // max(len(pre), 1), 19)
            seg[m[post]] = 1; b[m[post]] = np.minimum((np.arange(len(post)) * 20) // max(len(post), 1), 19)
        data.append({"W": torch.tensor(W, dtype=torch.float32), "q": torch.tensor(q, dtype=torch.float32), "ep": ep, "bin": seg * 20 + b, "ok": seg >= 0})
        print(h, "episodes", len(np.unique(ep)), "without lift", nolift, flush=True)

    def binmeans(E, val):       # E list per hand [N,4,4] -> [5, 40, 4, 4] (nan where empty) and within-hand spread
        M = torch.full((5, 40, 4, 4), float("nan")); wp, wr = [], []
        for i, d in enumerate(data):
            for k in range(40):
                m = d["ok"] & (d["bin"] == k) & ((d["ep"] >= 40) if val else (d["ep"] < 40))
                if m.sum() == 0:
                    continue
                x = E[i][torch.as_tensor(m)]; mu = tf.se3_mean(x, 0); M[i, k] = mu
                wp.append(float((x[:, :3, 3] - mu[:3, 3]).norm(dim=-1).mean() * 1e3)); wr.append(float(torch.rad2deg(tf.angle(x[:, :3, :3], mu[:3, :3])).mean()))
        return M, float(np.mean(wp)), float(np.mean(wr))

    def metric(E):
        M, wp, wr = binmeans(E, True); P, R = [], []
        for k in range(40):
            dp, dr = [], []
            for i in range(5):
                for j in range(i + 1, 5):
                    if torch.isnan(M[i, k]).any() or torch.isnan(M[j, k]).any():
                        continue
                    dp.append(float((M[i, k, :3, 3] - M[j, k, :3, 3]).norm() * 1e3)); dr.append(float(torch.rad2deg(tf.angle(M[i, k, :3, :3], M[j, k, :3, :3]))))
            P.append(np.mean(dp)); R.append(np.mean(dr))
        P, R = np.array(P), np.array(R)
        return {"pair_mm": float(np.median(P)), "pair_deg": float(np.median(R)), "pair_mm_pre": float(np.median(P[:20])), "pair_deg_pre": float(np.median(R[:20])),
                "pair_mm_post": float(np.median(P[20:])), "pair_deg_post": float(np.median(R[20:])), "within_mm": wp, "within_deg": wr}

    res = {}
    raw = [d["W"] for d in data]; res["raw"] = metric(raw)
    Mtr, _, _ = binmeans(raw, False); good = ~torch.isnan(Mtr).flatten(2).any(2).any(0)
    mean_b = tf.se3_mean(Mtr[:, good], 0); Cc = torch.stack([tf.se3_mean(tf.inv(mean_b) @ Mtr[i, good], 0) for i in range(5)])
    res["per_hand_const"] = metric([d["W"] @ tf.inv(Cc[i]) for i, d in enumerate(data)])
    if a.dump_const:
        json.dump({h: Cc[i].tolist() for i, h in enumerate(C.HANDS)}, open(a.dump_const, "w"))
    for spec in a.f:
        name, path = spec.split("=", 1); f = tf.F(); f.load_state_dict(torch.load(path, map_location="cpu")); f.eval()
        with torch.no_grad():
            res[name] = metric([d["W"] @ tf.inv(f(*H.features(d["q"]))) for d, H in zip(data, hands)])
    for k, v in res.items():
        print(f"{k:22s} pair {v['pair_mm']:6.1f} mm / {v['pair_deg']:5.1f} deg   pre {v['pair_mm_pre']:6.1f} / {v['pair_deg_pre']:5.1f}   "
              f"post {v['pair_mm_post']:6.1f} / {v['pair_deg_post']:5.1f}   within {v['within_mm']:5.1f} mm / {v['within_deg']:4.1f} deg", flush=True)
    if a.json:
        json.dump({"hands": list(C.HANDS), "methods": res, "f": a.f}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
