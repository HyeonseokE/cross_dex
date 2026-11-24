"""Wrist gap between hands under the policy's canonical wrist frame, on Dexonomy contact-matched scenes (RSS2027, 2026-11-24).

One scene = one Dexonomy Shadow grasp whose fingertip contacts all other hands reproduce (contact_match_grid.py). Per hand
the pose X_h of the canonical frame the multi-hand policies use as "wrist" (CanonicalHandModel of the latent space
checkpoint; fixed per hand, from the URDF geometry) in the object frame. A shared wrist channel assumes X_h equal across
hands for the same grasp; printed: |X_i - X_j| position (mm) / rotation (deg), median over scenes, per hand pair, and the
mean offset vector of every hand from Shadow in Shadow's canonical axes.
C_h (canonical frame in the URDF root link the pairs are stored in) is fitted on fingertips of both models (Kabsch).

  python canon_gap.py --pairs pairs.npz --urdf <urdf dir> --latent <space model.pt>
"""
import argparse
import itertools
import os

import numpy as np
import torch

from train_f6 import HANDS, SHADOW_MJ, SPEC  # noqa: F401
from urdf_fk import Hand


def kabsch(A, B):   # B ~ R A + t
    ca, cb = A.mean(0), B.mean(0); U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb)); R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = cb - R @ ca
    return T, float(np.linalg.norm(A @ R.T + T[:3, 3] - B, axis=1).max() * 1e3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pairs", required=True); ap.add_argument("--urdf", required=True); ap.add_argument("--latent", required=True)
    a = ap.parse_args()
    from dexverse_xlvla.eval_latent import load
    tr = load(a.latent); d = np.load(a.pairs); C = {}
    for h in HANDS:
        s = SPEC[h]; H = Hand(os.path.join(a.urdf, s[0]), s[1], s[2]); m = tr.hand_models[h]; torch.manual_seed(0)
        q = H.lower + torch.rand(200, len(H.names)) * (H.upper - H.lower)
        with torch.no_grad():
            tr_ = torch.stack([c[:, -1] for c in H.chain_points(q)], 1).double().numpy()       # [N, F, 3] root link frame
            nm = [n.replace(".", "_") for n in H.names]; miss = [n for n in m.dof_joints if n.replace(".", "_") not in nm]
            ang = torch.zeros(200, len(m.dof_joints))
            for i, n in enumerate(m.dof_joints):
                if n.replace(".", "_") in nm:
                    ang[:, i] = q[:, nm.index(n.replace(".", "_"))]
            tc = m._forward_internal(m.angles_to_normalized(ang).clamp(-1, 1))[0].double().numpy()   # [N, F', 3] canonical frame
        best = None
        for p in itertools.permutations(range(tc.shape[1]), tr_.shape[1]):
            T, e = kabsch(tc[:, list(p)].reshape(-1, 3), tr_.reshape(-1, 3))
            if best is None or e < best[1]:
                best = (T, e, p)
        C[h] = best[0]; print(f"{h:8s} canonical frame in root link: fit {best[1]:.2f} mm  tip order {best[2]}  model joints not in URDF {miss}", flush=True)
    ok = np.all([d[f"ok_{h}"] for h in HANDS[1:]], 0); X = {"shadow": d["palm_T"][ok].astype(np.float64) @ C["shadow"]}
    for h in HANDS[1:]:
        X[h] = d[f"T_{h}"][ok].astype(np.float64) @ C[h]
    print(f"scenes with all 6 hands: {int(ok.sum())} / {len(ok)}")
    ang = lambda A, B: np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", A[:, :3, :3], B[:, :3, :3]) - 1) / 2, -1, 1)))   # noqa: E731
    print("\nmedian wrist gap, position mm / rotation deg")
    print(" " * 9 + "".join(f"{h:>14s}" for h in HANDS))
    P = np.zeros((6, 6)); R = np.zeros((6, 6))
    for i, hi in enumerate(HANDS):
        for j, hj in enumerate(HANDS):
            P[i, j] = np.median(np.linalg.norm(X[hi][:, :3, 3] - X[hj][:, :3, 3], axis=1)) * 1e3; R[i, j] = np.median(ang(X[hi], X[hj]))
        print(f"{hi:9s}" + "".join(f"{P[i, j]:8.1f}/{R[i, j]:5.1f}" for j in range(6)))
    five = [0, 1, 2, 3]; g = lambda I, J: (np.mean([P[i, j] for i in I for j in J if i != j]), np.mean([R[i, j] for i in I for j in J if i != j]))   # noqa: E731
    for name, I, J in (("5-finger hands among themselves", five, five), ("allegro vs 5-finger", [4], five), ("leap vs 5-finger", [5], five), ("allegro vs leap", [4], [5])):
        print(f"{name:34s} {g(I, J)[0]:6.1f} mm {g(I, J)[1]:5.1f} deg")
    print("\noffset from shadow in shadow's canonical axes (x fingers, mm): mean vector, |mean|, mean |.|")
    S = X["shadow"]
    for h in HANDS[1:]:
        v = np.einsum("nji,nj->ni", S[:, :3, :3], X[h][:, :3, 3] - S[:, :3, 3]) * 1e3
        print(f"{h:8s} {np.round(v.mean(0), 1).tolist()}  |mean| {np.linalg.norm(v.mean(0)):.1f}  mean|.| {np.linalg.norm(v, axis=1).mean():.1f}  std {np.round(v.std(0), 1).tolist()}")


if __name__ == "__main__":
    main()
