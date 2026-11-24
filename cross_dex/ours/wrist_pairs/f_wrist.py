"""Wrist conversion between hands through an offset model (RSS2027, 2026-11-23): EE_common = palm_s o off_s^-1, palm_t = EE_common o off_t.

palm = the colleague's canonical palm frame (xhand-wrist-offset-f repo: reference FK of the six floating wrist joints on the
articulation root pose, palm_body o palm_transform). off:
  palm          identity (the canonical palm pose itself is copied)
  const:<json>  one constant 4x4 per hand (kettle_cmp.py --dump_const)
  f:<ckpt>      pd2.train_f.F on the posed hand geometry (finger joints of the same step)
The target palm pose is turned into the target hand's six wrist joints with their exact floating-wrist solver, warm-started
from the previous step.
"""
import json

import numpy as np
import torch

from cross_dex.ours.wrist_pairs.train_f_cmp import load_pd2


class FWrist:
    def __init__(self, xwo, spec):
        self.C, self.tf = load_pd2(xwo); self.kind, _, arg = spec.partition(":"); self.hands = {}
        if self.kind == "f":
            self.f = self.tf.F(); self.f.load_state_dict(torch.load(arg, map_location="cpu")); self.f.eval()
        elif self.kind == "const":
            self.const = {h: np.asarray(v, np.float64) for h, v in json.load(open(arg)).items()}
        elif self.kind != "palm":
            raise ValueError(spec)

    def hand(self, h):
        if h not in self.hands:
            self.hands[h] = self.C.Hand(h, "cpu")
        return self.hands[h]

    def _root(self, pose7):
        from xhand_latent.reference_kin import make_T
        return make_T(np.asarray(pose7[:3], np.float64), np.asarray(pose7[3:], np.float64))

    def palm(self, h, names, q, root7):          # six wrist joints (names, [T,6]) -> canonical palm poses [T,4,4]
        from xhand_latent.kinematics import palm_transform
        H = self.hand(h); kin = H.fk.reference; full = np.zeros((len(q), len(kin.joint_order)))
        for i, n in enumerate(names):
            full[:, kin.qidx[n]] = q[:, i]
        return np.asarray(kin.fk(full, root_T=self._root(root7))[H.spec["palm_body"]]) @ palm_transform(H.spec)

    def off(self, h, names, q):                  # finger joints (names, [T,n] rad) -> offsets [T,4,4]
        T = len(q)
        if self.kind == "palm":
            return np.tile(np.eye(4), (T, 1, 1))
        if self.kind == "const":
            return np.tile(self.const[h], (T, 1, 1))
        H = self.hand(h); nm = [n.replace(".", "_") for n in names]; x = torch.zeros(T, H.dof)
        for n in H.jn:
            x[:, H.fk.qidx[n]] = torch.as_tensor(q[:, nm.index(n)], dtype=torch.float32)
        with torch.no_grad():
            return self.f(*H.features(x)).double().numpy()

    def convert(self, src, dst, s_wn, s_wq, s_root, s_fn, s_fq, d_wn, d_root, d_fn, d_fq):
        """-> dst wrist joints [T,6] in the order d_wn, info."""
        from xhand_latent.kinematics import palm_transform
        from xhand_latent.wrist import floating_wrist
        Ps = self.palm(src, s_wn, s_wq, s_root); Pt = Ps @ np.linalg.inv(self.off(src, s_fn, s_fq)) @ self.off(dst, d_fn, d_fq)
        H = self.hand(dst); kin = H.fk.reference; body = Pt @ np.linalg.inv(palm_transform(H.spec)); root = self._root(d_root)
        q = np.zeros((1, len(kin.joint_order))); out = []; err = 0.0
        for t in range(len(body)):
            q, st = floating_wrist(kin, H.spec["palm_body"], H.spec["wrist_joints"], body[t:t + 1], root, q)
            out.append([q[0, kin.qidx[n]] for n in d_wn]); err = max(err, st["max_wrist_position_error_mm"])
        d = np.linalg.norm(Pt[:, :3, 3] - Ps[:, :3, 3], axis=1) * 1e3
        return np.asarray(out), {"palm_shift_mm_mean": round(float(d.mean()), 1), "palm_shift_mm_max": round(float(d.max()), 1), "ik_err_mm": round(err, 4)}


LEAP_BASE_TO_PALM = (0.1, 0.04, 0.01)      # palm_lower origin in base_link (= train_f_cmp.py)


def _se3_mean(Ts):
    Ts = np.asarray(Ts); U, _, Vt = np.linalg.svd(Ts[:, :3, :3].mean(0)); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    out = np.eye(4); out[:3, :3] = U @ D @ Vt; out[:3, 3] = Ts[:, :3, 3].mean(0)
    return out


class PolicyWristOffset:
    """Wrist offset model applied at INFERENCE to a policy trained with our canonical-frame wrist X (no retraining).

    With B_h = (pose of the offset model's palm frame in our canonical frame, fitted on fingertips of both hand models) and G_h(q) = B_h off_h(q)^-1, a hand's EE_common is
    X_h G_h. The policy's X is read as the wrist of its training hands: EE = X mean_s G_s(q_s), q_s = D_s(z) (+ self-residual)
    for every source hand s, and the target hand is commanded X_t = EE G_t(q_t)^-1 = X delta(z). The observed wrist goes
    the other way (X_state = X_t,observed delta(z_obs)^-1) so the policy sees the wrist its training hands would have.
    """
    def __init__(self, cfg, tr, target, fixes=None):
        self.fw = FWrist(cfg["xwo"], cfg["spec"]); self.tr, self.t, self.src, self.fixes = tr, target, list(cfg["sources"]), fixes or {}
        self.B, self.B_fit_mm = {}, {}
        for h in self.src + [target]:           # palm frame of the offset model in our canonical frame, fitted on fingertips
            m = tr.hand_models[h]; H = self.fw.hand(h); torch.manual_seed(0)
            qn = torch.rand(128, len(m.dof_joints)) * 2 - 1
            with torch.no_grad():
                ours = m._forward_internal(qn)[0].double().numpy()                         # [N, F, 3] canonical frame
            ang = ((qn + 1) * 0.5 * (m._upper - m._lower) + m._lower).double().numpy(); nm = [n.replace(".", "_") for n in m.dof_joints]
            x = np.zeros((128, H.dof), np.float32)
            for n in H.jn:
                x[:, H.fk.qidx[n]] = ang[:, nm.index(n)]
            theirs = np.asarray(H.tips(x), np.float64)[:, [f for f in range(5) if H.valid[f]]]  # [N, F, 3] their palm frame
            A, Bp = theirs.reshape(-1, 3), ours.reshape(-1, 3); ca, cb = A.mean(0), Bp.mean(0)
            U, _, Vt = np.linalg.svd((A - ca).T @ (Bp - cb)); R = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = cb - R @ ca; self.B[h] = T
            self.B_fit_mm[h] = float(np.linalg.norm(A @ R.T + T[:3, 3] - Bp, axis=1).max() * 1e3)
            assert self.B_fit_mm[h] < 1.0, (h, self.B_fit_mm[h])

    def angles(self, h, z):                    # latent [1, d] torch -> joint names, angles [1, n] (rad)
        from dexverse_xlvla.eval_latent import dec
        m = self.tr.hand_models[h]
        with torch.no_grad():
            qn = dec(self.tr, h, z)
            if h in self.fixes:
                qn = qn + self.fixes[h](z)
            qn = qn.clamp(-1, 1)
        return list(m.dof_joints), ((qn + 1) * 0.5 * (m._upper - m._lower) + m._lower).double().numpy()

    def G(self, h, names, ang):                # [N, n] rad -> [N, 4, 4]
        return self.B[h] @ np.linalg.inv(self.fw.off(h, names, ang))

    def delta_batch(self, z, t_names, t_ang):  # z [N, d], t_ang [N, n] -> [N, 4, 4];  X_target = X_policy @ delta
        if self.fw.kind != "f":                # the offset does not depend on the finger pose: one constant
            if getattr(self, "_const", None) is None:
                self._const = _se3_mean([self.B[s] @ np.linalg.inv(self.fw.off(s, [], np.zeros((1, 0)))[0]) for s in self.src]) @ \
                    np.linalg.inv(self.B[self.t] @ np.linalg.inv(self.fw.off(self.t, [], np.zeros((1, 0)))[0]))
            return np.tile(self._const, (len(z), 1, 1))
        Gs = np.stack([self.G(s, *self.angles(s, z)) for s in self.src]); Gt = self.G(self.t, t_names, t_ang)
        return np.stack([_se3_mean(Gs[:, i]) @ np.linalg.inv(Gt[i]) for i in range(len(z))])

    def delta(self, z, t_names, t_ang):
        return self.delta_batch(z, t_names, t_ang)[0]
