"""Finger-finger collision projection of a retargeted hand pose (wrapper side; the teleop retargeter is unchanged).

DexPilot (teleop retargeter) has no collision term, so a robot hand with thicker / closer fingers than the human one can
put fingers through each other. Per frame, the solved dex_q is moved to the closest joint configuration (least squares
in the retargeter's optimized joints; mimic joints follow through the retargeter's own kinematic adaptor) in which no
finger-finger pair of the retarget URDF's collision hulls (self_collision.HandCollision: same geometry, same pairs, same
open-pose exclusions as the quality metric) penetrates deeper than ``tol`` . Touching stays allowed (pinch contacts).

Linearised iterations: contacts (point p, normal n from geometry a to b, depth d) give the constraint
n . (J_b(p) - J_a(p)) dx >= d + margin; a small QP (SLSQP, joint limits, trust region |dx| <= step_max per joint) gives
dx; repeat until clear. The trust region matters: on deep overlaps (> 1 cm) the linearisation is poor and an unbounded
step swung single joints by ~90-115 deg (2026-11-11). Objective = |x - x0|^2 + smooth * |(x - x0) - offset_prev|^2: the
correction of the previous frame is kept as a prior so it does not flip between solutions frame to frame (reset() per
trajectory). EPA failures on deep
overlaps (no finite depth / normal) use the direction between the two geometry centres and a fixed push.
The retargeter's warm start keeps its own (raw) solution, so the projection never feeds back into DexPilot.
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
from scipy.optimize import minimize

from cross_dex.dataset.chunk_paired.self_collision import HandCollision


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class CollisionProjector(HandCollision):
    def __init__(self, urdf: str, dex_joint_names: list[str], target_idx=None, mimic_fn=None, lo=None, hi=None,
                 tol_mm: float = 1.0, margin_mm: float = 0.3, max_iter: int = 60, deep_push_mm: float = 3.0,
                 step_max: float = 0.05, smooth: float = 1.0, rate: float = np.deg2rad(5.0)):
        """target_idx: dex_q indices the retargeter optimizes (others stay fixed); mimic_fn: dex_q -> dex_q applying the
        retargeter's mimic adaptor (None = no mimic joints); lo/hi: dex_q joint limits of the retargeter."""
        super().__init__(urdf, dex_joint_names)
        self.enable_contacts()
        self.req_contacts(4)
        n = len(dex_joint_names)
        self.target_idx = np.arange(n) if target_idx is None else np.asarray(target_idx)
        self.mimic_fn = mimic_fn
        self.lo = np.full(n, -np.inf) if lo is None else np.asarray(lo, float)
        self.hi = np.full(n, np.inf) if hi is None else np.asarray(hi, float)
        self.v_index = [self.model.joints[self.model.getJointId(nm)].idx_v for nm in dex_joint_names]
        self.tol, self.margin, self.max_iter, self.deep = tol_mm / 1e3, margin_mm / 1e3, max_iter, deep_push_mm / 1e3
        self.step_max, self.smooth, self.rate = step_max, smooth, rate
        self.reset()
        # d dex_q / d x (x = optimized joints); linear (mimic: q_m = mult * q_s + off), measured once
        base = self.full(np.zeros(len(self.target_idx)), np.zeros(n))
        self.M = np.stack([self.full(np.eye(len(self.target_idx))[i], np.zeros(n)) - base
                           for i in range(len(self.target_idx))], axis=1)

    def reset(self):
        self.offset = np.zeros(len(self.target_idx))

    def req_contacts(self, k: int):
        for req in self.gdata.collisionRequests:
            req.num_max_contacts = k

    def full(self, x: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
        q = q_ref.copy()
        q[self.target_idx] = x
        return self.mimic_fn(q) if self.mimic_fn is not None else q

    def _contacts(self, dex_q: np.ndarray):
        """[(geom a, geom b, point, normal a->b, depth)] of active pairs deeper than 0, model data updated at dex_q."""
        q = pin.neutral(self.model)
        q[self.q_index] = dex_q
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateGeometryPlacements(self.model, self.data, self.geom, self.gdata)
        pin.computeJointJacobians(self.model, self.data, q)
        pin.computeCollisions(self.geom, self.gdata, False)
        out = []
        for i, res in enumerate(self.gdata.collisionResults):
            if not (self.active[i] and res.isCollision()):
                continue
            a, b = self.pairs[i]
            ca, cb = self.gdata.oMg[a].translation, self.gdata.oMg[b].translation
            for k in range(res.numContacts()):
                c = res.getContact(k)
                d, nrm, p = abs(c.penetration_depth), np.asarray(c.normal, float), np.asarray(c.pos, float)
                if not (np.isfinite(d) and d < 0.05 and np.all(np.isfinite(nrm)) and np.all(np.isfinite(p))
                        and np.linalg.norm(nrm) > 0.5):
                    nrm, d, p = cb - ca, self.deep, 0.5 * (ca + cb)        # EPA failure: push the centres apart
                    nrm = nrm / max(np.linalg.norm(nrm), 1e-9)
                    d = min(d, 0.05)
                out.append((a, b, p, nrm / np.linalg.norm(nrm), min(d, 0.05)))
        return out

    def _point_jac(self, g: int, p: np.ndarray) -> np.ndarray:
        jid = self.geom.geometryObjects[g].parentJoint
        J = pin.getJointJacobian(self.model, self.data, jid, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        Jp = J[:3] - _skew(p - self.data.oMi[jid].translation) @ J[3:]
        return Jp[:, self.v_index] @ self.M          # [3, n_x]

    def depth(self, dex_q: np.ndarray) -> float:
        c = self._contacts(dex_q)
        return max((x[4] for x in c), default=0.0)

    def _iterate(self, x: np.ndarray, x0: np.ndarray, q0: np.ndarray, lo, hi) -> tuple[np.ndarray, float, int]:
        it, dmax = 0, 0.0
        for it in range(self.max_iter + 1):
            cs = self._contacts(self.full(x, q0))
            dmax = max((c[4] for c in cs), default=0.0)
            if dmax <= self.tol or it == self.max_iter:
                break
            A = np.stack([n @ (self._point_jac(b, p) - self._point_jac(a, p)) for a, b, p, n, d in cs])
            rhs = np.array([d + self.margin for *_, d in cs])
            cons = [{"type": "ineq", "fun": lambda dx, A=A, r=rhs: A @ dx - r, "jac": lambda dx, A=A: A}]
            w, off = self.smooth, self.offset
            res = minimize(lambda dx: float(np.sum((x + dx - x0) ** 2) + w * np.sum((x + dx - x0 - off) ** 2)),
                           np.zeros_like(x), jac=lambda dx: 2 * (x + dx - x0) + 2 * w * (x + dx - x0 - off),
                           constraints=cons, method="SLSQP", options={"maxiter": 50},
                           bounds=list(zip(np.maximum(lo - x, -self.step_max), np.minimum(hi - x, self.step_max))))
            x = np.clip(x + np.clip(res.x, -self.step_max, self.step_max), lo, hi)
        return x, dmax, it

    def project(self, dex_q: np.ndarray) -> tuple[np.ndarray, dict]:
        """Closest collision-free dex_q, continuous over a trajectory (call reset() per trajectory).

        The correction (x - x0) may change by at most ``rate`` rad per joint between frames (every path below is
        boxed to previous correction +- rate). Where that is not enough to clear a deep overlap, the pose stays
        penetrating (the quality metric then drops that chunk) instead of jumping to another solution: unbounded
        re-solves flipped fingers to the other side, up to 59 deg in one frame (sharpa, Feix 1, 2026-11-11).
        1) raw pose clear -> move back towards it. 2) else start from the raw pose + the previous correction (warm
        start); if that is clear, shrink the correction towards the raw pose as far as it stays clear (bisection).
        3) else linearised trust-region iterations from there.
        Returns (dex_q, info: depth_before/after (m), iters, moved (rad, max abs over joints))."""
        q0 = np.asarray(dex_q, float)
        x0 = q0[self.target_idx].copy()
        lo, hi = self.lo[self.target_idx], self.hi[self.target_idx]
        prev = np.clip(x0 + self.offset, lo, hi)
        blo, bhi = np.maximum(lo, prev - self.rate), np.minimum(hi, prev + self.rate)
        box = lambda v: np.clip(v, blo, bhi)   # noqa: E731
        d0 = self.depth(self.full(x0, q0))
        it = 0
        if d0 <= self.tol:
            x = box(x0)
            dmax = d0 if np.allclose(x, x0) else self.depth(self.full(x, q0))
        else:
            xs = box(x0 + self.offset)
            ds = self.depth(self.full(xs, q0)) if np.any(self.offset) else d0
            if np.any(self.offset) and ds <= self.tol:
                a, b = 0.0, 1.0                      # depth(x0 + a*off) > tol, depth(x0 + b*off) <= tol
                for _ in range(6):
                    m = 0.5 * (a + b)
                    if self.depth(self.full(box(x0 + m * self.offset), q0)) <= self.tol:
                        b = m
                    else:
                        a = m
                x = box(x0 + b * self.offset)
                dmax = self.depth(self.full(x, q0))
            else:
                x, dmax, it = self._iterate(xs, x0, q0, blo, bhi)
        self.offset = x - x0
        q = self.full(x, q0)
        return q, {"depth_before": float(d0), "depth_after": float(dmax), "iters": it,
                   "moved": float(np.abs(q - q0).max())}


class TrajectoryProjector(CollisionProjector):
    """Collision correction of a whole trajectory at once (replaces the per-frame projection for the final data).

    Per-frame projection either leaves deep grasp-closure overlaps unresolved (correction rate-limited) or flips
    fingers between solutions (unlimited). Here the corrections C[t] = x[t] - x0[t] of all frames are one QP (OSQP),
    solved by sequential linearisation:
        min  sum_t |G_t C_t|^2 + eps |C_t|^2 + smooth * sum_t |C_{t+1} - C_t|^2
        s.t. n . (J_b(p) - J_a(p)) (C_t - C_t^k) >= d + margin   for every contact of frame t at the current iterate,
             joint limits, trust region |C_t - C_t^k| <= step_max
    G_t = fingertip position Jacobian at the raw pose (the cost is fingertip displacement, i.e. what the alignment is
    about, plus a small joint-space term). Contacts are re-linearised every outer iteration until no frame is deeper
    than tol (or max_outer)."""

    def __init__(self, urdf: str, dex_joint_names: list[str], tip_links: list[str], *a, eps: float = 1e-4,
                 smooth_w: float = 3e-3, max_outer: int = 40, slack_w: float = 1e4, **kw):
        super().__init__(urdf, dex_joint_names, *a, **kw)
        self.tip_frames = [self.model.getFrameId(n) for n in tip_links]
        assert all(f < self.model.nframes for f in self.tip_frames), f"tip links {tip_links} not in {urdf}"
        self.eps, self.smooth_w, self.max_outer, self.slack_w = eps, smooth_w, max_outer, slack_w

    def _tip_jac(self, dex_q: np.ndarray) -> np.ndarray:
        q = pin.neutral(self.model)
        q[self.q_index] = dex_q
        pin.computeJointJacobians(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        G = [pin.getFrameJacobian(self.model, self.data, f, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[:3] for f in self.tip_frames]
        return np.concatenate(G)[:, self.v_index] @ self.M          # [15, n_x]

    def project_traj(self, Q: np.ndarray) -> tuple[np.ndarray, dict]:
        import osqp
        import scipy.sparse as sp
        Q = np.asarray(Q, float)
        T, n = len(Q), len(self.target_idx)
        X0 = Q[:, self.target_idx]
        lo, hi = self.lo[self.target_idx], self.hi[self.target_idx]
        d_before = np.array([self.depth(q) for q in Q])
        C = np.zeros((T, n))
        if d_before.max() <= self.tol:
            return Q.copy(), {"depth_before": d_before, "depth_after": d_before, "outer": 0}
        # cost: block-diag(G^T G + eps) + smoothness (first differences)
        blocks = []
        for t in range(T):
            G = self._tip_jac(Q[t])
            blocks.append(G.T @ G + self.eps * np.eye(n))
        D = sp.diags([np.ones(T - 1), -np.ones(T - 1)], [0, 1], shape=(T - 1, T))
        P = sp.block_diag(blocks) + self.smooth_w * sp.kron(D.T @ D, sp.eye(n))
        P = P.tocsc()
        d_after, k = d_before, 0
        for k in range(self.max_outer):
            rows, lb = [], []
            d_now = np.zeros(T)
            for t in range(T):
                cs = self._contacts(self.full(X0[t] + C[t], Q[t]))
                for a, b, p, nrm, d in cs:
                    d_now[t] = max(d_now[t], d)
                    rows.append((t, nrm @ (self._point_jac(b, p) - self._point_jac(a, p))))
                    lb.append(d + self.margin)
            d_after = d_now
            if d_now.max() <= self.tol:
                break
            m = len(rows)
            N = T * n
            # variables [C (N), s (m)]: contact rows are soft (A_t C_t + s >= rhs, s >= 0, cost slack_w * s^2) so a
            # deep overlap that one trust-region step cannot clear still gives a feasible step towards it
            Ac = sp.lil_matrix((m, N + m))
            for i, (t, r) in enumerate(rows):
                Ac[i, t * n:(t + 1) * n] = r
                Ac[i, N + i] = 1.0
            lb_c = np.array(lb) + np.array([r @ C[t] for t, r in rows])
            # box: joint limits and trust region around the current iterate; slack >= 0
            box_lo = np.concatenate([np.maximum(lo - X0, C - self.step_max).ravel(), np.zeros(m)])
            box_hi = np.concatenate([np.minimum(hi - X0, C + self.step_max).ravel(), np.full(m, np.inf)])
            A = sp.vstack([Ac.tocsc(), sp.eye(N + m, format="csc")], format="csc")
            l = np.concatenate([lb_c, box_lo])
            u = np.concatenate([np.full(m, np.inf), box_hi])
            Pk = sp.block_diag([P, self.slack_w * sp.eye(m)], format="csc")
            solver = osqp.OSQP()
            solver.setup(sp.triu(Pk, format="csc"), np.zeros(N + m), A, l, u, verbose=False, polish=False,
                         max_iter=8000, eps_abs=1e-7, eps_rel=1e-7)
            solver.warm_start(x=np.concatenate([C.ravel(), np.zeros(m)]))
            res = solver.solve()
            try:
                xs = np.asarray(res.x, float)
            except (TypeError, ValueError):     # infeasible linearisation: osqp returns None entries
                xs = None
            if xs is None or xs.shape != (T * n + m,) or not np.all(np.isfinite(xs)):
                self.last_status = str(res.info.status)
                break
            C = np.clip(xs[:N].reshape(T, n), C - self.step_max, C + self.step_max)
        else:
            d_after = np.array([self.depth(self.full(X0[t] + C[t], Q[t])) for t in range(T)])
        out = np.stack([self.full(X0[t] + C[t], Q[t]) for t in range(T)])
        return out, {"depth_before": d_before, "depth_after": d_after, "outer": k}
