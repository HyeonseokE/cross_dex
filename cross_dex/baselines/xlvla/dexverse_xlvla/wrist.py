"""Hand-common wrist representation: world pose of the hand's CANONICAL frame (dexverse_xlvla.hands).

Two spaces per hand, both in the action-term joint order (= LeRobot ``observation.state`` / ``action`` names):
  joints  q: the six virtual wrist joint positions (``observation.state[:6]``),
  actions a: the env's absolute joint-position action terms, q = offset + scale * a (``action[:6]``; DexVerse uses the
             joint default as offset and its own joint order per hand, e.g. Shadow rotation is [z, y, x]).
Joints -> root link pose T(q) = exp(S_1 q_1)...exp(S_6 q_6) M in kinematic-chain order (screws, M and the action map
measured in the simulator by measure_wrist.py) -> canonical frame pose T(q) inv(C), C = root -> canonical transform of
the hand model. The inverse is damped Gauss-Newton, warm-started from the previous step.
Policy-side vector: [x, y, z, r6d_0..r6d_5] (first two columns of the rotation matrix).
"""

from __future__ import annotations

import json

import numpy as np
from scipy.linalg import expm
from scipy.spatial.transform import Rotation


def _se3(w, v):
    m = np.zeros((4, 4))
    m[:3, :3] = [[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]
    m[:3, 3] = v
    return m


def rot6d(r: np.ndarray) -> np.ndarray:
    return np.concatenate([r[..., :, 0], r[..., :, 1]], -1)


def from_rot6d(x: np.ndarray) -> np.ndarray:
    a, b = x[..., :3], x[..., 3:6]
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b - (a * b).sum(-1, keepdims=True) * a
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    return np.stack([a, b, np.cross(a, b)], -1)


class Wrist:
    def __init__(self, calib: dict, canon_root_to_canonical: np.ndarray):
        if "action_joints" not in calib:
            raise ValueError("old wrist calibration without the action map; re-run measure_wrist.py")
        self.S = [_se3(np.asarray(w), np.asarray(v)) for w, v in calib["screws"]]          # chain order
        self.M = np.asarray(calib["M"])
        self.Cinv = np.linalg.inv(np.asarray(canon_root_to_canonical, dtype=np.float64))
        self.perm = [calib["chain_joints"].index(n) for n in calib["action_joints"]]       # action dim -> chain index
        self.scale = np.asarray(calib["action_scale"], dtype=np.float64)
        self.offset = np.asarray(calib["action_offset"], dtype=np.float64)

    def pose(self, q: np.ndarray) -> np.ndarray:
        """Canonical frame pose for joints q (action-term order)."""
        qc = np.zeros(6)
        qc[self.perm] = q
        t = np.eye(4)
        for s, x in zip(self.S, qc):
            t = t @ expm(s * x)
        return t @ self.M @ self.Cinv

    def joints(self, a: np.ndarray) -> np.ndarray:
        return self.offset + self.scale * np.asarray(a, dtype=np.float64)

    def actions(self, q: np.ndarray) -> np.ndarray:
        return (np.asarray(q, dtype=np.float64) - self.offset) / self.scale

    def vec_from_joints(self, q: np.ndarray) -> np.ndarray:
        """(T, 6) wrist joints -> (T, 9) canonical-frame pose vectors."""
        out = []
        for row in np.atleast_2d(q):
            t = self.pose(row)
            out.append(np.concatenate([t[:3, 3], rot6d(t[:3, :3])]))
        return np.asarray(out)

    def vec_from_actions(self, a: np.ndarray) -> np.ndarray:
        return self.vec_from_joints(self.joints(np.atleast_2d(a)))

    def joints_from_vec(self, vec: np.ndarray, q_init: np.ndarray, iters: int = 30) -> np.ndarray:
        """(T, 9) pose vectors -> (T, 6) wrist joints, each step warm-started from the previous solution."""
        q, out = np.asarray(q_init, dtype=np.float64).copy(), []
        for row in np.atleast_2d(vec):
            target = np.eye(4)
            target[:3, :3] = from_rot6d(row[3:9])
            target[:3, 3] = row[:3]
            for _ in range(iters):
                err = self._err(self.pose(q), target)
                if np.linalg.norm(err) < 1e-7:
                    break
                jac = np.stack([(self._err(self.pose(q + d), target) - err) / 1e-5
                                for d in np.eye(6) * 1e-5], 1)
                q = q - np.linalg.solve(jac.T @ jac + 1e-6 * np.eye(6), jac.T @ err)
            out.append(q.copy())
        return np.asarray(out)

    def actions_from_vec(self, vec: np.ndarray, q_init: np.ndarray) -> np.ndarray:
        """(T, 9) pose vectors -> (T, 6) wrist actions; q_init = current wrist joints."""
        return self.actions(self.joints_from_vec(vec, q_init))

    @staticmethod
    def _err(t, target):
        w = Rotation.from_matrix(t[:3, :3] @ target[:3, :3].T).as_rotvec()
        return np.concatenate([t[:3, 3] - target[:3, 3], w])


def load_wrists(calib_json: str, models: dict) -> dict:
    calib = json.load(open(calib_json))
    return {h: Wrist(calib[h], models[h].canon.numpy()) for h in models if h in calib}
