"""Pivot calibration solver: controller -> wrist rotation centre offset (numpy only).

Keep the wrist on a fixed point and rotate the hand: the controller moves on a sphere around the wrist,
so every sample satisfies ``t_i + R_i r = p`` (r: wrist in the controller frame, p: fixed point),
linear in ``(r, p)`` -> least squares with outlier rejection. The rotation between controller and
hand is not observable this way (``calibrate_rotation.py`` measures it).
"""

from __future__ import annotations

import numpy as np

# Quality thresholds for the pivot fit.
MIN_SAMPLES = 100
MIN_ROTATION_SPAN_DEG = 30.0  # smallest principal spread of the rotation axes
GOOD_RMS_MM = 5.0
OUTLIER_FLOOR_M = 0.005  # never reject samples closer than this to the fit


def quat_wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
    """(N, 4) wxyz quaternions -> (N, 3, 3) rotation matrices."""
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=-1),
        ],
        axis=1,
    )


def _solve(t: np.ndarray, rots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(t)
    a = np.concatenate([rots, -np.broadcast_to(np.eye(3), (n, 3, 3))], axis=2).reshape(3 * n, 6)
    x, *_ = np.linalg.lstsq(a, -t.reshape(3 * n), rcond=None)
    return x[:3], x[3:]


def rotation_span_deg(rots: np.ndarray) -> np.ndarray:
    """Principal spreads (deg) of the samples' rotations around their mean. A small smallest value
    means the hand turned about too few axes: the offset along that axis is not observable."""
    ref = rots[len(rots) // 2]
    rel = np.einsum("ij,njk->nik", ref.T, rots)
    angle = np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1.0, 1.0))
    axis = np.stack([rel[:, 2, 1] - rel[:, 1, 2], rel[:, 0, 2] - rel[:, 2, 0], rel[:, 1, 0] - rel[:, 0, 1]], axis=1)
    norm = np.linalg.norm(axis, axis=1, keepdims=True)
    rotvec = np.where(norm > 1e-9, axis / np.maximum(norm, 1e-9), 0.0) * angle[:, None]
    rotvec -= rotvec.mean(axis=0)
    sv = np.linalg.svd(rotvec, compute_uv=False) / np.sqrt(len(rotvec))
    return np.degrees(2 * sv)  # ~ +-1 sigma span


def solve_pivot(poses: np.ndarray, outlier_iters: int = 2) -> dict:
    """Pivot fit for one hand. ``poses``: (N, 7) controller poses ``[x, y, z, qw, qx, qy, qz]``."""
    poses = np.asarray(poses, dtype=np.float64)
    keep = np.isfinite(poses).all(axis=1) & (np.linalg.norm(poses[:, 3:], axis=1) > 0.5)
    keep[1:] &= np.any(poses[1:] != poses[:-1], axis=1)  # drop frozen (bit-identical) frames
    keep &= np.any(poses[:, :3] != 0.0, axis=1)  # drop untracked placeholder poses
    t = poses[keep, :3]
    rots = quat_wxyz_to_rotmat(poses[keep, 3:])
    if len(t) < 10:
        raise ValueError(f"only {len(t)} valid samples")
    used = np.ones(len(t), dtype=bool)
    for _ in range(outlier_iters + 1):
        r, p = _solve(t[used], rots[used])
        residual = np.linalg.norm(t + rots @ r - p, axis=1)
        med = np.median(residual[used])
        mad = np.median(np.abs(residual[used] - med))
        new_used = residual < max(med + 4.0 * 1.4826 * mad, OUTLIER_FLOOR_M)
        if np.array_equal(new_used, used):
            break
        used = new_used
    span = rotation_span_deg(rots[used])
    rms_mm = float(np.sqrt(np.mean(residual[used] ** 2)) * 1000)
    warnings = []
    if used.sum() < MIN_SAMPLES:
        warnings.append(f"few samples ({used.sum()} < {MIN_SAMPLES}): record longer")
    if span.min() < MIN_ROTATION_SPAN_DEG:
        warnings.append(f"rotation too narrow (smallest span {span.min():.0f} deg < {MIN_ROTATION_SPAN_DEG:.0f}):"
                        " also tilt / twist the hand about other axes")
    if rms_mm > GOOD_RMS_MM:
        warnings.append(f"RMS {rms_mm:.1f} mm > {GOOD_RMS_MM} mm: the wrist probably moved, keep it fixed")
    return {
        "offset_ctrl_m": [float(v) for v in r],
        "offset_norm_m": float(np.linalg.norm(r)),
        "pivot_world_m": [float(v) for v in p],
        "rms_mm": rms_mm,
        "max_err_mm": float(residual[used].max() * 1000),
        "n_used": int(used.sum()),
        "n_valid": int(len(t)),
        "rotation_span_deg": [float(v) for v in span],
        "warnings": warnings,
    }
