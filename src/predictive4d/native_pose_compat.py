"""Guarded conversion of native Flow4DGS camera poses into true SE(3).

Native front-end uses floating-point camera tensors which can accumulate tiny
rotation-matrix drift. Project ONLY small numerical drift onto SO(3); reject
nonfinite, reflected, degenerate or substantially nonrigid matrices. Never use GT.
"""
from __future__ import annotations
import numpy as np
import torch
from .geometry import is_pose


def _numpy(a):
    return a.detach().cpu().numpy() if isinstance(a, torch.Tensor) else np.asarray(a)


def native_camera_pose(camera, max_orthogonality_error=0.02, max_projection_error=0.02):
    R = np.asarray(_numpy(camera.R), dtype=np.float64)
    t = np.asarray(_numpy(camera.T), dtype=np.float64).reshape(-1)
    if R.shape != (3, 3) or t.shape != (3,) or not np.isfinite(R).all() or not np.isfinite(t).all():
        raise ValueError('Native camera pose has invalid shape or nonfinite values')
    orth_error = float(np.linalg.norm(R.T @ R - np.eye(3), ord='fro'))
    det = float(np.linalg.det(R))
    if not np.isfinite(det) or det <= 0 or orth_error > max_orthogonality_error:
        raise ValueError(f'Native pose is not close to SO(3): orth_error={orth_error:.6g} det={det:.6g}')
    U, _, Vt = np.linalg.svd(R)
    R_so3 = U @ Vt
    if float(np.linalg.det(R_so3)) < 0:
        raise ValueError('Native rotation SVD produced a reflection')
    correction = float(np.linalg.norm(R_so3 - R, ord='fro'))
    if correction > max_projection_error:
        raise ValueError(f'Unsafe SO(3) projection: correction={correction:.6g}')
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R_so3
    T[:3, 3] = t
    if not is_pose(T):
        raise ValueError('Projected pose failed SE(3) validation')
    return T
