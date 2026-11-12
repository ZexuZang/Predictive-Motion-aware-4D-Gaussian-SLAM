"""Regressions for the late-sequence CV-only non-SE(3) camera crash."""
from types import SimpleNamespace
import numpy as np
import pytest
from predictive4d.native_pose_compat import native_camera_pose


def cam(R, t=(0., 0., 0.)):
    return SimpleNamespace(R=np.asarray(R, dtype=float), T=np.asarray(t, dtype=float))


def test_clean_rotation_remains_equal():
    T = native_camera_pose(cam(np.eye(3), (1, 2, 3)))
    assert np.allclose(T[:3, 3], [1, 2, 3])
    assert np.allclose(T[:3, :3], np.eye(3))


def test_small_numeric_drift_reorthogonalized_not_gt():
    R = np.eye(3); R[0, 1] = 0.0002
    T = native_camera_pose(cam(R))
    assert np.linalg.norm(T[:3, :3].T @ T[:3, :3] - np.eye(3)) < 1e-10


@pytest.mark.parametrize('bad', [np.diag([-1, 1, 1]),
                                 np.diag([2, 1, 1]),
                                 np.full((3, 3), float('nan'))])
def test_nonrigid_or_nonfinite_fails_closed(bad):
    with pytest.raises(ValueError):
        native_camera_pose(cam(bad))
