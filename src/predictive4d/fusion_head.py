"""Confidence-conditioned SE(3) fusion; GT is NEVER a runtime input.

Research prototype: a scalar learned interpolation weight for two W2C poses.
The geometric proposal is already conditioned on the neural prior in joint_ba;
treat this as adaptive *proposal interpolation*, not independent information
fusion or a calibrated probabilistic estimate.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from .geometry import invert, se3_exp, se3_log, is_pose

FEATURE_NAMES = (
    'match_fraction', 'log_matches', 'static_mass_fraction',
    'dynamic_mass_fraction', 'outlier_mass_fraction',
    'log_static_eigenvalue', 'log_hessian_condition', 'geometry_valid',
    'neural_pose_valid', 'neural_translation_m', 'neural_rotation_deg',
    'median_flow_m', 'p90_flow_m', 'mean_dynamic_prior',
    'pose_disagreement_m', 'pose_disagreement_deg',
    'valid_depth_ratio', 'geometric_residual_m',
)


def reliability_features(T_pred, T_geo, track_diag, neural_diag, flow, p_dynamic,
                         n_observations, valid_depth_ratio):
    """Compute features *before* fusion and without using GT or future frames."""
    if not is_pose(T_pred) or not is_pose(T_geo):
        raise ValueError('Fusion inputs must be valid W2C SE3')
    matched = float(track_diag.get('matches', 0))
    ratio = matched / max(1, int(n_observations))
    static = float(track_diag.get('static_mass', 0.))
    dynamic = float(track_diag.get('dynamic_mass', 0.))
    outlier = float(track_diag.get('outlier_mass', 0.))
    lam = float(track_diag.get('min_static_eigen', 0.))
    condition = float(track_diag.get('h_condition', 1e12))
    move = np.linalg.norm(np.asarray(flow, float), axis=1)
    relative = se3_log(T_geo @ invert(T_pred))
    pred_translation = float(neural_diag.get('raw_pose_translation', 0.))
    pred_rotation = float(neural_diag.get('raw_pose_rotation_deg', 0.))
    vals = np.array([
        ratio, np.log1p(max(matched, 0.)), static / max(1., matched),
        dynamic / max(1., matched), outlier / max(1., matched),
        np.log1p(max(lam, 0.)), np.log10(max(1., min(condition, 1e12))),
        float(track_diag.get('status') == 'joint_ok'),
        float(neural_diag.get('pose_accepted', False)),
        pred_translation, pred_rotation,
        float(np.median(move)) if len(move) else 0.,
        float(np.percentile(move, 90)) if len(move) else 0.,
        float(np.mean(p_dynamic)) if len(p_dynamic) else 0.,
        float(np.linalg.norm(relative[:3])),
        float(np.rad2deg(np.linalg.norm(relative[3:]))),
        float(valid_depth_ratio),
        float(track_diag.get('median_residual_m', 1.)),
    ], dtype=np.float32)
    if vals.shape != (len(FEATURE_NAMES),):
        raise AssertionError('Feature contract changed')
    return np.nan_to_num(vals, nan=0., posinf=12., neginf=-12.)


class ReliabilityFusionHead(nn.Module):
    def __init__(self, input_dim=len(FEATURE_NAMES)):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, features):
        """Returns scalar Geometry weight [B], 0=neural, 1=geometry."""
        return self.network(features).squeeze(-1)


def _twist_hat(xi):
    """Differentiable se(3) twist matrix, translation followed by rotation."""
    v, w = xi[..., :3], xi[..., 3:]
    z = torch.zeros_like(w[..., 0])
    r0 = torch.stack([z, -w[..., 2], w[..., 1], v[..., 0]], -1)
    r1 = torch.stack([w[..., 2], z, -w[..., 0], v[..., 1]], -1)
    r2 = torch.stack([-w[..., 1], w[..., 0], z, v[..., 2]], -1)
    r3 = torch.zeros_like(r0)
    return torch.stack([r0, r1, r2, r3], -2)


def fuse_torch(T_pred, relative_xi, w_geo):
    """Gradient-friendly left-invariant pose interpolation (batch tensors)."""
    return torch.matrix_exp(_twist_hat(relative_xi * w_geo.unsqueeze(-1))) @ T_pred


def fuse_numpy(T_pred, T_geo, weight):
    """Consistent offline/runtime interpolation (no GT required)."""
    w = float(weight)
    if not np.isfinite(w) or not 0 <= w <= 1:
        raise ValueError('Fusion weight must be within [0,1]')
    return se3_exp(w * se3_log(T_geo @ invert(T_pred))) @ T_pred


class FusionRuntime:
    """Frozen, self-contained learned gate + training feature normalization."""
    def __init__(self, checkpoint):
        obj = torch.load(str(checkpoint), map_location='cpu', weights_only=True)
        if obj.get('arch') != 'ReliabilityFusionHead-v1' or tuple(obj['feature_names']) != FEATURE_NAMES:
            raise ValueError('Incompatible or untrained fusion head checkpoint')
        self.geometry_source=obj.get('geometry_source','joint')
        if self.geometry_source not in ('joint','independent'):
            raise ValueError('Invalid fusion geometry source')
        self.model = ReliabilityFusionHead().eval()
        self.model.load_state_dict(obj['model'], strict=True)
        self.mean = np.asarray(obj['feature_mean'], np.float32)
        self.std = np.asarray(obj['feature_std'], np.float32)
        if self.mean.shape != (len(FEATURE_NAMES),) or self.std.shape != self.mean.shape:
            raise ValueError('Incompatible feature normalization')

    def predict_weight(self, features):
        f = np.asarray(features, np.float32)
        if f.shape != self.mean.shape or not np.isfinite(f).all():
            raise ValueError('Invalid fusion features')
        norm = np.clip((f - self.mean) / np.maximum(self.std, 1e-4), -8., 8.)
        with torch.inference_mode():
            return float(self.model(torch.from_numpy(norm[None]))[0].item())
