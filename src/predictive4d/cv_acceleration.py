"""Causal CV + learned 6-DoF acceleration camera prior.

Compatible with the CVAccelerationGRU-v1 checkpoint produced by
Colab_Bonn_CV_Gated_Acceleration_Training.ipynb. No GT is read at inference.
The network predicts a left-multiplicative SE(3) CV correction, expressed as
an *effective* six-dimensional acceleration, not a Euclidean world velocity.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch import nn
from .geometry import invert, se3_exp, se3_log, is_pose


class CVAccelerationGRU(nn.Module):
    """Exact training architecture, including normalization buffers."""

    def __init__(self, mean, std, scale, hidden=48):
        super().__init__()
        self.register_buffer('input_mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('input_std', torch.as_tensor(std, dtype=torch.float32))
        self.register_buffer('output_scale', torch.as_tensor(scale, dtype=torch.float32))
        self.gru = nn.GRU(13, int(hidden), batch_first=True)
        self.head = nn.Sequential(nn.Linear(int(hidden), int(hidden)), nn.SiLU(),
                                  nn.Linear(int(hidden), 6))

    def forward(self, motion_features):
        z = ((motion_features - self.input_mean) / self.input_std).clamp(-8, 8)
        encoded, _ = self.gru(z)
        return self.head(encoded[:, -1]) * self.output_scale


def motion_features_from_poses(poses, times):
    """Return [history-1,13] matching notebook's causal training inputs."""
    poses = [np.asarray(T, dtype=np.float64) for T in poses]
    times = np.asarray(times, dtype=np.float64)
    if len(poses) < 4 or len(times) != len(poses):
        raise ValueError('At least four poses with matching timestamps required')
    if not all(is_pose(T) for T in poses) or not np.isfinite(times).all():
        raise ValueError('Invalid historical camera pose or timestamp')
    dts = np.diff(times)
    if np.any(dts <= 1e-5):
        raise ValueError('History timestamps must strictly increase')
    velocities = np.stack([
        se3_log(poses[k] @ invert(poses[k-1])) / dts[k-1]
        for k in range(1, len(poses))
    ])
    historical_a = np.zeros_like(velocities)
    for k in range(1, len(velocities)):
        historical_a[k] = ((velocities[k] - velocities[k-1]) /
                           (0.5 * (dts[k] + dts[k-1])))
    return np.column_stack([velocities, historical_a, dts]).astype(np.float32)


class CVAccelerationRuntime:
    """Predict from completed, estimated poses only; no next-frame images/GT."""

    def __init__(self, checkpoint, device='cpu', linear_threshold=0.0,
                 angular_threshold=1.5, max_dt_ratio=2.0):
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f'CVAccelerationGRU checkpoint missing: {path}')
        obj = torch.load(str(path), map_location='cpu', weights_only=True)
        if obj.get('arch') != 'CVAccelerationGRU-v1' or not obj.get('trained', False):
            raise ValueError('Expected a trained CVAccelerationGRU-v1 checkpoint')
        self.history_poses = int(obj['history_poses'])
        if self.history_poses < 4 or int(obj['feature_dim']) != 13:
            raise ValueError('Unsupported acceleration history or feature dimension')
        weights = obj['model']
        self.net = CVAccelerationGRU(weights['input_mean'], weights['input_std'],
                                     weights['output_scale'], hidden=int(obj['hidden']))
        self.net.load_state_dict(weights, strict=True)
        self.device = torch.device(device)
        self.net.to(self.device).eval()
        self.linear_threshold = float(linear_threshold)
        self.angular_threshold = float(angular_threshold)
        self.max_dt_ratio = float(max_dt_ratio)
        if (self.linear_threshold < 0 or self.angular_threshold < 0 or
                not np.isfinite([self.linear_threshold,self.angular_threshold,
                                 self.max_dt_ratio]).all() or self.max_dt_ratio < 1):
            raise ValueError('Acceleration thresholds must be nonnegative and max_dt_ratio>=1')

    @torch.inference_mode()
    def predict(self, history, dt):
        """Return world->camera prior and JSON-compatible diagnostics."""
        if len(history) < self.history_poses:
            raise ValueError(f'Need {self.history_poses} completed pose observations')
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Invalid future time interval')
        previous = history[-self.history_poses:]
        poses = [p.T_cw for p in previous]
        times = [p.timestamp for p in previous]
        features = motion_features_from_poses(poses, times)
        past_dt = float(times[-1] - times[-2])
        # The original teacher-forced training used a same-step CV baseline.
        # Generalize its CV increment to the actual sensor prediction horizon.
        xi_last = se3_log(poses[-1] @ invert(poses[-2]))
        T_cv = se3_exp(xi_last * (float(dt) / past_dt)) @ poses[-1]
        dt_ratio = float(dt) / past_dt
        if dt_ratio > self.max_dt_ratio:
            return T_cv, {'camera_prior_mode':'cv_gru', 'acceleration_status':'long_gap_cv_only',
                          'dt_s':float(dt), 'dt_ratio':dt_ratio,
                          'linear_gate_active':False,'angular_gate_active':False}
        a = self.net(torch.from_numpy(features[None]).to(self.device))[0].cpu().numpy().astype(float)
        if not np.isfinite(a).all():
            raise FloatingPointError('Nonfinite GRU acceleration output')
        lin = bool(np.linalg.norm(a[:3]) >= self.linear_threshold)
        ang = bool(np.linalg.norm(a[3:]) >= self.angular_threshold)
        a_applied = a.copy()
        if not lin: a_applied[:3] = 0
        if not ang: a_applied[3:] = 0
        T = se3_exp(0.5 * float(dt)**2 * a_applied) @ T_cv
        if not is_pose(T):
            raise FloatingPointError('Non-SE(3) acceleration correction')
        return T, {
            'camera_prior_mode':'cv_gru','acceleration_status':'applied',
            'dt_s':float(dt),'dt_ratio':dt_ratio,
            'linear_gate_active':lin,'angular_gate_active':ang,
            'linear_accel_norm_mps2':float(np.linalg.norm(a[:3])),
            'angular_accel_norm_radps2':float(np.linalg.norm(a[3:])),
            'correction_norm':float(np.linalg.norm(0.5*float(dt)**2*a_applied))
        }
