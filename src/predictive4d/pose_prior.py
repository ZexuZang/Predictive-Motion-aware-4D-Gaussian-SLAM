"""Camera motion priors: causal constant-velocity or trainable PointNet-GRU.

The trainable model is an OPTIONAL experimental camera head. It is not the
original GenFlow3D Encoder and does not share its learned weights. Training
requires pose supervision; no weights are shipped.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from .geometry import predict_pose,se3_exp,se3_log,inverse


def fixed_sample(points,n=256):
    if len(points)==0:return np.zeros((n,3),np.float32)
    ix=np.linspace(0,len(points)-1,n,dtype=int)
    pts=np.asarray(points)[ix].astype(np.float32)
    # Camera coordinates, frame-independent scale, no ground-truth poses.
    return np.clip(pts,-20,20)


class CausalPosePrior:
    def predict(self,poses,history_camera_points):
        return predict_pose(poses),np.eye(6)*.08**2


def build_pose_prior(cfg):
    if cfg.get('pose_backend','constant_velocity')=='constant_velocity':
        return CausalPosePrior()
    if cfg.get('pose_backend')=='pose_gru':
        return LearnedPosePrior(cfg['pose_checkpoint'],device=cfg.get('pose_device','cpu'))
    raise ValueError('pose_backend must be constant_velocity or pose_gru')


class LearnedPosePrior:
    def __init__(self,checkpoint,device='cpu'):
        import torch
        from .pose_net import PoseGRU
        if not Path(checkpoint).is_file():raise FileNotFoundError(checkpoint)
        self.device=device;self.torch=torch
        self.model=PoseGRU().to(device)
        obj=torch.load(checkpoint,map_location=device,weights_only=True)
        self.model.load_state_dict(obj['model'])
        self.model.eval()
    def predict(self,poses,history_camera_points):
        if len(poses)<3 or len(history_camera_points)<3:
            return predict_pose(poses),np.eye(6)*.08**2
        h=list(history_camera_points)[-3:];p=list(poses)[-3:]
        clouds=np.stack([fixed_sample(x) for x in h])[None]
        twists=np.zeros((1,3,6),np.float32)
        for j in range(1,3):twists[0,j]=se3_log(p[j]@inverse(p[j-1]))
        torch=self.torch
        with torch.no_grad():
            mu,logvar=self.model(torch.from_numpy(clouds).to(self.device),
                                 torch.from_numpy(twists).to(self.device))
        mu=mu.cpu().numpy()[0];sig=np.exp(logvar.cpu().numpy()[0]);
        T=se3_exp(mu)@poses[-1]
        return T,np.diag(sig)
