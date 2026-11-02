"""Motion prediction backends.

* history_velocity: explicitly non-learned baseline, never calls GenFlow3D.
* genflow_official: optional adapter around official GenFlow3D source/weights.
No checkpoint is bundled. Predictions are *one timestep* displacements, in metres.
"""
from __future__ import annotations
from collections import deque
from pathlib import Path
import importlib
import sys
import numpy as np
from scipy.spatial import cKDTree


class HistoryVelocity:
    """Simple *causal* nearest-neighbour velocity propagation baseline."""
    def __init__(self, default_var=.1**2):
        self.last_points=None;self.last_flows=None;self.default_var=default_var
    def predict(self, history_points, query_points):
        if self.last_points is None or self.last_flows is None or len(self.last_points)==0:
            return np.zeros_like(query_points),np.repeat(np.eye(3)[None]*self.default_var,len(query_points),axis=0)
        dist,idx=cKDTree(self.last_points).query(query_points)
        flows=self.last_flows[idx].copy(); flows[dist>.15]=0.
        cov=np.repeat(np.eye(3)[None]*self.default_var,len(query_points),axis=0)
        cov[dist>.15]*=10
        return flows,cov
    def update(self,points,flows):
        self.last_points=np.asarray(points,float).copy()
        self.last_flows=np.asarray(flows,float).copy()


class OfficialGenFlow3D:
    """Wrap official models.GenFlow_rec_seq; NOT its training dataset loader.

    The original architecture uses geometric features == point xyz, and requires
    `pointnet2` custom CUDA compiled within its own environment. No checkpoint
    adaptation is performed. World vs sensor coordinate consistency, scale,
    indoor domain shift and temporal resampling must be validated by the user.
    """
    def __init__(self,repo,checkpoint,npoint=8192,length_pre=3,variance=.15**2,device='cuda'):
        import torch
        if not torch.cuda.is_available() and device.startswith('cuda'):
            raise RuntimeError('GenFlow3D requires CUDA with built PointNet2 extension')
        repo=Path(repo).resolve(); checkpoint=Path(checkpoint).resolve()
        if not (repo/'models.py').exists():raise FileNotFoundError(f'Official GenFlow3D models.py absent: {repo}')
        if not checkpoint.exists():raise FileNotFoundError(checkpoint)
        sys.path.insert(0,str(repo))
        try:
            models=importlib.import_module('models')
        except Exception as e:
            raise RuntimeError('Import official GenFlow3D failed; build pointnet2 and check CUDA env') from e
        self.model=models.GenFlow_rec_seq(npoint).to(device)
        # Only load checkpoints from trusted sources; unsafe pickle deserialization.
        ck=torch.load(str(checkpoint),map_location=device,weights_only=False)
        if isinstance(ck,dict) and 'state_dict' in ck:ck=ck['state_dict']
        if isinstance(ck,dict) and any(k.startswith('module.') for k in ck):
            ck={k.removeprefix('module.'):v for k,v in ck.items()}
        self.model.load_state_dict(ck,strict=True);self.model.eval()
        self.torch=torch;self.device=device;self.npoint=npoint;self.length_pre=length_pre
        self.var=variance;self.rng=np.random.default_rng(31)
        self.latest=None
    def _sample(self,pts):
        if not len(pts):raise ValueError('Empty point cloud')
        ids=self.rng.choice(len(pts),self.npoint,replace=len(pts)<self.npoint)
        return pts[ids].astype('float32')
    def predict(self, history_points, query_points):
        import torch
        if len(history_points)<self.length_pre:
            return np.zeros_like(query_points),np.repeat(np.eye(3)[None]*(self.var*25),len(query_points),axis=0)
        pcs=[self._sample(p) for p in history_points[-self.length_pre:]]
        pc_tensor=[torch.from_numpy(p)[None].to(self.device) for p in pcs]
        feat_tensors=[p.clone() for p in pc_tensor]  # Official dataset features == xyz
        # Original model requires gts list of len (history-1)+future; eval ignores values
        fake_gt=[torch.zeros((1,self.npoint,3),device=self.device) for _ in range(self.length_pre)]
        with torch.inference_mode():
            pred,fps,loss=self.model(pc_tensor,feat_tensors,fake_gt)
        flow=pred[0][-1].detach().cpu().numpy()
        if flow.ndim==3:flow=flow[0]
        if flow.shape==(3,self.npoint):flow=flow.T
        if flow.shape!=(self.npoint,3):raise RuntimeError(f'Unexpected GenFlow output shape {flow.shape}')
        if not np.isfinite(flow).all():raise RuntimeError('GenFlow output nonfinite')
        sampled=pcs[-1]
        _,indices=cKDTree(sampled).query(query_points)
        flows=flow[indices].astype(float)
        # Fixed assumed variance is UNCALIBRATED: do not use for statistical claims.
        cov=np.repeat(np.eye(3)[None]*self.var,len(query_points),axis=0)
        return flows,cov
    def update(self,points,flows):
        pass


def build_motion_backend(cfg):
    if cfg['motion_backend']=='history_velocity':
        return HistoryVelocity(cfg.get('flow_variance',.04))
    if cfg['motion_backend']=='genflow_official':
        return OfficialGenFlow3D(cfg['genflow_repo'],cfg['genflow_checkpoint'],
            npoint=int(cfg.get('genflow_npoint',8192)),length_pre=int(cfg.get('genflow_length_pre',3)),
            variance=float(cfg.get('flow_variance',.04)),device=cfg.get('genflow_device','cuda'))
    raise ValueError('motion_backend must be history_velocity or genflow_official')
