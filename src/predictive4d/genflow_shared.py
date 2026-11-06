"""Optional ACTUAL official GenFlow3D encoder+decoder with a shared Camera Head.

Requires official upstream repo and CUDA PointNet2 extension, plus authentic
pretrained weights. THIS FILE CANNOT RUN UNTIL THOSE DEPENDENCIES ARE BUILT.
Train Camera+Dynamic heads separately; official network weights are frozen in
first stage. No automatic claim of indoor RGB-D generalization.
"""
from __future__ import annotations
import importlib
import sys
from pathlib import Path
import torch
from torch import nn

class OfficialGenFlowShared(nn.Module):
    """One upstream GenFlow_rec_seq model, one encoder, pretrained flow decoder;
    camera and dynamic-classifier head receive encoder features from same pass.
    """
    def __init__(self,genflow_model,feature_dim=32,hidden=128):
        super().__init__()
        self.genflow=genflow_model
        for param in self.genflow.parameters():param.requires_grad_(False)
        self.genflow.eval()
        self.hidden=hidden
        self.camera_gru=nn.GRU(feature_dim+7,hidden,batch_first=True)
        self.camera_head=nn.Sequential(nn.Linear(hidden,128),nn.ReLU(),nn.Linear(128,12))
        self.dynamic_head=nn.Sequential(nn.Linear(feature_dim+hidden+3,96),nn.ReLU(),nn.Linear(96,1))
        self._features=None
        # The hook observes the ONE encoder already used by the official decoder.
        self._hook=self.genflow.unit.encoder.register_forward_hook(self._capture)

    def _capture(self,module,inputs,output):
        if self._features is None:self._features=output[2][0]

    def forward(self,points,twists,dts,velocity_hint):
        B,L,N,F=points.shape
        if F!=7 or twists.shape!=(B,L,6) or dts.shape!=(B,L,1):
            raise ValueError('Unexpected feature dimensions')
        if velocity_hint.shape!=(B,N,3):raise ValueError('Velocity hint shape invalid')
        self._features=None
        xyz=[points[:,i,:,:3].contiguous() for i in range(L)]
        # Upstream forward expects xyz point features. In inference mode GT values
        # are not used; zeros merely satisfy its historical forward signature.
        dummy=[torch.zeros_like(xyz[0]) for _ in range(L)]
        self.genflow.eval()
        with torch.no_grad():flow_preds,_,_=self.genflow(xyz,xyz,dummy)
        feature=self._features
        self._features=None
        if feature is None or feature.shape[0]!=B*L or feature.shape[-1]!=N:
            raise RuntimeError('Official shared encoder hook returned unexpected shape')
        C=feature.shape[1]
        if self.camera_gru.input_size!=C+7:raise RuntimeError('Upstream encoder channel mismatch')
        f=feature.reshape(L,B,C,N).permute(1,0,3,2) # B,L,N,C
        pooled=f.mean(dim=2)
        temporal,_=self.camera_gru(torch.cat([pooled,twists,dts],dim=-1))
        context=temporal[:,-1]
        camera=self.camera_head(context)
        flow=flow_preds[0][-1]
        if flow.shape==(B,3,N):flow=flow.transpose(1,2)
        if flow.shape!=(B,N,3):raise RuntimeError(f'Official flow layout mismatch {flow.shape}')
        dyn=self.dynamic_head(torch.cat([f[:,-1],context[:,None].expand(-1,N,-1),flow],dim=-1)).squeeze(-1)
        # Fixed variance proxy NOT calibrated from official pretrained model.
        return {'pose_xi':camera[:,:6],'pose_logvar':camera[:,6:].clamp(-10,3),
                'flow':flow,'flow_logvar':torch.full_like(flow,-3.8),
                'dynamic_logits':dyn}

    def train(self,mode=True):
        super().train(mode)
        # Keep pretrained motion decoder frozen + eval even during head training.
        self.genflow.eval()
        return self

    def head_state_dict(self):
        return {name:module.state_dict() for name,module in {
            'camera_gru':self.camera_gru,'camera_head':self.camera_head,
            'dynamic_head':self.dynamic_head}.items()}

    def load_head_state_dict(self,state):
        for name,module in [('camera_gru',self.camera_gru),('camera_head',self.camera_head),('dynamic_head',self.dynamic_head)]:
            module.load_state_dict(state[name],strict=True)


def create_official_shared(repo,checkpoint,npoint,device,hidden=128):
    repo=Path(repo).resolve();checkpoint=Path(checkpoint).resolve()
    if not (repo/'models.py').is_file():raise FileNotFoundError(f'GenFlow3D models.py missing at {repo}')
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    if str(repo) not in sys.path:sys.path.insert(0,str(repo))
    try:mod=importlib.import_module('models')
    except Exception as exc:raise RuntimeError('Official GenFlow3D import failed: build PointNet2 CUDA extension first') from exc
    backbone=mod.GenFlow_rec_seq(npoint).to(device)
    # Upstream pretrained files can contain wrapper dicts. Only load trusted models.
    state=torch.load(checkpoint,map_location=device,weights_only=False)
    if isinstance(state,dict) and 'state_dict' in state:state=state['state_dict']
    if isinstance(state,dict) and any(k.startswith('module.') for k in state):
        state={k.removeprefix('module.'):v for k,v in state.items()}
    backbone.load_state_dict(state,strict=True)
    return OfficialGenFlowShared(backbone,hidden=hidden).to(device)
