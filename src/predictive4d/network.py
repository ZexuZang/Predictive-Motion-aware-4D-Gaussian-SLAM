"""One genuinely shared trainable spatiotemporal encoder, two neural heads.

This is a compact GenFlow3D-STYLE PointNet/GRU architecture, not upstream
GenFlow3D's pretrained BDCB/STIB/diffusion network. See README for adapter status.
"""
from __future__ import annotations
import torch
from torch import nn

class JointMotionNet(nn.Module):
    def __init__(self,hidden=128):
        super().__init__()
        self.hidden=int(hidden)
        # The EXACT SAME encoder output feeds both heads (and can be trained jointly).
        self.encoder=nn.Sequential(nn.Linear(7,64),nn.ReLU(),nn.Linear(64,self.hidden),nn.ReLU(),
                                   nn.Linear(self.hidden,self.hidden),nn.ReLU())
        self.temporal=nn.GRU(self.hidden+6+1,self.hidden,batch_first=True)
        self.camera_head=nn.Sequential(nn.Linear(self.hidden,128),nn.ReLU(),nn.Linear(128,12))
        self.scene_head=nn.Sequential(nn.Linear(self.hidden*2+3,128),nn.ReLU(),nn.Linear(128,6))
        self.dynamic_head=nn.Sequential(nn.Linear(self.hidden*2+3,64),nn.ReLU(),nn.Linear(64,1))

    def forward(self,points,twists,dts,velocity_hint):
        """points [B,L,N,7]: world XYZ, RGB, valid/geom confidence.
        twists [B,L,6]: past estimated left SE3 increments; dts [B,L,1].
        velocity_hint [B,N,3]: causal NN world displacement of latest 2 clouds.
        All outputs refer to the NEXT timestamp, never use next RGB-D as input.
        """
        if points.ndim!=4 or points.shape[-1]!=7:raise ValueError("points must [B,L,N,7]")
        B,L,N,_=points.shape
        if twists.shape!=(B,L,6) or dts.shape!=(B,L,1) or velocity_hint.shape!=(B,N,3):
            raise ValueError("twist/dt/hint dimensions inconsistent")
        emb=self.encoder(points)                    # [B,L,N,H], SHARED
        pooled=emb.mean(dim=2)                    # [B,L,H]
        encoded,_=self.temporal(torch.cat([pooled,twists,dts],-1))
        global_feature=encoded[:,-1]              # [B,H]
        camera=self.camera_head(global_feature)   # [B,12]
        feat=torch.cat([emb[:,-1],global_feature[:,None].expand(-1,N,-1),velocity_hint],-1)
        scene=self.scene_head(feat)
        # Flow expressed as per-frame WORLD displacement in meters.
        return {"pose_xi":camera[:,:6],"pose_logvar":camera[:,6:].clamp(-10,3),
                "flow":scene[:,:,:3],"flow_logvar":scene[:,:,3:].clamp(-10,3),
                "dynamic_logits":self.dynamic_head(feat).squeeze(-1)}


def load_model(checkpoint,device="cpu",config=None):
    """Strictly load a trained checkpoint. Never silently run random weights."""
    import pathlib
    path=pathlib.Path(checkpoint)
    if not path.is_file():raise FileNotFoundError(f"Neural checkpoint missing: {path}")
    # weights_only prevents unrestricted pickle loading of untrusted checkpoints.
    obj=torch.load(str(path),map_location=device,weights_only=True)
    if not obj.get("trained",False):
        raise RuntimeError("Checkpoint is not marked as trained")
    if obj.get('arch')=='JointMotionNet-v1':
        net=JointMotionNet(int(obj["hidden"])).to(device)
        net.load_state_dict(obj["model"],strict=True)
    elif obj.get('arch')=='OfficialSharedGenFlow-v1':
        from .genflow_shared import create_official_shared
        cfg=config or {}
        net=create_official_shared(cfg['genflow_repo'],cfg['genflow_checkpoint'],
                int(obj['n']),device,hidden=int(obj['hidden']))
        net.load_head_state_dict(obj['head_model'])
    else:
        raise RuntimeError(f"Unknown neural architecture: {obj.get('arch')}")
    net.eval()
    return net,obj
