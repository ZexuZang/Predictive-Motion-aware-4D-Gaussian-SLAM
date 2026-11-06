"""Trains both neural heads with supervised synthetic camera motion and dynamic 3D flow.

Train on separate sequences for serious benchmarks. Synthetic supervision is NOT
proof of domain generalization to TUM/Bonn RGB-D.
"""
from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import torch
from torch.utils.data import Dataset,DataLoader
from torch import nn
from .network import JointMotionNet

class SyntheticSequence(Dataset):
    """Physically meaningful synthetic 3D motion; persistent point identities.

    Camera label = left world->camera relative twist; scene label = WORLD
    per-step point displacement (not optical flow nor camera-space flow).
    """
    def __init__(self,count=512,n=256,length=3,seed=123):
        self.count,self.n,self.length,self.seed=count,n,length,seed
    def __len__(self):return self.count
    def __getitem__(self,index):
        rng=np.random.default_rng(self.seed+index)
        n,L=self.n,self.length
        # 3D structure: stable room, moving clusters, and varying depth.
        xyz=np.column_stack([rng.uniform(-2,2,n),rng.uniform(-1.2,1.2,n),rng.uniform(1.0,5.0,n)])
        dynamic=(rng.random(n)<rng.uniform(.1,.4)).astype(np.float32)
        # Coherent object-wise motion rather than independent point noise.
        dvec=rng.normal(0,.035,3);dvec=np.clip(dvec,-.09,.09)
        acceleration=rng.normal(0,.006,3)
        rgb=rng.uniform(.2,.9,(n,3));rgb[dynamic>0]=np.clip(rgb[dynamic>0]+.1,0,1)
        cams=rng.normal(0,.015,(L+1,6));cams[:,3:]*=.4
        cams[:,0]+=.01
        # Pose history is supervised only in TRAIN; real inference uses past estimates.
        twists=np.zeros((L,6),np.float32);twists[1:]=cams[1:L].astype(np.float32)
        clouds=[]
        for t in range(L):
            disp=t*dvec+.5*t*(t-1)*acceleration
            pos=xyz+dynamic[:,None]*disp
            features=np.column_stack([pos,rgb,np.ones(n)])
            clouds.append(features.astype(np.float32))
        # Actual next world displacement for points at timestep L-1.
        target_flow=dynamic[:,None]*(dvec+(L-1)*acceleration)
        hint=clouds[-1][:,:3]-clouds[-2][:,:3]
        # The next camera increment depends on history but is not trivially exact.
        next_xi=(.85*cams[L-1]+.15*cams[L]).astype(np.float32)
        return (torch.from_numpy(np.stack(clouds)),torch.from_numpy(twists),
                torch.from_numpy(np.full((L,1),.1,np.float32)),
                torch.from_numpy(hint.copy()),torch.from_numpy(next_xi),
                torch.from_numpy(target_flow.astype(np.float32)),
                torch.from_numpy(dynamic))

def train_synthetic(out,steps=250,batch=8,n=256,history=3,hidden=128,lr=8e-4,device="cpu",seed=123):
    torch.manual_seed(seed);np.random.seed(seed)
    trainset=SyntheticSequence(max(steps*batch,256),n,history,seed)
    loader=DataLoader(trainset,batch_size=batch,shuffle=True,num_workers=0)
    model=JointMotionNet(hidden).to(device);opt=torch.optim.AdamW(model.parameters(),lr=lr)
    model.train();last={};updates=0
    for clouds,twists,dts,hint,xi_gt,flow_gt,dyn_gt in loader:
        if updates>=steps:break
        vals=[x.to(device) for x in (clouds,twists,dts,hint,xi_gt,flow_gt,dyn_gt)]
        clouds,twists,dts,hint,xi_gt,flow_gt,dyn_gt=vals
        outpred=model(clouds,twists,dts,hint)
        pose=(outpred['pose_xi']-xi_gt).abs().mean()
        flow=(outpred['flow']-flow_gt).abs().mean()
        dyn=nn.functional.binary_cross_entropy_with_logits(outpred['dynamic_logits'],dyn_gt)
        # learned Gaussian uncertainty proxy, regularized to prevent collapse
        pose_nll=((outpred['pose_xi']-xi_gt).square()*torch.exp(-outpred['pose_logvar'])+
                  outpred['pose_logvar']).mean()
        flow_nll=((outpred['flow']-flow_gt).square()*torch.exp(-outpred['flow_logvar'])+
                  outpred['flow_logvar']).mean()
        loss=3*pose+3*flow+.12*dyn+.01*(pose_nll+flow_nll)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step()
        updates+=1
        last={"loss":float(loss.item()),"pose_l1":float(pose.item()),"flow_l1":float(flow.item()),"dyn_bce":float(dyn.item())}
        if updates%max(1,steps//10)==0 or updates==1:
            print(f"train {updates:4d}/{steps}: "+" ".join(f"{k}={v:.4f}" for k,v in last.items()),flush=True)
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"arch":"JointMotionNet-v1","hidden":hidden,"trained":True,"train_source":"synthetic_only",
                "steps":updates,"history":history,"n":n,"last_loss":last,"model":model.cpu().state_dict()},path)
    return path


def train_official_head_synthetic(out,genflow_repo,genflow_checkpoint,steps=50,batch=1,
                                  n=8192,history=3,device='cuda',seed=123):
    """Train ONLY newly attached Camera/Dynamic heads; GenFlow flow remains pretrained.
    A functional research integration requiring compiled upstream PointNet2/CUDA.
    Synthetic supervision is not sufficient for meaningful indoor benchmarks.
    """
    from .genflow_shared import create_official_shared
    if not torch.cuda.is_available() and str(device).startswith('cuda'):
        raise RuntimeError('Official GenFlow3D requires a working CUDA environment')
    torch.manual_seed(seed)
    model=create_official_shared(genflow_repo,genflow_checkpoint,n,device)
    loader=DataLoader(SyntheticSequence(max(steps*batch,256),n,history,seed),
                      batch_size=batch,shuffle=True,num_workers=0)
    optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-4)
    model.train()
    for i,batch_tensors in enumerate(loader):
        if i>=steps:break
        points,twists,dts,hint,pose_gt,flow_gt,dynamic_gt=[v.to(device) for v in batch_tensors]
        pred=model(points,twists,dts,hint)
        loss=3*(pred['pose_xi']-pose_gt).abs().mean()+.15*nn.functional.binary_cross_entropy_with_logits(
            pred['dynamic_logits'],dynamic_gt)
        optim.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
        optim.step()
        if i%10==0:print(f'official-head step {i+1}/{steps}: {loss.item():.4f}',flush=True)
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'arch':'OfficialSharedGenFlow-v1','trained':True,'hidden':128,
                'history':history,'n':n,'train_source':'synthetic_head_only',
                'genflow_checkpoint_source':str(genflow_checkpoint),
                'head_model':model.head_state_dict()},path)
    print('Saved official shared adapter trained head checkpoint:',path)
    return path
