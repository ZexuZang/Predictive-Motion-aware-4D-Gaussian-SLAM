"""Fine-tune the CAMERA branch on an INDEPENDENT real RGB-D training sequence.

Uses GT poses exclusively in the training sequence to form labels and align
historical input clouds. Do not train on the sequence later used for ATE!
Scene Flow Head and shared encoder are frozen in this fine-tuning stage.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from scipy.spatial import cKDTree
from .data import frame_stream,load_intrinsics,read_gt
from .geometry import unproject,transform,invert,se3_log
from .network import load_model


def train_real_camera(data,intrinsics,checkpoint,out,epochs=15,frames=0,depth_scale=5000.,
                      device='cpu',eval_data=None,lr=3e-4):
    train_root=Path(data).resolve()
    if eval_data and train_root==Path(eval_data).resolve():
        raise ValueError('Train and test sequence must differ!')
    K=load_intrinsics(intrinsics)
    net,meta=load_model(checkpoint,device)
    if meta['arch']!='JointMotionNet-v1':
        raise ValueError('train-real-camera currently supports lightweight joint network')
    L=int(meta['history']);n=int(meta['n'])
    gt=read_gt(train_root/'groundtruth.txt')
    if len(gt)<L+1:raise ValueError('Insufficient training GT poses')
    gt_ts=np.asarray([t for t,_ in gt]);sequence=[]
    for i,frame in enumerate(frame_stream(train_root,max_frames=frames,depth_scale=depth_scale)):
        j=int(np.argmin(np.abs(gt_ts-frame.timestamp)))
        if abs(gt_ts[j]-frame.timestamp)>.02:continue
        xyz,rgb,_=unproject(frame,K,max_points=max(n,300),pixel_stride=2,seed=i+500)
        if len(xyz)<n//2:continue
        ix=np.linspace(0,len(xyz)-1,n,dtype=int)
        Twc=gt[j][1];Tcw=invert(Twc)
        world=transform(Twc,xyz[ix]);features=np.column_stack([world,rgb[ix],np.ones(n)]).astype(np.float32)
        sequence.append((frame.timestamp,features,Tcw))
    if len(sequence)<L+1:raise ValueError('Too few GT-aligned training frames')
    # Explicitly freeze shared encoder and Flow/Dynamic heads in real pose fine-tune;
    # synthetic Scene Flow weights remain unchanged. Camera head learns better pose.
    for p in net.parameters():p.requires_grad_(False)
    for part in (net.temporal,net.camera_head):
        for p in part.parameters():p.requires_grad_(True)
    optim=torch.optim.AdamW([p for p in net.parameters() if p.requires_grad],lr=lr)
    net.train();windows=0
    for ep in range(epochs):
        errors=[]
        for end in range(L,len(sequence)):
            hist=sequence[end-L:end];target=sequence[end]
            x=np.stack([f[1] for f in hist]);tw=np.zeros((L,6),np.float32)
            dts=np.full((L,1),.1,np.float32)
            for k in range(1,L):
                tw[k]=se3_log(hist[k][2]@invert(hist[k-1][2])).astype(np.float32)
                dts[k,0]=max(.001,hist[k][0]-hist[k-1][0])
            _,j=cKDTree(x[-2,:,:3]).query(x[-1,:,:3])
            hint=np.clip(x[-1,:,:3]-x[-2,j,:3],-.5,.5).astype(np.float32)
            label=se3_log(target[2]@invert(hist[-1][2])).astype(np.float32)
            inp=lambda arr:torch.from_numpy(arr[None]).to(device)
            output=net(inp(x),inp(tw),inp(dts),inp(hint))
            # Balance translation/rotation separately; no real scene flow labels.
            truth=torch.from_numpy(label[None]).to(device)
            loss=((output['pose_xi'][:,:3]-truth[:,:3]).abs().mean() +
                  (output['pose_xi'][:,3:]-truth[:,3:]).abs().mean())
            optim.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad],1.)
            optim.step();errors.append(float(loss.item()));windows+=1
        print(f'epoch {ep+1}/{epochs} camera_L1={np.mean(errors):.5f}',flush=True)
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'arch':'JointMotionNet-v1','hidden':meta['hidden'],'trained':True,
        'history':L,'n':n,'train_source':'synthetic_plus_independent_real_camera',
        'real_training_sequence':str(train_root),'camera_training_windows':windows,
        'model':net.cpu().state_dict()},path)
    return path
