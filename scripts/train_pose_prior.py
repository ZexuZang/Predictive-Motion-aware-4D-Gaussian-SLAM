"""Train OPTIONAL pose GRU with GT from a SEPARATE TRAINING dataset.

Usage example:
PYTHONPATH=src python scripts/train_pose_prior.py --data /path/to/training_data \
  --intrinsics /path/to/training_intrinsics.yaml --gt /path/to/train/groundtruth.txt \
  --out checkpoints/pose_gru_train_only.pt --epochs 10

Do NOT train using evaluation sequence GT and report ATE on the same frames.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from predictive4d.pose_net import PoseGRU
from predictive4d.pose_prior import fixed_sample
from predictive4d.geometry import inverse,se3_log,unproject
from predictive4d.data import frame_stream,read_tum_gt,load_intrinsics


def main():
    import torch
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True);p.add_argument('--intrinsics',required=True)
    p.add_argument('--gt',required=True);p.add_argument('--out',required=True)
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--max-frames',type=int,default=0)
    p.add_argument('--depth-scale',type=float,default=5000)
    args=p.parse_args()
    gt=read_tum_gt(args.gt);ts=np.array([x[0] for x in gt]);frames=list(frame_stream(args.data,args.max_frames,depth_scale=args.depth_scale))
    if len(frames)<5:raise ValueError('Need at least 5 matched frames')
    K=load_intrinsics(args.intrinsics,frames[0].rgb.shape[1],frames[0].rgb.shape[0]);seq=[]
    for f in frames:
        k=int(np.argmin(abs(ts-f.timestamp)))
        if abs(ts[k]-f.timestamp)>.02:continue
        pc,_=unproject(f.depth,K,stride=4,max_points=512)
        seq.append((fixed_sample(pc),inverse(gt[k][1])))
    if len(seq)<5:raise ValueError('Too few temporally aligned GT frames')
    inputs=[];twists=[];targets=[]
    for end in range(3,len(seq)):
        cl=np.stack([seq[j][0] for j in range(end-3,end)]);pose=[seq[j][1] for j in range(end-3,end+1)]
        d=np.zeros((3,6),dtype=np.float32)
        for j in range(1,3):d[j]=se3_log(pose[j]@inverse(pose[j-1]))
        target=se3_log(pose[-1]@inverse(pose[-2])).astype(np.float32)
        inputs.append(cl);twists.append(d);targets.append(target)
    device='cuda' if torch.cuda.is_available() else 'cpu'; model=PoseGRU().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=2e-4)
    P=torch.tensor(np.array(inputs),dtype=torch.float32,device=device)
    D=torch.tensor(np.array(twists),dtype=torch.float32,device=device)
    Y=torch.tensor(np.array(targets),dtype=torch.float32,device=device)
    for epoch in range(args.epochs):
        model.train();mu,lv=model(P,D)
        loss=.5*torch.mean(((Y-mu)**2)*torch.exp(-lv)+lv)
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step();print(f'epoch {epoch+1}/{args.epochs} nll={loss.item():.5f}')
    Path(args.out).parent.mkdir(exist_ok=True,parents=True)
    torch.save({'model':model.cpu().state_dict(),'source':'separate supervised training','frame_count':len(seq)},args.out)
    print('Saved',args.out,'— calibrate covariance on held-out sequences before using for claims.')

if __name__=='__main__':main()
