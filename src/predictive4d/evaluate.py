"""Basic Horn-aligned ATE evaluation; GT only read after SLAM ends."""
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import invert
from .data import read_gt

def write_trajectory(path,timestamps,poses):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        for t,T in zip(timestamps,poses):
            Twc=invert(T);p=Twc[:3,3];q=Rotation.from_matrix(Twc[:3,:3]).as_quat()
            f.write(f'{t:.8f} '+' '.join(f'{v:.9f}' for v in np.r_[p,q])+'\n')

def evaluate_ate(times,poses,gt_path,tol=.02):
    gt=read_gt(gt_path)
    if len(gt)<3:return {'status':'no_valid_GT'}
    ts=np.array([t for t,_ in gt]);paired=[]
    for t,T in zip(times,poses):
        k=int(np.argmin(np.abs(ts-t)))
        if abs(ts[k]-t)<=tol:paired.append((invert(T)[:3,3],gt[k][1][:3,3]))
    if len(paired)<3:return {'status':'too_few_matches','matched':len(paired)}
    a=np.stack([v[0] for v in paired]);b=np.stack([v[1] for v in paired]);am=a.mean(0);bm=b.mean(0)
    U,_,Vt=np.linalg.svd((a-am).T@(b-bm));D=np.eye(3);D[-1,-1]=np.sign(np.linalg.det(Vt.T@U.T))
    R=Vt.T@D@U.T;res=np.linalg.norm((a-am)@R.T+bm-b,axis=1)
    return {'status':'OK','matched':len(paired),'ATE_RMSE_m':float(np.sqrt(np.mean(res**2))),
            'ATE_mean_m':float(res.mean()),'ATE_median_m':float(np.median(res))}
