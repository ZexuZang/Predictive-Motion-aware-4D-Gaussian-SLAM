"""Trajectory I/O and Horn aligned translation ATE (not a complete SLAM benchmark)."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .geometry import inverse
from .data import read_tum_gt


def write_trajectory(path, times,poses_cw):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        for stamp,Tcw in zip(times,poses_cw):
            Twc=inverse(Tcw);p=Twc[:3,3];q=Rotation.from_matrix(Twc[:3,:3]).as_quat()
            f.write(f'{stamp:.8f} '+' '.join(f'{x:.9f}' for x in np.r_[p,q])+'\n')


def evaluate_ate(times,poses_cw,gt_path,match_tolerance=.02):
    gt=read_tum_gt(gt_path)
    if len(gt)<3:return {'status':'GT not available/too short'}
    ts=np.array([v[0] for v in gt]);paired=[]
    for stamp,Tcw in zip(times,poses_cw):
        i=np.argmin(np.abs(ts-stamp))
        if abs(ts[i]-stamp)<=match_tolerance:
            paired.append((inverse(Tcw)[:3,3],gt[i][1][:3,3]))
    if len(paired)<3:return {'status':'Too few GT timestamps matched','matched':len(paired)}
    a=np.array([x[0] for x in paired]);b=np.array([x[1] for x in paired]);ac=a.mean(0);bc=b.mean(0)
    H=(a-ac).T@(b-bc);U,_,Vt=np.linalg.svd(H)
    D=np.eye(3);D[2,2]=np.sign(np.linalg.det(Vt.T@U.T));R=Vt.T@D@U.T
    aligned=(a-ac)@R.T+bc
    e=np.linalg.norm(aligned-b,axis=1)
    return {'status':'OK','matched':len(e),'ATE_RMSE_m':float(np.sqrt(np.mean(e**2))),
            'ATE_mean_m':float(e.mean()),'ATE_median_m':float(np.median(e))}
