"""World/camera SE(3) transforms and guarded rigid tracking."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

def se3_exp(xi):
    xi=np.asarray(xi,float).reshape(6);v=xi[:3];w=xi[3:];th=np.linalg.norm(w)
    W=np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
    if th<1e-7:V=np.eye(3)+.5*W+(W@W)/6
    else:V=np.eye(3)+(1-np.cos(th))/th**2*W+(th-np.sin(th))/th**3*(W@W)
    T=np.eye(4);T[:3,:3]=Rotation.from_rotvec(w).as_matrix();T[:3,3]=V@v
    return T

def se3_log(T):
    w=Rotation.from_matrix(T[:3,:3]).as_rotvec();th=np.linalg.norm(w)
    W=np.array([[0,-w[2],w[1]],[w[2],0,-w[0]],[-w[1],w[0],0]])
    V=np.eye(3)+.5*W+(W@W)/6 if th<1e-7 else np.eye(3)+(1-np.cos(th))/th**2*W+(th-np.sin(th))/th**3*(W@W)
    return np.r_[np.linalg.solve(V,T[:3,3]),w]

def invert(T):
    R=T[:3,:3];T2=np.eye(4);T2[:3,:3]=R.T;T2[:3,3]=-R.T@T[:3,3]
    return T2

def transform(T,points):
    return np.asarray(points)@T[:3,:3].T+T[:3,3]

def is_pose(T):
    return bool(np.asarray(T).shape==(4,4) and np.isfinite(T).all()
        and np.linalg.norm(T[:3,:3].T@T[:3,:3]-np.eye(3))<1e-4
        and abs(np.linalg.det(T[:3,:3])-1)<1e-4)

def motion_size(T_new,T_old):
    rel=T_new@invert(T_old)
    return np.linalg.norm(rel[:3,3]),np.linalg.norm(Rotation.from_matrix(rel[:3,:3]).as_rotvec())

def cap_pose(T_new,T_old,max_translation,max_rotation_deg):
    if not is_pose(T_new):return T_old.copy(),False
    relative=T_new@invert(T_old)
    xi=se3_log(relative);t=np.linalg.norm(xi[:3]);r=np.linalg.norm(xi[3:]);bound=np.deg2rad(max_rotation_deg)
    if t>max_translation or r>bound: return T_old.copy(),False
    return T_new,True

def unproject(frame,K,max_points=1600,pixel_stride=3,seed=42):
    d=frame.depth;h,w=d.shape;ys,xs=np.mgrid[0:h:pixel_stride,0:w:pixel_stride]
    yy,xx=ys.reshape(-1),xs.reshape(-1);z=d[yy,xx]
    ok=np.isfinite(z)&(z>.15)&(z<12)
    yy,xx,z=yy[ok],xx[ok],z[ok]
    if len(z)>max_points:
        ids=np.sort(np.random.default_rng(seed).choice(len(z),max_points,replace=False))
        yy,xx,z=yy[ids],xx[ids],z[ids]
    pts=np.column_stack([(xx-K[0,2])*z/K[0,0],(yy-K[1,2])*z/K[1,1],z]).astype(np.float64)
    rgb=frame.rgb[yy,xx].astype(np.float32)/255.
    mask=frame.dynamic_mask[yy,xx] if frame.dynamic_mask is not None else None
    return pts,rgb,mask

def nearest_unique(src,dst,max_dist=.25):
    if not len(src) or not len(dst):return np.empty(0,int),np.empty(0,int),np.empty(0)
    if not np.isfinite(src).all() or not np.isfinite(dst).all():raise ValueError("Nonfinite correspondence input")
    tree=cKDTree(dst); d,j=tree.query(src,distance_upper_bound=max_dist)
    ok=np.isfinite(d); ordered=np.flatnonzero(ok)[np.argsort(d[ok])]
    used=set();ii=[];jj=[];dd=[]
    for i in ordered:
        if int(j[i]) in used:continue
        used.add(int(j[i]));ii.append(i);jj.append(j[i]);dd.append(d[i])
    return np.asarray(ii,int),np.asarray(jj,int),np.asarray(dd,float)

def rigid_fit(world,observed,weights):
    world=np.asarray(world,float);observed=np.asarray(observed,float);w=np.asarray(weights,float)
    mask=(w>1e-8)&np.isfinite(world).all(axis=1)&np.isfinite(observed).all(axis=1)
    if mask.sum()<6:raise ValueError("Insufficient matches")
    a,b,w=world[mask],observed[mask],w[mask];w=w/w.sum()
    am=(a*w[:,None]).sum(0);bm=(b*w[:,None]).sum(0)
    H=((a-am)*w[:,None]).T@(b-bm);U,s,Vt=np.linalg.svd(H)
    if s[1]<1e-8:raise ValueError("Degenerate geometry")
    D=np.diag([1,1,np.sign(np.linalg.det(Vt.T@U.T))]);R=Vt.T@D@U.T
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=bm-R@am
    return T

def match_hypotheses(T,prev_world,flow,dyn_prob,current,max_dist):
    static=transform(T,prev_world);dynamic=transform(T,prev_world+flow)
    # Select motion hypothesis using causal network probability, not current GT.
    use_dyn=dyn_prob>0.5
    predicted=np.where(use_dyn[:,None],dynamic,static)
    i,j,d=nearest_unique(predicted,current,max_dist)
    return i,j,d,use_dyn[i]
