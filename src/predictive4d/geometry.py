"""SE(3), camera geometry, robust registration and optimization. Units: metres."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation


def skew(v):
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)


def se3_exp(xi):
    """SE3 left increment, xi=[translation(3), rotation vector(3)]."""
    xi = np.asarray(xi, float).reshape(6)
    v, w = xi[:3], xi[3:]
    theta = float(np.linalg.norm(w)); W = skew(w)
    if theta < 1e-8:
        V = np.eye(3) + 0.5*W + (W @ W)/6
    else:
        V = (np.eye(3) + (1-np.cos(theta))/(theta**2)*W
             + (theta-np.sin(theta))/(theta**3)*(W @ W))
    T = np.eye(4); T[:3,:3] = Rotation.from_rotvec(w).as_matrix(); T[:3,3] = V @ v
    return T


def se3_log(T):
    """SE3 log with stable numerical solve for V inverse."""
    T=np.asarray(T, float); w=Rotation.from_matrix(T[:3,:3]).as_rotvec()
    theta=np.linalg.norm(w); W=skew(w)
    if theta < 1e-8: V=np.eye(3)+.5*W+(W@W)/6
    else: V=np.eye(3)+(1-np.cos(theta))/theta**2*W+(theta-np.sin(theta))/theta**3*(W@W)
    v=np.linalg.solve(V,T[:3,3])
    return np.r_[v,w]


def inverse(T):
    R=T[:3,:3]; q=np.eye(4);q[:3,:3]=R.T;q[:3,3]=-R.T@T[:3,3];return q


def transform(T, points):
    pts=np.asarray(points,float)
    return pts @ T[:3,:3].T + T[:3,3]


def unproject(depth, K, stride=1, max_points=2500, rng=None):
    """Return sampled camera-space points and their (row,col) image coordinates."""
    h,w=depth.shape; yy,xx=np.mgrid[0:h:stride,0:w:stride]
    yy=yy.reshape(-1); xx=xx.reshape(-1); z=depth[yy,xx]
    good=np.isfinite(z)&(z>0.1)&(z<30)
    yy,xx,z=yy[good],xx[good],z[good]
    if max_points and len(z)>max_points:
        if rng is None: rng=np.random.default_rng(0)
        ids=np.sort(rng.choice(len(z),size=max_points,replace=False)); yy,xx,z=yy[ids],xx[ids],z[ids]
    X=(xx-K[0,2])*z/K[0,0];Y=(yy-K[1,2])*z/K[1,1]
    return np.column_stack([X,Y,z]),np.column_stack([yy,xx]).astype(int)


def project(points_c,K):
    pts=np.asarray(points_c,float);z=pts[:,2]
    safe=np.where(z>1e-6,z,1e-6)
    uv=np.column_stack([K[0,0]*pts[:,0]/safe+K[0,2],K[1,1]*pts[:,1]/safe+K[1,2]])
    return uv,z>1e-6


def weighted_rigid(src,dst,weights):
    """World->camera weighted Umeyama/Kabsch *without scale*."""
    src=np.asarray(src,float);dst=np.asarray(dst,float);w=np.asarray(weights,float).reshape(-1)
    ok=np.isfinite(src).all(1)&np.isfinite(dst).all(1)&(w>1e-8)
    if np.count_nonzero(ok)<3: raise ValueError('Not enough correspondences')
    src,dst,w=src[ok],dst[ok],w[ok];w=w/w.sum()
    a=(src*w[:,None]).sum(0);b=(dst*w[:,None]).sum(0)
    C=((src-a)*w[:,None]).T@(dst-b)
    U,s,Vt=np.linalg.svd(C);D=np.diag([1.,1.,np.sign(np.linalg.det(Vt.T@U.T))]);R=Vt.T@D@U.T
    if s[1] < 1e-10:raise ValueError('Degenerate point geometry')
    t=b-R@a;T=np.eye(4);T[:3,:3]=R;T[:3,3]=t
    return T


def pose_gn(T,src_world,obs_camera,weights,flow_world=None,iterations=4,prior=None,prior_strength=0.,damping=1e-5):
    """Robust point-to-point SE3 GN, left perturbation. Prior is weak linearized 6D pose anchor."""
    if len(src_world)<6:return T.copy()
    T=T.copy();flow=np.zeros_like(src_world) if flow_world is None else np.asarray(flow_world)
    src=np.asarray(src_world)+flow;dst=np.asarray(obs_camera)
    base=np.asarray(weights,float).clip(0,1e5)
    for _ in range(iterations):
        p=transform(T,src);r=dst-p
        mag=np.linalg.norm(r,axis=1); huber=np.minimum(1.,.08/np.maximum(mag,1e-8))
        w=base*huber
        J=np.zeros((len(p),3,6));J[:,:,:3]=-np.eye(3)
        for i,pi in enumerate(p):J[i,:,3:]=skew(pi)
        H=np.einsum('nij,nik,n->jk',J,J,w);g=np.einsum('nij,ni,n->j',J,r,w)
        if prior is not None and prior_strength>0:
            e=se3_log(T@inverse(prior));H+=prior_strength*np.eye(6);g+=prior_strength*e
        H+=damping*np.eye(6)
        try:delta=np.linalg.solve(H,-g)
        except np.linalg.LinAlgError:break
        step=np.linalg.norm(delta)
        if step>0.5:delta*=.5/step
        T=se3_exp(delta)@T
        if step<1e-6:break
    return T


def predict_pose(trajectory):
    """World->camera constant relative motion; valid without GT poses."""
    if len(trajectory)<2:return np.array(trajectory[-1],copy=True)
    return trajectory[-1]@inverse(trajectory[-2])@trajectory[-1]
