"""Lightweight Gaussian primitive map and non-rigid scene-flow adapter.

This map represents Gaussian primitives, but is NOT the official Flow4DGS mapping
kernel, opacity GMM, or adaptive densification implementation.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

@dataclass
class GaussianMap:
    mean:np.ndarray        # N 3 world centres
    cov:np.ndarray         # N 3 3 spatial covariance (NOT prediction uncertainty)
    color:np.ndarray       # N 3 [0,1]
    opacity:np.ndarray     # N
    motion_variance:np.ndarray # N; scalar uncertainty proxy, uncalibrated
    @classmethod
    def from_points(cls,xyz,color,scale=.02,var=.04,opacity=.7):
        n=len(xyz)
        return cls(np.asarray(xyz,float).reshape(-1,3).copy(),
          np.repeat((np.eye(3)*scale**2)[None],n,axis=0),
          np.asarray(color,float).reshape(-1,3).copy(),
          np.full(n,opacity),np.full(n,var))
    def __len__(self):return len(self.mean)
    def save(self,path):
        Path(path).parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,mean=self.mean,cov=self.cov,color=self.color,
                            opacity=self.opacity,motion_variance=self.motion_variance)
    def append(self,other,max_size=1200):
        if len(other)==0:return
        self.mean=np.concatenate([self.mean,other.mean]);self.cov=np.concatenate([self.cov,other.cov]);
        self.color=np.concatenate([self.color,other.color]);self.opacity=np.concatenate([self.opacity,other.opacity]);
        self.motion_variance=np.concatenate([self.motion_variance,other.motion_variance]);
        if len(self)>max_size:
            ix=np.linspace(0,len(self)-1,max_size,dtype=int)
            for key in ('mean','cov','color','opacity','motion_variance'):
                setattr(self,key,getattr(self,key)[ix])


def nearest_field(points,flow,query,max_distance=.20):
    if not len(points):return np.zeros_like(query)
    distance,idx=cKDTree(points).query(query)
    v=np.asarray(flow)[idx].copy();v[distance>max_distance]=0
    return v


def transfer_motion(gaussians,points,flows,confidence=None,k=12,radius=.15,
                    regularization=1e-4,min_spatial_rank=3):
    """Return predicted Gaussian state (does not mutate). Uses local affine flow.
    If local points are planar/degenerate, regularization shrinks unsupported F.
    """
    if len(gaussians)==0:return GaussianMap.from_points(np.empty((0,3)),np.empty((0,3)))
    if not len(points):return GaussianMap(gaussians.mean.copy(),gaussians.cov.copy(),gaussians.color.copy(),gaussians.opacity.copy(),gaussians.motion_variance.copy())
    tree=cKDTree(points); kk=min(k,len(points));distance,idx=tree.query(gaussians.mean,k=kk)
    if kk==1:distance=distance[:,None];idx=idx[:,None]
    out_mu=gaussians.mean.copy();out_cov=gaussians.cov.copy();out_var=gaussians.motion_variance.copy()
    conf=np.ones(len(points)) if confidence is None else np.clip(np.asarray(confidence),0,1)
    for i in range(len(gaussians)):
        valid=distance[i]<radius
        if not np.any(valid):out_var[i]*=2;continue
        ix=idx[i,valid];v=flows[ix];x=points[ix]
        ww=np.exp(-.5*(distance[i,valid]/max(radius/2,1e-6))**2)*conf[ix]
        if ww.sum()<1e-10:continue
        ww=ww/ww.sum();xb=(x*ww[:,None]).sum(0);vb=(v*ww[:,None]).sum(0)
        q=x-xb;u=v-vb
        G=(q*ww[:,None]).T@q
        C=(u*ww[:,None]).T@q
        A=C@np.linalg.inv(G+regularization*np.eye(3))
        if np.linalg.eigvalsh(G)[0]<1e-5 or len(x)<4:
            # Planar point sets are rank deficient: conservative translation update.
            A=np.zeros((3,3))
        out_mu[i]=gaussians.mean[i]+vb+A@(gaussians.mean[i]-xb)
        F=np.eye(3)+A
        if np.linalg.det(F)<.2 or np.linalg.det(F)>5:
            F=np.eye(3)  # reject physically implausible local deformation
        updated=F@gaussians.cov[i]@F.T
        ev,U=np.linalg.eigh(updated);ev=np.clip(ev,1e-7,.12**2)
        out_cov[i]=(U*ev)@U.T;out_var[i]=max(1e-4,gaussians.motion_variance[i]*1.1)
    return GaussianMap(out_mu,out_cov,gaussians.color.copy(),gaussians.opacity.copy(),out_var)
