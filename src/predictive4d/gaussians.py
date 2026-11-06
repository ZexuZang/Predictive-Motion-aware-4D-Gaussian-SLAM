"""World-anchored 4D Gaussian state and causal local scene-flow deformation.

This implements 3D Gaussian parameters and temporal deformation; it is NOT
Flow4DGS-SLAM's differentiable CUDA renderer, GMM temporal opacity or trainer.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image
from .geometry import transform

@dataclass
class GaussianMap:
    mean:np.ndarray
    cov:np.ndarray
    color:np.ndarray
    opacity:np.ndarray
    motion_var:np.ndarray

    @classmethod
    def empty(cls):
        return cls(np.zeros((0,3)),np.zeros((0,3,3)),np.zeros((0,3)),np.zeros(0),np.zeros(0))
    @classmethod
    def from_points(cls,pts,color,variance=.02**2):
        n=len(pts)
        return cls(np.asarray(pts,float).reshape(n,3).copy(),
            np.repeat((np.eye(3)*variance)[None],n,axis=0),
            np.asarray(color,float).reshape(n,3).copy(),
            np.full(n,.8),np.full(n,.03**2))
    def __len__(self):return len(self.mean)
    def append(self,other,limit):
        if not len(other):return
        for attr in ('mean','cov','color','opacity','motion_var'):
            setattr(self,attr,np.concatenate([getattr(self,attr),getattr(other,attr)],axis=0))
        if len(self)>limit:
            # Explicit deterministic capacity policy, not a learned pruning method.
            ids=np.linspace(0,len(self)-1,limit,dtype=int)
            for attr in ('mean','cov','color','opacity','motion_var'):
                setattr(self,attr,getattr(self,attr)[ids])
    def save(self,path):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,mean=self.mean,cov=self.cov,color=self.color,
                            opacity=self.opacity,motion_var=self.motion_var)


def propagate_gaussians(gmap,prev_xyz,flows,flow_var,radius=.24,k=12):
    """Strictly historical prediction: nearest flow + local affine gradient.
    Uses translation-only when neighborhood geometry is insufficient.
    """
    if not len(gmap):return GaussianMap.empty()
    out=GaussianMap(gmap.mean.copy(),gmap.cov.copy(),gmap.color.copy(),gmap.opacity.copy(),gmap.motion_var.copy())
    if not len(prev_xyz):return out
    tree=cKDTree(prev_xyz);kk=min(k,len(prev_xyz));dd,ids=tree.query(gmap.mean,k=kk)
    if kk==1:dd=dd[:,None];ids=ids[:,None]
    for i in range(len(gmap)):
        valid=dd[i]<radius
        if not np.any(valid):out.motion_var[i]*=2;continue
        idx=ids[i,valid];x=prev_xyz[idx];v=flows[idx]
        w=np.exp(-.5*(dd[i,valid]/max(radius/2,1e-5))**2)
        w/=w.sum();xb=np.sum(x*w[:,None],axis=0);vb=np.sum(v*w[:,None],axis=0)
        # Observability-aware local displacement Jacobian: only infer directions
        # supported by the historical 3D neighborhood. Typical RGB-D surfaces
        # are locally rank-2; their normal derivative is unobservable.
        A=np.zeros((3,3))
        if len(x)>=5:
            q=x-xb;u=v-vb
            G=(q*w[:,None]).T@q
            C=(u*w[:,None]).T@q
            if np.isfinite(G).all() and np.isfinite(C).all():
                eig,Q=np.linalg.eigh(G)
                good=eig>max(1e-7,1e-3*max(float(eig[-1]),1e-12))
                rank=int(np.count_nonzero(good))
                if rank>=2:
                    # Pseudoinverse restricted to supported tangent directions.
                    inv=Q[:,good]@np.diag(1/(eig[good]+1e-3))@Q[:,good].T
                    A=C@inv
                    norm=float(np.linalg.norm(A,ord=2))
                    if norm>.4:A*=.4/norm
        # Rank < 2: translation-only; preserves orientation and scale.
        out.mean[i]+=vb+A@(gmap.mean[i]-xb)
        F=np.eye(3)+A;new=F@gmap.cov[i]@F.T
        ev,Q=np.linalg.eigh(new);out.cov[i]=(Q*np.clip(ev,1e-7,.12**2))@Q.T
        out.motion_var[i]=max(1e-5,.5*np.mean(flow_var[idx])+0.5*gmap.motion_var[i])
    return out

class GaussianMapper:
    def __init__(self,cfg):
        self.cfg=cfg;self.static=GaussianMap.empty();self.dynamic=GaussianMap.empty()
    def predict_dynamic(self,prev_world,flow,flow_var):
        return propagate_gaussians(self.dynamic,prev_world,flow,flow_var)
    def integrate(self,world,rgb,dyn_prob,predicted_dynamic,frame_index):
        # Current-frame observations are assimilated only AFTER all pose updates.
        self.dynamic=predicted_dynamic
        n=int(self.cfg['append_gaussians'])
        for dynamic in (False,True):
            threshold=float(self.cfg['min_dynamic_confidence'] if dynamic else self.cfg['max_static_confidence'])
            ix=np.flatnonzero(dyn_prob>=threshold if dynamic else dyn_prob<=threshold)
            if not len(ix):continue
            ix=ix[np.linspace(0,len(ix)-1,min(n,len(ix)),dtype=int)]
            new=GaussianMap.from_points(world[ix],rgb[ix])
            g=self.dynamic if dynamic else self.static
            g.append(new,int(self.cfg['max_dynamic_map' if dynamic else 'max_static_map']))
    def snapshot(self,folder,index):
        folder=Path(folder)
        self.static.save(folder/'static_maps'/f'{index:06d}.npz')
        self.dynamic.save(folder/'dynamic_maps'/f'{index:06d}.npz')


def render_preview(static,dynamic,T_cw,K,height,width,out_width=320,out_height=240):
    """CPU elliptical Gaussian alpha-composite preview (not CUDA differentiable)."""
    canvas=np.zeros((out_height,out_width,3),np.float32)+.05
    sx=out_width/width;sy=out_height/height
    K2=K.copy();K2[0]*=sx;K2[1]*=sy
    comps=[]
    for gmap,is_dynamic in ((static,False),(dynamic,True)):
        if not len(gmap):continue
        p=transform(T_cw,gmap.mean); R=T_cw[:3,:3]
        for i,point in enumerate(p):
            x,y,z=point
            if not np.isfinite(point).all() or z<=.15:continue
            u=K2[0,0]*x/z+K2[0,2];v=K2[1,1]*y/z+K2[1,2]
            if u< -15 or u>out_width+15 or v<-15 or v>out_height+15:continue
            J=np.array([[K2[0,0]/z,0,-K2[0,0]*x/z**2],
                        [0,K2[1,1]/z,-K2[1,1]*y/z**2]])
            S=J@R@gmap.cov[i]@R.T@J.T+np.eye(2)*.5
            if not np.isfinite(S).all() or np.linalg.det(S)<=1e-9:continue
            comps.append((z,u,v,S,gmap.color[i],gmap.opacity[i]))
    # Far-to-near alpha blending; no trained visibility model.
    comps.sort(key=lambda x:-x[0])
    for z,u,v,S,color,alpha in comps:
        radius=min(9,max(2,int(np.ceil(2.5*np.sqrt(np.linalg.eigvalsh(S)[-1])))))
        x0=max(0,int(u)-radius);x1=min(out_width,int(u)+radius+1)
        y0=max(0,int(v)-radius);y1=min(out_height,int(v)+radius+1)
        if x0>=x1 or y0>=y1:continue
        yy,xx=np.mgrid[y0:y1,x0:x1];dx=np.stack([xx-u,yy-v],axis=-1)
        try:inv=np.linalg.inv(S)
        except np.linalg.LinAlgError:continue
        w=(alpha*np.exp(-.5*np.einsum('...i,ij,...j->...',dx,inv,dx)))[...,None]
        canvas[y0:y1,x0:x1]=canvas[y0:y1,x0:x1]*(1-w)+color[None,None,:]*w
    return Image.fromarray(np.uint8(np.clip(canvas*255,0,255)))
