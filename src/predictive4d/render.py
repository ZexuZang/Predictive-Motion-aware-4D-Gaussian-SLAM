"""CPU 3D Gaussian elliptical alpha splatting for functional reference outputs.

Not equivalent to diff-gaussian-rasterization's performance/gradients/SH.
This reference also supports optional *numerical* photometric pose refinement.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from PIL import Image
from .geometry import transform,project,se3_exp


def scaled_K(K,w0,h0,w1,h1):
    A=np.array(K,float,copy=True);sx=w1/w0;sy=h1/h0
    A[0,:]*=sx;A[1,:]*=sy;return A


def render_gaussians(gaussians,Tcw,K,h,w,max_items=850):
    """Approximate elliptical Gaussian alpha splatting into RGB+Depth+opacity."""
    rgb=np.zeros((h,w,3),dtype=np.float32); dep=np.zeros((h,w),dtype=np.float32)
    alpha=np.zeros((h,w),dtype=np.float32)
    if len(gaussians)==0:return rgb,dep,alpha
    if len(gaussians)>max_items:
        idx=np.linspace(0,len(gaussians)-1,max_items,dtype=int)
    else:idx=np.arange(len(gaussians))
    means=transform(Tcw,gaussians.mean[idx]);cov=gaussians.cov[idx];R=Tcw[:3,:3]
    uv,valid=project(means,K)
    # Far-to-near for over-compositing, a CPU approximation.
    for j in np.argsort(-means[:,2]):
        x,y,z=means[j]
        if not valid[j] or z<.12 or z>40:continue
        u,v=uv[j];
        if u<-15 or u>w+15 or v<-15 or v>h+15:continue
        J=np.array([[K[0,0]/z,0,-K[0,0]*x/z**2],
                    [0,K[1,1]/z,-K[1,1]*y/z**2]])
        S=J@R@cov[j]@R.T@J.T+np.eye(2)*.65**2
        vals,_=np.linalg.eigh(S);radius=int(np.ceil(2.8*np.sqrt(np.clip(vals.max(),.25,100))))
        radius=min(radius,18)
        x0=max(0,int(np.floor(u))-radius);x1=min(w,int(np.ceil(u))+radius+1)
        y0=max(0,int(np.floor(v))-radius);y1=min(h,int(np.ceil(v))+radius+1)
        if x1<=x0 or y1<=y0:continue
        yy,xx=np.mgrid[y0:y1,x0:x1];delta=np.stack([xx-u,yy-v],axis=-1)
        inv=np.linalg.inv(S);q=np.einsum('...i,ij,...j->...',delta,inv,delta)
        a=np.clip(gaussians.opacity[idx[j]]*np.exp(-.5*q),0,.95)
        aa=alpha[y0:y1,x0:x1];rgb_region=rgb[y0:y1,x0:x1];dregion=dep[y0:y1,x0:x1]
        # Front-to-back equivalent via far-to-near over operator
        rgb[y0:y1,x0:x1]=rgb_region*(1-a[...,None])+gaussians.color[idx[j]]*a[...,None]
        dep[y0:y1,x0:x1]=dregion*(1-a)+z*a
        alpha[y0:y1,x0:x1]=aa*(1-a)+a
    return rgb,dep,alpha


def resize_rgbd(rgb,depth,w,h):
    im=np.asarray(Image.fromarray(rgb).resize((w,h),Image.Resampling.BILINEAR),np.float32)/255
    # Bilinear interpolation of depth can mix discontinuities; only for preview pose refinement.
    d=np.asarray(Image.fromarray(np.asarray(depth,np.float32)).resize((w,h),Image.Resampling.NEAREST))
    return im,d


def refine_pose_render(T_init,static_map,frame,K,render_width=96,render_height=72,max_nfev=9,
                       enabled=False):
    """CPU numerical difference of rasterization photometric+depth loss.
    Expensive; off by default. Real Flow4DGS uses CUDA differentiable rendering.
    """
    if not enabled or len(static_map)<30:return T_init.copy()
    rgb,d=resize_rgbd(frame.rgb,frame.depth,render_width,render_height)
    Kr=scaled_K(K,frame.rgb.shape[1],frame.rgb.shape[0],render_width,render_height)
    # Small pose prior makes this conservative and prevents invalid image-only drift.
    stride=4
    def fun(delta):
        T=se3_exp(delta)@T_init
        rend,dep,a=render_gaussians(static_map,T,Kr,render_height,render_width,max_items=240)
        sl=(slice(None,None,stride),slice(None,None,stride))
        use=(d[sl]>.1)&(a[sl]>.06)
        diff_color=(rend[sl]-rgb[sl])*.35
        diff_depth=(dep[sl]-d[sl])*.12
        diff_color[~use]=0;diff_depth[~use]=0
        return np.r_[diff_color.reshape(-1),diff_depth.reshape(-1), delta*2.5]
    try:
        res=least_squares(fun,np.zeros(6),max_nfev=max_nfev,loss='soft_l1',f_scale=.2)
        if np.linalg.norm(res.x)<.25:return se3_exp(res.x)@T_init
    except (ValueError,np.linalg.LinAlgError):pass
    return T_init.copy()


def save_preview(filepath,map_static,map_dynamic,Tcw,K,shape,width=128,height=96):
    from pathlib import Path
    w0,h0=shape[1],shape[0];Kr=scaled_K(K,w0,h0,width,height)
    rgb_s,ds,as_=render_gaussians(map_static,Tcw,Kr,height,width)
    rgb_d,dd,ad=render_gaussians(map_dynamic,Tcw,Kr,height,width)
    combined=np.clip(rgb_s*(1-ad[...,None])+rgb_d,0,1)
    Path(filepath).parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray(np.uint8(combined*255)).save(filepath)
