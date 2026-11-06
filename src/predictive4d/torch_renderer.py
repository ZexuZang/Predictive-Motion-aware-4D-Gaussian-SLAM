"""Differentiable low-resolution 3D Gaussian RGB-D renderer and mapper.

Torch-native differentiable, *not* official Flow4DGS CUDA rasterizer nor
exact transmittance-sorted compositing. Uses a soft Gaussian mixture renderer.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from .gaussians import GaussianMap


def soft_gaussian_rgbd(means,cov,colors,opacity,T,K,height=72,width=96,orig_hw=(480,640)):
    """Differentiable soft normalized Gaussian mixture in the camera image.

    means [G,3], cov [G,3,3], colors [G,3], opacity [G];
    T [4,4] world->camera; K [3,3] intrinsics at original resolution.
    Returns RGB [H,W,3], depth [H,W], weight [H,W].
    """
    if means.shape[0]==0:
        raise ValueError('Need nonempty Gaussian set')
    dtype=means.dtype;dev=means.device
    fx=K[0,0]*width/orig_hw[1];fy=K[1,1]*height/orig_hw[0]
    cx=K[0,2]*width/orig_hw[1];cy=K[1,2]*height/orig_hw[0]
    R=T[:3,:3];t=T[:3,3]
    pc=means@R.T+t
    x,y,z=pc.unbind(-1)
    z_safe=z.clamp(min=.15)
    u=fx*x/z_safe+cx;v=fy*y/z_safe+cy
    # Projection Jacobian of perspective projection.
    z2=z_safe.square()
    j=torch.zeros((len(z),2,3),device=dev,dtype=dtype)
    j[:,0,0]=fx/z_safe;j[:,0,2]=-fx*x/z2
    j[:,1,1]=fy/z_safe;j[:,1,2]=-fy*y/z2
    cov_camera=R[None]@cov@R.T[None]
    S=j@cov_camera@j.transpose(-1,-2)+torch.eye(2,device=dev,dtype=dtype)*.8
    # 2x2 inverse with defensive epsilon and differentiable operations.
    a=S[:,0,0];b=S[:,0,1];c=S[:,1,0];d=S[:,1,1]
    det=(a*d-b*c).clamp(min=1e-5)
    ia=d/det;ib=-b/det;ic=a/det
    py,px=torch.meshgrid(torch.arange(height,device=dev,dtype=dtype),
                         torch.arange(width,device=dev,dtype=dtype),indexing='ij')
    du=px.reshape(1,-1)-u[:,None];dv=py.reshape(1,-1)-v[:,None]
    mahal=ia[:,None]*du.square()+2*ib[:,None]*du*dv+ic[:,None]*dv.square()
    valid=(z>.2)&(z<12.)
    # Depth confidence discourages far Gaussians from overpowering near ones.
    weights=torch.exp(-.5*mahal.clamp(min=0,max=100))*opacity[:,None]*valid[:,None]
    weights=weights*torch.exp(-.10*z_safe[:,None])
    mass=weights.sum(0)
    alpha=1-torch.exp(-mass)
    color=(weights.T@colors)/(mass[:,None]+1e-6)
    depth=(weights.T@z_safe[:,None]).squeeze(1)/(mass+1e-6)
    return (alpha[:,None]*color).reshape(height,width,3),depth.reshape(height,width),mass.reshape(height,width)


def optimize_gaussian_keyframe(mapper,frame,T_cw,K,device='cpu',steps=1,
                              height=72,width=96,max_static=64,max_dynamic=32,lr=.001):
    """Photometric+depth gradient step on a subset of Gaussian centers/colors.

    Observations enter ONLY AFTER motion/pose prediction and dynamic feedback.
    This is a lightweight, low-resolution mapper to validate differentiable
    dataflow; not the upstream full Gaussian mapping optimizer.
    """
    if steps<=0:return {'status':'disabled'}
    maps=[];ids=[];kinds=[]
    for key,cap in (('static',max_static),('dynamic',max_dynamic)):
        g=getattr(mapper,key)
        if len(g):
            idx=np.linspace(0,len(g)-1,min(cap,len(g)),dtype=int)
            maps.append(g);ids.append(idx);kinds.append(key)
    if not maps:return {'status':'no_gaussians'}
    means=np.concatenate([g.mean[idx] for g,idx in zip(maps,ids)])
    cov=np.concatenate([g.cov[idx] for g,idx in zip(maps,ids)])
    cols=np.concatenate([g.color[idx] for g,idx in zip(maps,ids)])
    opa=np.concatenate([g.opacity[idx] for g,idx in zip(maps,ids)])
    def ten(x):return torch.as_tensor(x,device=device,dtype=torch.float32)
    mu=torch.nn.Parameter(ten(means));color_logits=torch.nn.Parameter(torch.logit(ten(cols).clamp(.01,.99)))
    # Fixed Gaussian shape in this mapping optimizer; deformation of the
    # covariance is handled by the historical local-affine adapter.
    sigma=ten(cov);alpha=ten(opa).clamp(.05,.99)
    T=ten(T_cw);K_t=ten(K)
    gt_color=ten(frame.rgb.copy()).permute(2,0,1)[None]/255.
    gt_depth=ten(frame.depth.copy())[None,None]
    obs_color=F.interpolate(gt_color,(height,width),mode='bilinear',align_corners=False)[0].permute(1,2,0)
    obs_depth=F.interpolate(gt_depth,(height,width),mode='nearest')[0,0]
    depth_valid=(obs_depth>.15)&(obs_depth<12)
    opt=torch.optim.Adam([mu,color_logits],lr=lr)
    final=None
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        rgb,dep,mass=soft_gaussian_rgbd(mu,sigma,torch.sigmoid(color_logits),alpha,T,K_t,height,width,frame.depth.shape)
        mask=(mass>.05)&depth_valid
        if mask.sum()<8:break
        loss_color=(rgb[mask]-obs_color[mask]).abs().mean()
        loss_depth=(dep[mask]-obs_depth[mask]).abs().mean()
        loss=loss_color+.1*loss_depth+1e-4*(mu-ten(means)).square().mean()
        if not torch.isfinite(loss):return {'status':'nonfinite_loss_skipped'}
        loss.backward();torch.nn.utils.clip_grad_norm_([mu,color_logits],max_norm=1.)
        opt.step();final=float(loss.item())
    if final is None:return {'status':'insufficient_visibility'}
    with torch.no_grad():
        mu_np=mu.cpu().numpy();color_np=torch.sigmoid(color_logits).cpu().numpy()
    if not (np.isfinite(mu_np).all() and np.isfinite(color_np).all()):
        return {'status':'nonfinite_update_skipped'}
    pos=0
    for g,idx in zip(maps,ids):
        m=len(idx);g.mean[idx]=mu_np[pos:pos+m];g.color[idx]=color_np[pos:pos+m];pos+=m
    return {'status':'optimized','loss':final,'gaussians_optimized':len(means)}
