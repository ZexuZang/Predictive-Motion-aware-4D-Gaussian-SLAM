"""Causal neural motion prediction, static geometric tracking, gated dynamic feedback.

Dynamic feedback uses PRIOR predicted Gaussians (no current-frame map update)
and excludes static-used current observations; do not treat all reused RGB-D
measurements as statistically independent evidence.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import torch
from .geometry import se3_exp,se3_log,invert,transform,cap_pose,rigid_fit,nearest_unique,match_hypotheses
from .observability import static_camera_information,weak_direction_test

@dataclass
class Past:
    xyz_world: np.ndarray
    rgb: np.ndarray
    T_cw: np.ndarray
    timestamp: float
    dynamic_mask: np.ndarray | None = None

class NeuralPredictor:
    def __init__(self,net,config,device):
        self.net=net;self.cfg=config;self.device=device

    def predict(self,history,dt):
        """Return prior camera pose, neural point world flow, dynamic probs, flow var.
        History contains ONLY previous estimated frame states.
        """
        L=int(self.cfg['history_length']);n=int(self.cfg['points_per_frame'])
        if len(history)<L:raise ValueError('not enough history for neural predictor')
        h=history[-L:]
        xyz=[];features=[];ids=[]
        for i,fr in enumerate(h):
            if len(fr.xyz_world)==0:raise ValueError('Empty history cloud')
            idx=np.linspace(0,len(fr.xyz_world)-1,n,dtype=int)
            x=fr.xyz_world[idx]
            c=fr.rgb[idx]
            feat=np.column_stack([x,c,np.ones(len(x))]).astype(np.float32)
            xyz.append(x);features.append(feat);ids.append(idx)
        # NN hint is causal. It can be noisy on sampled surfaces and should not
        # be mistaken for supervised scene flow.
        if L>1:
            _,j=cKDTree(xyz[-2]).query(xyz[-1])
            hint=(xyz[-1]-xyz[-2][j]).astype(np.float32)
            hint=np.clip(hint,-.5,.5)
        else:hint=np.zeros((n,3),np.float32)
        twists=np.zeros((L,6),np.float32);dts=np.full((L,1),.1,np.float32)
        for i in range(1,L):
            twists[i]=se3_log(h[i].T_cw@invert(h[i-1].T_cw)).astype(np.float32)
            dts[i,0]=max(.001,h[i].timestamp-h[i-1].timestamp)
        dev=self.device
        with torch.inference_mode():
            result=self.net(
                torch.tensor(np.stack(features)[None],device=dev),
                torch.tensor(twists[None],device=dev),
                torch.tensor(dts[None],device=dev),
                torch.tensor(hint[None],device=dev))
        xi=result['pose_xi'][0].cpu().numpy().astype(float)
        flow=result['flow'][0].cpu().numpy().astype(float)
        p=torch.sigmoid(result['dynamic_logits'][0]).cpu().numpy().astype(float)
        var=np.exp(result['flow_logvar'][0].cpu().numpy().astype(float))
        var_pose=np.exp(result['pose_logvar'][0].cpu().numpy().astype(float))
        if not np.isfinite(xi).all() or not np.isfinite(flow).all():
            raise FloatingPointError('Joint network produced nonfinite motion')
        # Supervision uses displacement per dt=0.1s; scale modest dt differences.
        ratio=float(np.clip(dt/.1,.3,2.))
        xi*=ratio;flow*=ratio
        T_candidate=se3_exp(xi)@h[-1].T_cw
        T_pred,pose_valid=cap_pose(T_candidate,h[-1].T_cw,
            float(self.cfg['max_pose_translation_m'])*ratio,
            float(self.cfg['max_pose_rotation_deg'])*ratio)
        # Reject rather than silently substitute constant velocity.
        mags=np.linalg.norm(flow,axis=1);maxflow=float(self.cfg['max_flow_m'])*ratio
        flow=flow*np.minimum(1.,maxflow/np.maximum(mags,1e-10))[:,None]
        # Map predictions back to all points in the last observed frame.
        _,which=cKDTree(xyz[-1]).query(h[-1].xyz_world)
        return T_pred,flow[which],p[which],var[which],{
            'pose_accepted':bool(pose_valid),
            'raw_pose_translation':float(np.linalg.norm(xi[:3])),
            'raw_pose_rotation_deg':float(np.rad2deg(np.linalg.norm(xi[3:]))),
            'mean_motion_confidence':float(p.mean()),
            'pose_variance_proxy':var_pose.tolist(),
        }

def static_pose_correction(T0,world,flow,prob,current,cfg,mask_previous=None):
    """Incremental geometric registration. Never uses GT poses."""
    if len(world)==0:return T0.copy(),dict(status='empty',matched=0),np.empty(0,int)
    static=(prob<float(cfg['max_static_confidence']))
    if mask_previous is not None:static=static&(~mask_previous)
    indices=np.flatnonzero(static)
    T=T0.copy();static_obs=np.empty(0,int);match_count=0;median=np.inf
    if len(indices)<int(cfg['min_static_matches']):
        return T,dict(status='low_static_prior',matched=0),static_obs
    for _ in range(3):
        projected=transform(T,world[indices]);a,b,d=nearest_unique(projected,current,float(cfg['max_match_distance']))
        match_count=len(a);median=float(np.median(d)) if len(d) else np.inf
        if match_count<int(cfg['min_static_matches']):break
        use=d<=max(.05,2.5*np.median(d))
        if np.count_nonzero(use)<int(cfg['min_static_matches']):break
        try:Tfit=rigid_fit(world[indices[a[use]]],current[b[use]],1/(.02+d[use])**2)
        except (ValueError,np.linalg.LinAlgError):break
        Tnew,accepted=cap_pose(Tfit,T0,float(cfg['max_pose_translation_m']),float(cfg['max_pose_rotation_deg']))
        if not accepted:break
        T=Tnew;static_obs=b[use]
    return T,dict(status='tracking_ok' if len(static_obs)>=int(cfg['min_static_matches']) else 'tracking_weak',
        matched=int(len(static_obs)),median_residual_m=median),static_obs

def dynamic_pose_feedback(T,prev_T,current,static_world,static_used,prior_map,cfg):
    """Independent predicted-dynamic Gaussian candidates, cautious SE3 correction.

    - Only map state predicted from HISTORY, before observation correction.
    - Static-used observations EXCLUDED from dynamic matches.
    - Reliability gates residuals and predicted motion variance.
    - Static evidence protects against deterioration; also bounds pose jump.
    """
    diag={'status':'disabled','accepted':False,'dynamic_matches':0,'pose_delta_m':0.}
    if not cfg.get('feedback_enabled',True):return T,diag
    if not len(prior_map.mean):return T,{**diag,'status':'no_dynamic_map'}
    Hstatic,static_information_matches=static_camera_information(T,static_world,current,float(cfg['max_match_distance']))
    diag['static_information_matches']=int(static_information_matches)
    if static_information_matches>=int(cfg['feedback_static_weak_below']):
        eig=np.linalg.eigvalsh(Hstatic)
        if eig[0]>max(1e-5,eig[-1]*float(cfg.get('weak_information_fraction',.08))):
            return T,{**diag,'status':'static_strong_skip'}
    pred_cam=transform(T,prior_map.mean)
    # Dynamic Gaussian centers should correspond to distinct current points.
    maxdist=float(cfg['max_match_distance'])
    i,j,d=nearest_unique(pred_cam,current,maxdist)
    if len(i)==0:return T,{**diag,'status':'no_dynamic_correspondences'}
    allowed=~np.isin(j,static_used)
    i,j,d=i[allowed],j[allowed],d[allowed]
    # Unit-consistent innovation gating: residual / sqrt(sensor^2+motion variance).
    sigma=np.sqrt(.035**2+np.maximum(1e-5,prior_map.motion_var[i]))
    gate=d<=np.minimum(.20,float(cfg['feedback_gate_sigma'])*sigma)
    i,j,d,sigma=i[gate],j[gate],d[gate],sigma[gate]
    diag['dynamic_matches']=int(len(i))
    if len(i)<int(cfg['feedback_min_dynamic_matches']):return T,{**diag,'status':'not_enough_reliable_dynamic'}
    src=prior_map.mean[i];dst=current[j]
    try:
        Tfit=rigid_fit(src,dst,1/(sigma**2))
    except (ValueError,np.linalg.LinAlgError):return T,{**diag,'status':'geometry_degenerate'}
    Tfit,accepted=cap_pose(Tfit,T,float(cfg['feedback_max_translation_m']),float(cfg['feedback_max_rotation_deg']))
    if not accepted:return T,{**diag,'status':'pose_jump_rejected'}
    # Static information's weak eigenspace controls when dynamic compensation
    # may contribute. Full-rank static geometry does not need this feedback.
    if bool(cfg.get('require_weak_static_direction',True)):
        weak_ok,weak_info=weak_direction_test(T,Tfit,Hstatic,
            float(cfg.get('weak_information_fraction',.08)))
        diag.update(weak_info)
        if not weak_ok and static_information_matches>0:
            return T,{**diag,'status':'no_reliable_weak_direction'}
        if static_information_matches==0:
            diag['weak_direction_status']='all_directions_weak_no_static_anchor'
    # Dynamic photogeometric residual must improve, and when static evidence
    # exists it cannot deteriorate materially on a held-out static subset.
    before=np.median(np.linalg.norm(transform(T,src)-dst,axis=1))
    after=np.median(np.linalg.norm(transform(Tfit,src)-dst,axis=1))
    if not after+1e-6<before:return T,{**diag,'status':'no_dynamic_improvement'}
    if len(static_world)>=12:
        q=static_world[np.linspace(0,len(static_world)-1,min(180,len(static_world)),dtype=int)]
        # Compare nearest-neighbor static residual on current observations.
        tree=cKDTree(current)
        db,_=tree.query(transform(T,q));da,_=tree.query(transform(Tfit,q))
        if np.median(da)>max(.035,np.median(db))*float(cfg['feedback_static_degradation']):
            return T,{**diag,'status':'static_consistency_rejected'}
    jump=np.linalg.norm(Tfit[:3,3]-T[:3,3])
    return Tfit,{**diag,'status':'accepted','accepted':True,'pose_delta_m':float(jump),
                 'median_before_m':float(before),'median_after_m':float(after)}
