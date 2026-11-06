"""Optional REAL Flow4DGS-SLAM codebase integration entry points.

Called FROM patched upstream FrontEnd.tracking and BackEnd.keyframe handlers.
No rendering is emulated here: once hooked, native upstream continues to use
its own CUDA rasterizer, GaussianModel, keyframe optimizer and GMM logic.

The packet generator is separately run. That two-pass version is NOT
full end-to-end online neural SLAM, and should not be reported as such.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from .geometry import is_pose
from .official_bridge import SCHEMA,map_flow_to_official_gaussian_offsets


def _packet(config,frame_index):
    settings=config.get('predictive4d',{})
    folder=settings.get('packet_dir')
    if not folder:return None
    file=Path(folder)/f'{frame_index:06d}.npz'
    if not file.is_file():return None
    packet=np.load(file,allow_pickle=False)
    if str(packet['schema'])!=SCHEMA or int(packet['index'])!=int(frame_index):
        raise ValueError('Prediction packet schema/frame mismatch')
    return file,packet


def apply_camera_packet(config,viewpoint,frame_index,frontend=None):
    """Set predicted native camera W2C pose, optionally run joint Schur tracking.

    When frontend is supplied, use current RGB-D for motion-compensated
    correction, then (if valid) historical native dynamic Gaussians for gated
    dynamic-to-camera feedback BEFORE native renderer refines the pose.
    """
    hit=_packet(config,frame_index)
    if hit is None:return False
    if config.get('model_params',{}).get('flow_cam',False):
        raise RuntimeError('Set model_params.flow_cam=false when applying neural camera prior')
    filename,p=hit
    T=np.asarray(p['T_cw'],float)
    if not is_pose(T):raise FloatingPointError('Invalid SE3 neural camera prior')
    options=config.get('predictive4d',{})
    if frontend is not None and options.get('native_joint_refine',False):
        from .flow4dgs_online import _as_numpy_camera_frame
        from .geometry import unproject
        from .joint_ba import optimize_joint_pose
        from .tracking import dynamic_pose_feedback
        from .gaussians import GaussianMap
        from scipy.spatial import cKDTree
        camera_frame=_as_numpy_camera_frame(viewpoint)
        K=viewpoint.intrinsic
        K=K.detach().cpu().numpy() if isinstance(K,torch.Tensor) else np.asarray(K)
        cfg=frontend._predictive4d_runtime.cfg if hasattr(frontend,'_predictive4d_runtime') else options.get('joint_config')
        if not isinstance(cfg,dict):
            raise RuntimeError('Native joint tracking requires online runtime or predictive4d.joint_config')
        camera,_,_=unproject(camera_frame,K,int(cfg['max_points']),
                                      int(cfg['pixel_stride']),int(cfg['seed'])+frame_index)
        if len(camera)<40:raise RuntimeError('Insufficient current RGB-D for neural motion correction')
        # Viewpoint is currently initialized to previous refined pose by native code.
        previous=np.eye(4)
        previous[:3,:3]=viewpoint.R.detach().cpu().numpy()
        previous[:3,3]=viewpoint.T.detach().cpu().numpy()
        T,gamma,used,diag=optimize_joint_pose(T,previous,p['points_world'],p['flow_world'],
                              p['dynamic_probability'],p['flow_variance'],camera,cfg)
        setattr(viewpoint,'predictive4d_joint_diag',diag)
        # Native dynamic-Gaussian feedback uses world-space positions stored in
        # the historical native map. Does not use Gaussians updated at frame t.
        gauss=getattr(frontend,'gaussians',None)
        if gauss is not None and hasattr(gauss,'dygs') and hasattr(gauss,'get_dygs_xyz'):
            with torch.no_grad():
                dyn=gauss.get_dygs_xyz.detach()
                # Native frontend may not have a previous dynamic keyframe yet.
                if len(dyn)>0 and len(getattr(gauss,'kf_list',[]))>0:
                    last_idx=len(gauss.kf_list)-1
                    dyn=(dyn+gauss.get_delta_xyz(last_idx).detach()).cpu().numpy()
                else:dyn=np.zeros((0,3))
                static=gauss.get_xyz.detach()[~gauss.dygs].cpu().numpy()
            if len(dyn):
                offsets,accepted=map_flow_to_official_gaussian_offsets(filename,dyn,
                        max_distance=float(options.get('max_gaussian_match_m',.15)),
                        min_dynamic_prob=float(options.get('min_dynamic_probability',.65)))
                if int(accepted.sum())>=int(cfg.get('feedback_min_dynamic_matches',8)):
                    prior=GaussianMap.from_points(dyn[accepted]+offsets[accepted],
                                                  np.zeros((int(accepted.sum()),3)))
                    distances,source_ids=cKDTree(p['points_world']).query(dyn[accepted])
                    prior.motion_var[:]=np.maximum(1e-5,p['flow_variance'][source_ids].mean(axis=1))
                    T,feedback_diag=dynamic_pose_feedback(T,previous,camera,static,used,prior,cfg)
                    setattr(viewpoint,'predictive4d_feedback_diag',feedback_diag)
    R=torch.as_tensor(T[:3,:3],device=viewpoint.device,dtype=viewpoint.R.dtype)
    t=torch.as_tensor(T[:3,3],device=viewpoint.device,dtype=viewpoint.T.dtype)
    viewpoint.update_RT(R,t)
    return True

def apply_dynamic_packet(backend,frame_index,previous_keyframe_id):
    """Replace upstream *optical-flow-derived offset* with neural world flow.

    Only applies when current/previous native keyframe IDs and native dynamic
    Gaussian indices are aligned and valid. Otherwise returns False, leaving
    native fallback flow path unchanged. Never copies our different Gaussian
    arrays directly into the native GaussianModel.
    """
    hit=_packet(backend.config,frame_index)
    if hit is None:return False
    filename,p=hit
    gs=backend.gaussians
    if previous_keyframe_id is None:return False
    if previous_keyframe_id not in gs.kf_list or frame_index not in gs.kf_list:
        raise RuntimeError('Native Gaussian keyframe IDs not ready for offset update')
    if frame_index==previous_keyframe_id:return False
    previous_idx=gs.kf_list.index(previous_keyframe_id)
    if frame_index!=gs.kf_list[-1]:
        raise RuntimeError('Expected append-only latest keyframe for native offset update')
    with torch.no_grad():
        xyz=(gs.get_dygs_xyz.detach()+gs.get_delta_xyz(previous_idx).detach())
    if xyz.ndim!=2 or xyz.shape[-1]!=3:raise RuntimeError('Unexpected native dynamic xyz layout')
    if len(xyz)==0:return False
    world_xyz=xyz.cpu().numpy()
    offs,accepted=map_flow_to_official_gaussian_offsets(filename,world_xyz,
        max_distance=float(backend.config.get('predictive4d',{}).get('max_gaussian_match_m',.15)),
        min_dynamic_prob=float(backend.config.get('predictive4d',{}).get('min_dynamic_probability',.65)))
    minimum=int(backend.config.get('predictive4d',{}).get('min_gaussian_matches',12))
    if int(accepted.sum())<minimum:return False
    maxmotion=float(backend.config.get('predictive4d',{}).get('max_gaussian_motion_m',.10))
    accepted &= (np.linalg.norm(offs,axis=1)<maxmotion)
    if int(accepted.sum())<minimum:return False
    # Native update_delta_xyz creates a new per-keyframe nn.Parameter and copies
    # previous offsets + selected predicted world displacements.
    ids=torch.as_tensor(np.flatnonzero(accepted),dtype=torch.long,device=xyz.device)
    values=torch.as_tensor(offs[accepted],dtype=xyz.dtype,device=xyz.device)
    gs.update_delta_xyz(frame_index,previous_keyframe_id,values=values,
                        idx=ids,max_motion=maxmotion)
    return True
