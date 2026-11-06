"""Online native-Flow4DGS frontend adapter (prior generated BEFORE current tracking).

Reads ONLY completed camera states with frame_id < current_frame_id; does not
read current frame RGB-D to predict prior. The current RGB-D is still processed
normally by Flow4DGS's native renderer/tracker after initialization.

Requires separate trained neural checkpoint (official GenFlow or lightweight).
Requires explicit source frame time spacing in seconds; normalized Gaussian
frame indices must not be confused with physical sensor timestamps.
"""
from __future__ import annotations
from pathlib import Path
from collections import OrderedDict
from types import SimpleNamespace
import numpy as np
import torch
import yaml
from .geometry import unproject,invert,transform,is_pose
from .tracking import Past,NeuralPredictor
from .network import load_model
from .official_bridge import export_frame_packet


def _as_numpy_camera_frame(camera):
    depth=np.asarray(camera.depth,float).squeeze()
    image=camera.original_image
    if isinstance(image,torch.Tensor):image=image.detach().cpu().numpy()
    if image.ndim==3 and image.shape[0]==3:image=np.moveaxis(image,0,-1)
    image=np.asarray(image,float)
    if image.shape[:2]!=depth.shape:raise ValueError('Native RGB/depth dimensions not aligned')
    if image.max()<=1.01:image=image*255.
    image=np.asarray(np.clip(image,0,255),dtype=np.uint8)
    # Native viewpoint.motion_mask often indicates pixels allowed for static
    # tracking, so invert only when explicitly configured. Avoid guessing.
    return SimpleNamespace(rgb=image,depth=depth,dynamic_mask=None)


def _native_T(camera):
    R=camera.R.detach().cpu().numpy() if isinstance(camera.R,torch.Tensor) else np.asarray(camera.R)
    t=camera.T.detach().cpu().numpy() if isinstance(camera.T,torch.Tensor) else np.asarray(camera.T)
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=t
    if not is_pose(T):raise ValueError('Native camera state is not valid W2C SE(3)')
    return T


def cache_completed_frame(frontend, frame_index):
    """Store sampled camera-space RGB-D BEFORE native Camera.clean() drops it.

    Keep camera-space points (not world coordinates). At inference, native
    optimized camera poses are read again, so backend BA updates are respected.
    """
    options=frontend.config.get('predictive4d',{})
    if not options.get('online',False):return False
    camera=frontend.cameras.get(frame_index)
    if camera is None or camera.depth is None or camera.original_image is None:
        return False
    cache=getattr(frontend,'_predictive4d_frame_cache',None)
    if cache is None:
        cache=OrderedDict()
        frontend._predictive4d_frame_cache=cache
    if frame_index in cache:return True
    model_yaml=options.get('model_config')
    if not model_yaml:raise ValueError('predictive4d.model_config required for frame cache')
    cfg=yaml.safe_load(Path(model_yaml).read_text(encoding='utf8'))
    K=camera.intrinsic
    K=K.detach().cpu().numpy() if isinstance(K,torch.Tensor) else np.asarray(K)
    pc,rgb,_=unproject(_as_numpy_camera_frame(camera),K,int(cfg['max_points']),
                        int(cfg['pixel_stride']),int(cfg['seed'])+int(frame_index))
    if not len(pc):raise RuntimeError(f'Frame {frame_index}: no usable depth points to cache')
    cache[int(frame_index)]=(pc.copy(),rgb.copy())
    capacity=max(int(cfg['history_length'])+2,int(options.get('history_cache_frames',8)))
    while len(cache)>capacity:cache.popitem(last=False)
    return True


class OnlineNeuralRuntime:
    def __init__(self,config):
        options=config.get('predictive4d',{})
        checkpoint=options.get('checkpoint')
        if not checkpoint:raise ValueError('predictive4d.checkpoint is required for online neural tracking')
        model_yaml=options.get('model_config')
        if not model_yaml:raise ValueError('predictive4d.model_config is required')
        self.cfg=yaml.safe_load(Path(model_yaml).read_text(encoding='utf8'))
        self.dt=float(options.get('seconds_per_frame',0))
        if not 0<self.dt<=2:raise ValueError('Set predictive4d.seconds_per_frame in seconds')
        self.device=options.get('device','cuda' if torch.cuda.is_available() else 'cpu')
        self.model,self.metadata=load_model(checkpoint,self.device,config=self.cfg)
        if int(self.metadata['history'])!=int(self.cfg['history_length']):
            raise ValueError('Network history length disagrees with model config')
        self.pred=NeuralPredictor(self.model,self.cfg,self.device)

    def predict(self,frontend,viewpoint,cur_frame_idx):
        # Exclude the current frame; only already-completed native camera poses.
        L=int(self.cfg['history_length'])
        keys=sorted(int(k) for k in frontend.cameras if int(k)<int(cur_frame_idx))[-L:]
        if len(keys)<L:return False
        K=viewpoint.intrinsic
        K=K.detach().cpu().numpy() if isinstance(K,torch.Tensor) else np.asarray(K)
        history=[]
        for idx in keys:
            old=frontend.cameras[idx]
            cache=getattr(frontend,'_predictive4d_frame_cache',{})
            if idx in cache:
                pc,rgb=cache[idx]
            else:
                if old.depth is None or old.original_image is None:
                    raise RuntimeError(
                        f'Historical frame {idx} has been cleaned without caching. '
                        'Install the predictive4d FrontEnd.cleanup hook.'
                    )
                frame=_as_numpy_camera_frame(old)
                pc,rgb,_=unproject(frame,K,int(self.cfg['max_points']),
                                        int(self.cfg['pixel_stride']),int(self.cfg['seed'])+idx)
            if not len(pc):raise RuntimeError(f'No depth samples at native frame {idx}')
            T=_native_T(old)
            history.append(Past(transform(invert(T),pc),rgb,T,idx*self.dt,None))
        next_time=cur_frame_idx*self.dt
        dt=next_time-history[-1].timestamp
        T,flow,dyn,var,info=self.pred.predict(history,dt)
        out=frontend.config.get('predictive4d',{}).get('packet_dir')
        if not out:raise ValueError('predictive4d.packet_dir must be configured')
        export_frame_packet(out,cur_frame_idx,next_time,T,
                    history[-1].xyz_world,flow,dyn,var)
        return True


def predict_online_prior(frontend,cur_frame_idx,viewpoint):
    options=frontend.config.get('predictive4d',{})
    if not options.get('online',False):return False
    if not hasattr(frontend,'_predictive4d_runtime'):
        frontend._predictive4d_runtime=OnlineNeuralRuntime(frontend.config)
    return frontend._predictive4d_runtime.predict(frontend,viewpoint,cur_frame_idx)
