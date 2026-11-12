"""Native Flow4DGS frontend: causal CV+GRU camera and official GenFlow3D flow.

No compact surrogate network, no frame-t GT in prediction, no random-trained scene heads.
Important limitations: motion covariance proxies are NOT calibrated; native
Flow4DGS backend still carries its original RAFT-derived fallback and renderer.
"""
from __future__ import annotations
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import torch
from .geometry import transform, invert, is_pose, se3_exp, se3_log
from .cv_acceleration import CVAccelerationRuntime
from .tracking import Past
from .geometry import unproject
from .official_bridge import export_frame_packet
from .flow4dgs_native import apply_camera_packet
from .native_pose_compat import native_camera_pose as _native_T


def _frame_of(camera):
    """All arrays are completed historical native frames, not future RGB-D."""
    if camera.depth is None or camera.original_image is None:
        raise RuntimeError('Native historical RGB-D already cleaned without cache')
    depth = camera.depth
    if isinstance(depth, torch.Tensor): depth = depth.detach().cpu().numpy()
    depth = np.asarray(depth, dtype=np.float32).squeeze()
    rgb = camera.original_image
    if isinstance(rgb, torch.Tensor): rgb = rgb.detach().cpu().numpy()
    rgb = np.asarray(rgb)
    if rgb.ndim == 3 and rgb.shape[0] == 3: rgb = np.moveaxis(rgb, 0, -1)
    if rgb.shape[:2] != depth.shape: raise RuntimeError('RGB depth shapes mismatch')
    if rgb.max() <= 1.01: rgb = rgb * 255
    rgb = np.asarray(np.clip(rgb, 0, 255), dtype=np.uint8)
    # Flow4DGS motion_mask marks pixels allowed for static tracking.
    if not hasattr(camera,'motion_mask') or camera.motion_mask is None:
        raise RuntimeError('Native static-motion mask missing: no fake dynamic prior')
    mask = camera.motion_mask
    if isinstance(mask, torch.Tensor): mask = mask.detach().cpu().numpy()
    mask = np.asarray(mask, dtype=bool).squeeze()
    if mask.shape != depth.shape: raise RuntimeError('Native motion mask shape mismatch')
    return SimpleNamespace(rgb=rgb, depth=depth, dynamic_mask=~mask)


def cache_completed_native(frontend, frame_idx):
    opts = frontend.config.get('predictive4d', {})
    if not opts.get('online', False): return False
    camera = frontend.cameras.get(frame_idx)
    if camera is None: return False
    cache = getattr(frontend, '_paper_cv_gru_cache', None)
    if cache is None:
        cache = OrderedDict()
        frontend._paper_cv_gru_cache = cache
    if frame_idx in cache: return True
    n = int(opts.get('genflow_points', 8192))
    K = camera.intrinsic
    if isinstance(K,torch.Tensor): K = K.detach().cpu().numpy()
    xyz,rgb,mask = unproject(_frame_of(camera), np.asarray(K),
                            max_points=n, pixel_stride=int(opts.get('pixel_stride',2)),
                            seed=int(opts.get('sampling_seed',2027))+int(frame_idx))
    if len(xyz) != n:
        raise RuntimeError(f'GenFlow3D needs {n} real depth samples, got {len(xyz)} at {frame_idx}')
    cache[int(frame_idx)] = (xyz, rgb, mask.astype(bool))
    cap = max(9, int(opts.get('history_cache_frames',10)))
    while len(cache) > cap: cache.popitem(last=False)
    return True


class NativePaperRuntime:
    def __init__(self, config):
        opts = config.get('predictive4d', {})
        self.opts = opts
        self.device = opts.get('device','cuda')
        if self.device.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('Native full pipeline requires CUDA GPU')
        timing_path=Path(opts['timestamp_file'])
        if not timing_path.is_file(): raise FileNotFoundError('Need actual dataset timestamps .npy')
        self.timestamps=np.load(timing_path,allow_pickle=False).astype(float)
        if (len(self.timestamps)<10 or not np.isfinite(self.timestamps).all() or
                np.any(np.diff(self.timestamps)<=0)):
            raise ValueError('Invalid per-frame timestamps')
        self.cv = CVAccelerationRuntime(opts['acceleration_checkpoint'], self.device,
             linear_threshold=float(opts.get('linear_threshold_mps2',0.0)),
             angular_threshold=float(opts.get('angular_threshold_radps2',1.5)))
        self.history = self.cv.history_poses
        if self.history != 6:
            raise RuntimeError('Expected six-pose GRU trained checkpoint')
        repo = Path(opts['genflow_repo']).resolve()
        checkpoint = Path(opts['genflow_checkpoint']).resolve()
        if not repo.joinpath('models.py').is_file() or not checkpoint.is_file():
            raise FileNotFoundError('Official GenFlow3D source/checkpoint required')
        import sys, importlib
        sys.path.insert(0,str(repo))
        mod = importlib.import_module('models')
        self.points = int(opts.get('genflow_points',8192))
        self.flow = mod.GenFlow_rec_seq(self.points).to(self.device).eval()
        weights = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if isinstance(weights, dict) and 'state_dict' in weights: weights=weights['state_dict']
        if isinstance(weights, dict) and 'model' in weights and isinstance(weights['model'],dict): weights=weights['model']
        if any(k.startswith('module.') for k in weights):
            weights={k.removeprefix('module.'):v for k,v in weights.items()}
        self.flow.load_state_dict(weights,strict=True)
        self.frame_records=[]
        self._logged_frames=set()
        self.out = Path(opts['packet_dir'])
        self.out.mkdir(parents=True,exist_ok=True)

    @torch.inference_mode()
    def predict(self, frontend, viewpoint, idx):
        if idx>=len(self.timestamps): raise RuntimeError('Native frame exceeds input timestamp array')
        all_previous = sorted(int(k) for k in frontend.cameras if int(k)<int(idx))
        indices = all_previous[-self.history:]
        if len(indices)<self.history:
            return False
        cache = getattr(frontend,'_paper_cv_gru_cache',{})
        states=[]; cloud_history=[]; previous_mask=None
        for n in indices:
            cam = frontend.cameras[n]
            if n in cache: xyz,rgb,mask = cache[n]
            else:
                K=cam.intrinsic
                if isinstance(K,torch.Tensor):K=K.detach().cpu().numpy()
                xyz,rgb,mask=unproject(_frame_of(cam),np.asarray(K),
                    max_points=self.points,pixel_stride=int(self.opts.get('pixel_stride',2)),
                    seed=int(self.opts.get('sampling_seed',2027))+n)
                if len(xyz)!=self.points:raise RuntimeError('Missing real input points for native GenFlow')
            Tcw = _native_T(cam)
            if not is_pose(Tcw): raise RuntimeError(f'Invalid native camera {n} W2C')
            world = transform(invert(Tcw),xyz).astype(np.float32)
            states.append(Past(world,rgb,Tcw,float(self.timestamps[n]),mask))
            cloud_history.append(world)
            previous_mask=mask
        mode=self.opts.get('camera_mode','cv_gru')
        if mode=='cv_gru':
            dt=float(self.timestamps[idx]-self.timestamps[indices[-1]])
            prior, info=self.cv.predict(states,dt)
        elif mode=='cv':
            prev,prevprev=states[-1].T_cw,states[-2].T_cw
            xi=se3_log(prev@invert(prevprev))
            dt=float(self.timestamps[idx]-self.timestamps[indices[-1]])
            last_dt=float(self.timestamps[indices[-1]]-self.timestamps[indices[-2]])
            prior=se3_exp(xi*(dt/last_dt))@prev
            info={'camera_prior_mode':'cv','acceleration_status':'disabled'}
        else:
            raise ValueError(f'Unsupported camera_mode={mode}')
        # Official GenFlow3D forward: L observed world point clouds, one future flow.
        # Crucially, the last flow is forecast from the LAST OBSERVED frame to t.
        input_clouds=[torch.from_numpy(q[None]).to(self.device) for q in cloud_history[-3:]]
        dummy=[torch.zeros_like(input_clouds[0]) for _ in range(3)]
        out,_,_=self.flow(input_clouds,input_clouds,dummy)
        if not isinstance(out,(tuple,list)) or len(out)!=4:
            raise RuntimeError('Unexpected official GenFlow3D decoder output schema')
        v=out[0][-1]
        if v.shape==(1,3,self.points):v=v.transpose(1,2)
        if v.shape!=(1,self.points,3):
            raise RuntimeError(f'Unexpected GenFlow flow layout {tuple(v.shape)}')
        flow_world=v[0].detach().cpu().numpy().astype(np.float32)
        if not np.isfinite(flow_world).all():raise RuntimeError('Nonfinite official flow')
        # Existing native dynamic/static mask is the upstream RAFT/YOLO result
        # for frame idx-1, never GT or synthetic learned dynamic logits.
        prob = np.where(previous_mask.astype(bool),.99,.01).astype(np.float32)
        # IMPORTANT: this variance is a PROXY, not uncertainty calibration.
        var = np.full_like(flow_world,float(self.opts.get('flow_variance_proxy_m2',.03**2)))
        packet=export_frame_packet(self.out,idx,float(self.timestamps[idx]),prior,
                  cloud_history[-1],flow_world,prob,var)
        info.update({'frame':int(idx),'pose_source':('CVAccelerationGRU-v1' if mode=='cv_gru' else 'CV'),
                    'scene_source':'official_GenFlow_rec_seq','packet':str(packet),
                    'dynamic_mask_source':'native_previous_motion_mask',
                    'uncertainty_calibrated':False,'flow_max_m':float(np.linalg.norm(flow_world,axis=1).max())})
        # A single activation record per frame (not the number of retries).
        if int(idx) not in self._logged_frames:
            with (self.out/'runtime_diagnostics.jsonl').open('a') as f:
                f.write(json.dumps(info)+'\n')
            self._logged_frames.add(int(idx))
        return True


def predict_native_paper(frontend,cur_frame_idx,viewpoint):
    opts=frontend.config.get('predictive4d',{})
    if not opts.get('online',False):return False
    if not hasattr(frontend,'_paper_cv_gru_runtime'):
        frontend._paper_cv_gru_runtime=NativePaperRuntime(frontend.config)
    success=frontend._paper_cv_gru_runtime.predict(frontend,viewpoint,cur_frame_idx)
    if success:
        ok=apply_camera_packet(frontend.config,viewpoint,cur_frame_idx,frontend)
        if not ok:raise RuntimeError('Native camera packet did not apply')
    return success
