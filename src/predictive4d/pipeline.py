"""Causal RGB-D geometric/4D Gaussian-primitive research pipeline.

Provides every STEP as runnable code, but baseline H/V motion, NN association,
and CPU splatting are deliberately distinct from trained GenFlow3D/Flow4DGS CUDA.
"""
from __future__ import annotations
from collections import deque
from pathlib import Path
import json,time
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from .geometry import unproject,transform,inverse,predict_pose
from .association import match_prev_to_current
from .prediction import build_motion_backend
from .pose_prior import build_pose_prior
from .motion import estimate_pose_and_prob,correct_flow
from .gaussian import GaussianMap,transfer_motion
from .render import save_preview,refine_pose_render
from .evaluate import write_trajectory,evaluate_ate
from .feedback import refine_pose_with_dynamic_gaussians
from .depth_motion import foreground_depth_innovation


class RGBDPredictiveSLAM:
    def __init__(self,K,cfg,outdir):
        self.K=np.asarray(K,float);self.cfg=cfg;self.outdir=Path(outdir);self.outdir.mkdir(parents=True,exist_ok=True)
        self.rng=np.random.default_rng(42);self.motion=build_motion_backend(cfg)
        self.camera_predictor=build_pose_prior(cfg);self.history_camera=deque(maxlen=5)
        self.poses=[];self.times=[];self.history=deque(maxlen=5)
        self.prev_world=None;self.prev_static_prior=None
        self.static_map=GaussianMap.from_points(np.empty((0,3)),np.empty((0,3)))
        self.dynamic_map=GaussianMap.from_points(np.empty((0,3)),np.empty((0,3)))
        self.records=[]

    def _new_map(self,world,color,num):
        if not len(world):return GaussianMap.from_points(np.empty((0,3)),np.empty((0,3)))
        indices=self.rng.choice(len(world),min(num,len(world)),replace=False)
        return GaussianMap.from_points(world[indices],color[indices],scale=.018)

    def _make_preview(self,idx,frame,T):
        if int(self.cfg['preview_interval'])>0 and idx%int(self.cfg['preview_interval'])==0:
            save_preview(self.outdir/'previews'/f'{idx:06d}.png',self.static_map,
                         self.dynamic_map,T,self.K,frame.depth.shape,
                         width=int(self.cfg['render_width']),height=int(self.cfg['render_height']))

    def _snapshot(self,idx):
        every=int(self.cfg['snapshot_interval'])
        if every and idx%every==0:
            self.dynamic_map.save(self.outdir/'dynamic_maps'/f'{idx:06d}.npz')
            self.static_map.save(self.outdir/'static_maps'/f'{idx:06d}.npz')

    def process(self,frame):
        ix=len(self.poses);beg=time.perf_counter()
        P,uv=unproject(frame.depth,self.K,stride=int(self.cfg['sample_stride']),
                       max_points=int(self.cfg['max_points']),rng=self.rng)
        if len(P)<30:raise RuntimeError(f'Frame {ix} has insufficient valid depth')
        colors=frame.rgb[uv[:,0],uv[:,1]].astype(float)/255.
        if ix==0:
            T=np.eye(4);world=P.copy()
            if frame.dynamic_mask is not None:
                dm=frame.dynamic_mask[uv[:,0],uv[:,1]]
                self.static_map=self._new_map(world[~dm],colors[~dm],int(self.cfg['initial_gaussians']))
                self.dynamic_map=self._new_map(world[dm],colors[dm],int(self.cfg['initial_gaussians']//2))
                prior=np.where(dm,.02,float(self.cfg['static_prior']))
            else:
                self.static_map=self._new_map(world,colors,int(self.cfg['initial_gaussians']))
                prior=np.full(len(world),float(self.cfg['static_prior']))
            self.prev_world=world;self.prev_static_prior=prior
            self.poses.append(T);self.times.append(frame.timestamp);self.history.append(world);self.history_camera.append(P)
            self._snapshot(ix);self._make_preview(ix,frame,T)
            self.records.append({'frame':ix,'matched':0,'dynamic_ratio':float(np.mean(frame.dynamic_mask[uv[:,0],uv[:,1]])) if frame.dynamic_mask is not None else 0.,'status':'bootstrap','seconds':time.perf_counter()-beg})
            return self.records[-1]

        T_prior,pose_cov=self.camera_predictor.predict(self.poses,self.history_camera)
        history_points=list(self.history)
        try:
            pred_flow,pred_unc=self.motion.predict(history_points,self.prev_world)
        except Exception as exc:
            raise RuntimeError(f'Motion backend failed at frame={ix}') from exc
        pred_flow=np.asarray(pred_flow,float)
        if pred_flow.shape!=self.prev_world.shape:raise ValueError('Flow backend must return one 3D vector per previous point')
        a,b,dist=match_prev_to_current(self.prev_world,P,T_prior,pred_flow,
                                      max_dist=float(self.cfg['max_match_distance']))
        if len(a)<int(self.cfg['min_matches']):
            T=T_prior.copy();gam=np.empty((0,3));status='tracking_fallback'
        else:
            prior_static=np.clip(self.prev_static_prior[a],.05,.95)
            # Approximate mean static prior for current matches; proper per-point semantic
            # priors could be added here with a full calibrated E-step implementation.
            T,gam=estimate_pose_and_prob(self.prev_world[a],P[b],pred_flow[a],T_prior,
                   prior_pose=T_prior,prior_sigma=float(self.cfg['prior_sigma']),
                   obs_std=float(self.cfg['obs_std']),flow_std=float(self.cfg['flow_std']),
                   p_static=float(np.mean(prior_static)),outer=int(self.cfg['outer_iterations']))
            status='tracking_ok'
        # Strictly history-derived dynamic Gaussian hypothesis for optional pose feedback.
        dynamic_pred=transfer_motion(self.dynamic_map,self.prev_world,pred_flow,k=12,radius=.18)
        if bool(self.cfg.get('dynamic_feedback_enabled',False)) and len(a):
            T,feedback_diag=refine_pose_with_dynamic_gaussians(
                T,dynamic_pred,P,static_world=self.static_map.mean,
                gate_when_static_weak=bool(self.cfg.get('feedback_only_weak_static',False)))
        else:feedback_diag={'accepted':0,'status':'disabled'}
        if len(self.static_map)>30 and bool(self.cfg.get('render_pose_refinement',False)):
            T=refine_pose_render(T,self.static_map,frame,self.K,
                render_width=int(self.cfg['render_width']),render_height=int(self.cfg['render_height']),
                max_nfev=int(self.cfg['render_refine_max_nfev']),enabled=True)
        world=transform(inverse(T),P)
        static_prob=np.full(len(P),float(self.cfg['static_prior']))
        dynamic_prob=np.zeros(len(P))
        observed_flow=np.zeros_like(P)
        if len(a):
            static_prob[b]=gam[:,0]
            dynamic_prob[b]=gam[:,1]
            Vcorr=correct_flow(self.prev_world[a],P[b],T,pred_flow[a],
                    obs_var=float(self.cfg['obs_std'])**2,
                    flow_var=float(self.cfg['flow_std'])**2)
            # Dynamic-only output: static points cannot have free world motion.
            observed_flow[b]=Vcorr*gam[:,1,None]
        # Optional depth reprojection innovation, primarily for foreground arrivals;
        # coarse, noisy and explicitly NOT a learned replacement for BDCB.
        try:
            depth_new=foreground_depth_innovation(self.prev_world,T,P,uv,self.K,frame.depth.shape,
                                                 threshold=float(self.cfg.get('depth_motion_threshold',.22)))
            dynamic_prob[depth_new]=np.maximum(dynamic_prob[depth_new],.8)
            static_prob[depth_new]=np.minimum(static_prob[depth_new],.04)
        except (ValueError,IndexError):pass
        # Optional trusted semantic / PreOurs mask for controlled ablations.
        if frame.dynamic_mask is not None:
            dm=frame.dynamic_mask[uv[:,0],uv[:,1]]
            dynamic_prob[dm]=np.maximum(dynamic_prob[dm],.99)
            static_prob[dm]=np.minimum(static_prob[dm],.005)
        # Predict dynamic Gaussian map using only PREVIOUS points and motion prior.
        if len(self.dynamic_map):
            self.dynamic_map=dynamic_pred
            # Optional observation correction for associated current dynamic points.
            dyn=np.flatnonzero(dynamic_prob>float(self.cfg['dynamic_confidence_threshold']))
            if len(dyn):
                tree=cKDTree(world[dyn]);d,j=tree.query(self.dynamic_map.mean)
                accepted=d<.12
                self.dynamic_map.mean[accepted]=.75*self.dynamic_map.mean[accepted]+.25*world[dyn[j[accepted]]]
        # Integrate sampled current static points only if confidently static or unseen;
        # beware: unmatched moving regions are not perfectly classified in this baseline.
        sids=np.flatnonzero(static_prob>float(self.cfg['static_confidence_threshold']))
        if len(sids):
            self.static_map.append(self._new_map(world[sids],colors[sids],int(self.cfg['append_gaussians'])),
                            max_size=int(self.cfg['max_static_gaussians']))
        dids=np.flatnonzero(dynamic_prob>float(self.cfg['dynamic_confidence_threshold']))
        if len(dids):
            self.dynamic_map.append(self._new_map(world[dids],colors[dids],int(self.cfg['append_gaussians'])),
                             max_size=int(self.cfg['max_dynamic_gaussians']))
        self.motion.update(world,observed_flow)
        self.prev_world=world.copy();self.prev_static_prior=static_prob.copy()
        self.history.append(world.copy());self.history_camera.append(P.copy())
        self.poses.append(T);self.times.append(float(frame.timestamp))
        self._snapshot(ix);self._make_preview(ix,frame,T)
        self.records.append({'frame':ix,'matched':int(len(a)),
            'dynamic_ratio':float(np.mean(dynamic_prob>float(self.cfg['dynamic_confidence_threshold']))),
            'static_gaussians':int(len(self.static_map)),
            'dynamic_gaussians':int(len(self.dynamic_map)),
            'pose_translation_cw':list(map(float,T[:3,3])),
            'status':status,'dynamic_feedback':feedback_diag,'seconds':time.perf_counter()-beg})
        return self.records[-1]

    def run(self,frames,gt_path=None):
        for frame in tqdm(frames,desc='Predictive4D'):
            rec=self.process(frame)
            tqdm.write(f"frame={rec['frame']:03d} status={rec['status']} matched={rec['matched']} dyn={rec['dynamic_ratio']:.3f}")
        write_trajectory(self.outdir/'trajectory_est.tum',self.times,self.poses)
        if self.poses:
            np.savez_compressed(self.outdir/'camera_poses.npz',T_cw=np.stack(self.poses),
                                timestamps=np.asarray(self.times))
        evalr=evaluate_ate(self.times,self.poses,gt_path) if gt_path else {'status':'No GT provided'}
        metrics={'frames':len(self.times),'eval':evalr,'mean_frame_seconds':float(np.mean([r['seconds'] for r in self.records])),
                 'final_static_gaussians':len(self.static_map),'final_dynamic_gaussians':len(self.dynamic_map),
                 'scope':'Reference CPU point-Gaussian map, not official optimized Flow4DGS backend'}
        (self.outdir/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
        (self.outdir/'frame_records.json').write_text(json.dumps(self.records,indent=2),encoding='utf8')
        return metrics
