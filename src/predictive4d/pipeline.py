"""Unified 4-part neural SLAM pipeline: joint prediction, tracking, feedback, 4D map.

Strictly causal: no current RGBD enters network prior, no evaluation GT enters tracking.
"""
from __future__ import annotations
import json,time
from collections import deque
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
from .geometry import unproject,invert,transform,is_pose,nearest_unique
from .tracking import Past,NeuralPredictor,dynamic_pose_feedback
from .joint_ba import optimize_joint_pose
from .gaussians import GaussianMapper,render_preview
from .evaluate import evaluate_ate,write_trajectory
from .official_bridge import export_frame_packet
from .torch_renderer import optimize_gaussian_keyframe

class RGBDPredictiveSLAM:
    def __init__(self,K,cfg,network,device,out):
        self.K=np.asarray(K,float);self.cfg=cfg;self.out=Path(out);self.out.mkdir(parents=True,exist_ok=True)
        self.predictor=NeuralPredictor(network,cfg,device);self.mapper=GaussianMapper(cfg)
        self.history=deque(maxlen=max(5,int(cfg['history_length'])))
        self.poses=[];self.timestamps=[];self.records=[];self.frames_ok=0

    def process(self,frame):
        ix=len(self.poses);t0=time.perf_counter()
        camera,rgb,mask=unproject(frame,self.K,int(self.cfg['max_points']),int(self.cfg['pixel_stride']),int(self.cfg['seed'])+ix)
        if len(camera)<40:raise RuntimeError(f'Frame {ix}: insufficient depth points')
        if ix==0:
            T=np.eye(4);world=camera.copy();probs=mask.astype(float) if mask is not None else np.zeros(len(camera))+.05
            self.mapper.integrate(world,rgb,probs,self.mapper.dynamic,ix)
            record={'frame':ix,'status':'bootstrap','pose_source':'identity_initialization','matched':0,
                    'feedback_status':'not_applicable','feedback_accepted':False,'dynamic_ratio':float((probs>.6).mean())}
        else:
            prev=self.history[-1];dt=float(frame.timestamp-prev.timestamp)
            if not np.isfinite(dt) or dt<=0:raise ValueError('Nonmonotonic RGBD timestamps')
            if len(self.history)<int(self.cfg['history_length']):
                T_prior=prev.T_cw.copy();flow=np.zeros_like(prev.xyz_world)
                dyn_prob=prev.dynamic_mask.astype(float) if prev.dynamic_mask is not None else np.zeros(len(flow))+.05
                flow_var=np.ones((len(flow),3))*.03**2
                netdiag={'pose_accepted':False,'status':'warmup_no_future_information'}
                pose_source='warmup_geometry'
            else:
                T_prior,flow,dyn_prob,flow_var,netdiag=self.predictor.predict(list(self.history),dt)
                pose_source='neural_shared_encoder'
            if prev.dynamic_mask is not None:
                dyn_prob=np.maximum(dyn_prob,prev.dynamic_mask.astype(float)*.99)
            if not all(np.isfinite(a).all() for a in (T_prior,flow,dyn_prob,flow_var)):
                raise FloatingPointError(f'Nonfinite predictive state at frame {ix}')
            # Predict map strictly from historical information before assimilation.
            predicted_dynamic=self.mapper.predict_dynamic(prev.xyz_world,flow,flow_var)
            if self.cfg.get('export_official_packets',False) and pose_source=='neural_shared_encoder':
                export_frame_packet(self.out/'official_packets',ix,frame.timestamp,T_prior,
                                    prev.xyz_world,flow,dyn_prob,flow_var)
            T,posteriors,used,track_diag=optimize_joint_pose(
                T_prior,prev.T_cw,prev.xyz_world,flow,dyn_prob,flow_var,camera,self.cfg)
            T2,feeddiag=dynamic_pose_feedback(T,prev.T_cw,camera,self.mapper.static.mean,used,predicted_dynamic,self.cfg)
            T=T2
            if not is_pose(T):raise FloatingPointError('Estimated transform is not SE3')
            tracked=(track_diag['status']=='joint_ok' or feeddiag['accepted'])
            # Keep a held pose on loss; DO NOT integrate an uncertain frame into map.
            if not tracked:T=prev.T_cw.copy();pose_source='held_pose_tracking_lost'
            else:self.frames_ok+=1
            world=transform(invert(T),camera)
            probs=np.zeros(len(camera))+.05
            # Causal association and motion-aware posterior, not labelled GT masks.
            predicted=transform(T,prev.xyz_world+flow)
            ia,ib,d=nearest_unique(predicted,camera,float(self.cfg['max_match_distance']))
            if len(posteriors):
                probs[used]=np.maximum(probs[used],posteriors[:,1])
            if len(ia):
                residual_static=np.linalg.norm(transform(T,prev.xyz_world[ia])-camera[ib],axis=1)
                # Residual evidence is gated by network dynamic probability.
                # A motion hypothesis must explain observations BETTER than
                # the static hypothesis; network confidence alone is insufficient.
                motion_advantage=(residual_static-d)
                reliable_moving=(residual_static>.08)&(motion_advantage>.035)&(
                    np.linalg.norm(flow[ia],axis=1)>.05)&(d<.12)
                evidence=np.where(reliable_moving,np.maximum(dyn_prob[ia],.82),dyn_prob[ia])
                probs[ib]=np.clip(evidence,0,1)
            if mask is not None:probs[mask]=.99
            if tracked:self.mapper.integrate(world,rgb,probs,predicted_dynamic,ix)
            mapping_diag={'status':'not_keyframe'}
            if tracked and ix%max(1,int(self.cfg.get('mapping_opt_interval',4)))==0:
                mapping_diag=optimize_gaussian_keyframe(self.mapper,frame,T,self.K,
                    self.predictor.device,steps=int(self.cfg.get('mapping_opt_steps',0)),
                    height=int(self.cfg.get('mapping_opt_height',72)),
                    width=int(self.cfg.get('mapping_opt_width',96)),
                    max_static=int(self.cfg.get('mapping_opt_max_static',64)),
                    max_dynamic=int(self.cfg.get('mapping_opt_max_dynamic',32)))
            # If lost: update predicted dynamic state? freeze map instead to avoid drift.
            record={'frame':ix,'status':'tracking_ok' if tracked else 'tracking_lost',
                    'pose_source':pose_source,'matched':int(track_diag.get('matches',track_diag.get('matched',0))),
                    'feedback_status':feeddiag['status'],'feedback_accepted':bool(feeddiag['accepted']),
                    'feedback_dynamic_matches':int(feeddiag['dynamic_matches']),
                    'dynamic_ratio':float((probs>.6).mean()),
                    'prediction':netdiag,'tracking':track_diag,'mapping':mapping_diag,
                    'static_gaussians':len(self.mapper.static),'dynamic_gaussians':len(self.mapper.dynamic)}
        if ix==0:world=camera.copy()
        self.poses.append(T.copy());self.timestamps.append(float(frame.timestamp))
        self.history.append(Past(world.copy(),rgb.copy(),T.copy(),float(frame.timestamp),mask.copy() if mask is not None else None))
        rec=record
        rec['seconds_process']=float(time.perf_counter()-t0)
        self.records.append(rec)
        interval=int(self.cfg['snapshot_interval']);preview=int(self.cfg['preview_interval'])
        if interval>0 and ix%interval==0:self.mapper.snapshot(self.out,ix)
        if preview>0 and ix%preview==0:
            im=render_preview(self.mapper.static,self.mapper.dynamic,T,self.K,frame.depth.shape[0],
                frame.depth.shape[1],int(self.cfg['preview_width']),int(self.cfg['preview_height']))
            path=self.out/'previews'/f'{ix:06d}.png';path.parent.mkdir(parents=True,exist_ok=True);im.save(path)
        return rec

    def run(self,frames,gt_path=None):
        for frame in tqdm(frames,desc='NeuralPredictive4D'):
            rec=self.process(frame)
            tqdm.write(f"frame={rec['frame']:03d} {rec['status']} matches={rec['matched']} "
                f"dyn={rec['dynamic_ratio']:.3f} feedback={rec['feedback_status']}")
        write_trajectory(self.out/'trajectory_est.tum',self.timestamps,self.poses)
        np.savez_compressed(self.out/'camera_poses.npz',T_cw=np.stack(self.poses),timestamps=np.asarray(self.timestamps))
        eva=evaluate_ate(self.timestamps,self.poses,gt_path) if gt_path else {'status':'no_gt'}
        metrics={'frames':len(self.poses),'tracking_ok':self.frames_ok,'tracking_lost':sum(x['status']=='tracking_lost' for x in self.records),
                 'dynamic_feedback_accepted':sum(bool(x.get('feedback_accepted')) for x in self.records),
                 'eval':eva,'mean_process_s':float(np.mean([x['seconds_process'] for x in self.records])),
                 'static_gaussians':len(self.mapper.static),'dynamic_gaussians':len(self.mapper.dynamic),
                 'scope':'Neural joint heads + causal feedback + torch differentiable low-res 4D Gaussian mapping; NOT native Flow4DGS CUDA backend'}
        (self.out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
        (self.out/'frame_records.json').write_text(json.dumps(self.records,indent=2),encoding='utf8')
        return metrics
