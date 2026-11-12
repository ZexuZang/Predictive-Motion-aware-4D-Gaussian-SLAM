"""Generate a complete native Bonn config with CV/CV+GRU and real frame times.

Run AFTER preparing the official checkout and datasets. This is deliberately
independent of notebook cell globals. No GT is used by the camera prior.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import json
import sys
import numpy as np
import yaml


def config_dict(flow: Path, data: Path, output: Path, timestamps: Path,
                acc_ckpt: Path, genflow: Path, genflow_ckpt: Path,
                mode: str, packet_dir: Path, joint_cfg: dict) -> dict:
    base = yaml.safe_load((flow/'configs/rgbd/bonn/config.yaml').read_text())
    scene = yaml.safe_load((flow/'configs/rgbd/bonn/bonn_ps_trk.yaml').read_text())
    scene.pop('inherit_from',None)
    def merge(dst,src):
        for k,v in src.items():
            if isinstance(v,dict) and isinstance(dst.get(k),dict):
                merge(dst[k],v)
            else:dst[k]=v
    merge(base,scene)
    base['Dataset']['dataset_path']=str(data.resolve())
    base['Results'].update(save_dir=str(output.resolve()),eval_rendering=True,
                           save_results=True,save_trj=True,use_gui=False,use_wandb=False)
    base['model_params'].update(flow_cam=False,flow_mask=True,flow_offset=True,
                                dynamic_model='offset',use_gmm=True,use_gmm_rot=True,
                                use_knn=True,insertion=True)
    base['predictive4d']={
       'online':True,'camera_mode':mode,'acceleration_checkpoint':str(acc_ckpt.resolve()),
       'genflow_repo':str(genflow.resolve()),'genflow_checkpoint':str(genflow_ckpt.resolve()),
       'genflow_points':8192,'pixel_stride':2,'sampling_seed':2027,
       'timestamp_file':str(timestamps.resolve()),'packet_dir':str(packet_dir.resolve()),
       'device':'cuda','history_cache_frames':10,
       'linear_threshold_mps2':0.0,'angular_threshold_radps2':1.5,
       'flow_variance_proxy_m2':.03**2,'native_joint_refine':True,
       'joint_config':joint_cfg,'max_gaussian_match_m':.15,
       'min_dynamic_probability':.65,'min_gaussian_matches':12,
       'max_gaussian_motion_m':.10,'max_prediction_gap_frames':12,
    }
    return base


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--genflow',type=Path,required=True)
    p.add_argument('--genflow-checkpoint',type=Path,required=True)
    p.add_argument('--cv-gru-checkpoint',type=Path,required=True)
    p.add_argument('--joint-config',type=Path,required=True)
    p.add_argument('--mode',choices=['cv','cv_gru'],default='cv_gru')
    p.add_argument('--packet-dir',type=Path,required=True)
    p.add_argument('--python',default=sys.executable)
    p.add_argument('--config-out',type=Path,required=True)
    args=p.parse_args()
    required=[args.data/'rgb.txt',args.data/'depth.txt',args.data/'groundtruth.txt',
              args.genflow/'models.py',args.genflow_checkpoint,args.cv_gru_checkpoint,
              args.upstream/'slam.py']
    for file in required:
        if not file.is_file():raise FileNotFoundError(file)
    # Ask native TUM parser for precise video indices (not raw rgb.txt length).
    cmd=[args.python,'-c',
        'import sys,json; from utils.dataset import TUMParser; '
        'print("NATIVE_PARSED_PATHS="+json.dumps(TUMParser(sys.argv[1]).color_paths))',
         str(args.data.resolve())]
    proc=subprocess.run(cmd,cwd=args.upstream,capture_output=True,text=True)
    if proc.returncode:raise RuntimeError(proc.stderr[-3000:])
    lines=[s for s in proc.stdout.splitlines() if s.startswith('NATIVE_PARSED_PATHS=')]
    if len(lines)!=1:raise RuntimeError('Native parser did not return exactly one path list')
    paths=json.loads(lines[0].split('=',1)[1]);
    raw=[line.split() for line in (args.data/'rgb.txt').read_text().splitlines()
         if line.strip() and not line.lstrip().startswith('#')]
    time_by_path={str((args.data/row[1]).resolve()):float(row[0]) for row in raw}
    ts=np.array([time_by_path[str(Path(v).resolve())] for v in paths],dtype=float)
    if len(ts)<6 or not np.isfinite(ts).all() or not np.all(np.diff(ts)>0):
        raise ValueError('Native RGB timestamp sequence is invalid')
    args.output.mkdir(parents=True,exist_ok=True)
    timestamps=args.output/(args.mode+'_native_input_timestamps.npy')
    np.save(timestamps,ts)
    joint_cfg=yaml.safe_load(args.joint_config.read_text())
    joint_cfg['uncertainty_calibrated']=False
    joint_cfg.update(max_points=8192,points_per_frame=8192,history_length=3)
    config=config_dict(args.upstream,args.data,args.output,timestamps,
                       args.cv_gru_checkpoint,args.genflow,args.genflow_checkpoint,
                       args.mode,args.packet_dir,joint_cfg)
    args.config_out.parent.mkdir(parents=True,exist_ok=True)
    args.config_out.write_text(yaml.safe_dump(config,sort_keys=False))
    print(json.dumps({'frames':len(ts),'timestamp_file':str(timestamps),
                      'config':str(args.config_out),'mode':args.mode},indent=2))

if __name__=='__main__':main()
