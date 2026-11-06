"""CLI. Checkpoints are compulsory for neural inference; no random inference."""
import argparse,json
from pathlib import Path
import yaml,torch
from .data import frame_stream,load_intrinsics
from .network import load_model
from .pipeline import RGBDPredictiveSLAM
from .train import train_synthetic,train_official_head_synthetic
from .train_real_pose import train_real_camera

def main():
    parser=argparse.ArgumentParser(description='Neural joint camera + dynamic 4D Gaussian SLAM')
    sp=parser.add_subparsers(dest='command',required=True)
    tr=sp.add_parser('train-synthetic',help='Train BOTH heads on synthetic point sequence labels')
    tr.add_argument('--out',required=True);tr.add_argument('--steps',type=int,default=250)
    tr.add_argument('--batch',type=int,default=8);tr.add_argument('--points',type=int,default=256)
    tr.add_argument('--device',default='auto');tr.add_argument('--history',type=int,default=3)
    of=sp.add_parser('train-official-head-synthetic',help='Train camera/dynamic heads on actual official pretrained GenFlow3D')
    of.add_argument('--genflow-repo',required=True);of.add_argument('--genflow-checkpoint',required=True)
    of.add_argument('--out',required=True);of.add_argument('--steps',type=int,default=50)
    of.add_argument('--batch',type=int,default=1);of.add_argument('--points',type=int,default=8192)
    of.add_argument('--history',type=int,default=3);of.add_argument('--device',default='cuda')
    rt=sp.add_parser('train-real-camera',help='Fine-tune camera head using a separate GT training sequence')
    rt.add_argument('--data',required=True);rt.add_argument('--intrinsics',required=True)
    rt.add_argument('--checkpoint',required=True);rt.add_argument('--out',required=True)
    rt.add_argument('--epochs',type=int,default=15);rt.add_argument('--frames',type=int,default=0)
    rt.add_argument('--depth-scale',type=float,default=5000.)
    rt.add_argument('--device',default='auto');rt.add_argument('--eval-data',default=None)
    ru=sp.add_parser('run',help='Run real RGB-D with mandatory trained joint checkpoint')
    ru.add_argument('--data',required=True);ru.add_argument('--intrinsics',required=True)
    ru.add_argument('--checkpoint',required=True);ru.add_argument('--config',default='configs/joint.yaml')
    ru.add_argument('--frames',type=int,default=20);ru.add_argument('--stride',type=int,default=1)
    ru.add_argument('--depth-scale',type=float,default=5000.)
    ru.add_argument('--dynamic-mask-dir',default=None);ru.add_argument('--gt',default=None)
    ru.add_argument('--out',default='outputs/joint_run');ru.add_argument('--device',default='auto')
    args=parser.parse_args()
    device=('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device
    if args.command=='train-synthetic':
        path=train_synthetic(args.out,steps=args.steps,batch=args.batch,n=args.points,history=args.history,device=device)
        print('Trained neural checkpoint:',path);return
    if args.command=='train-official-head-synthetic':
        train_official_head_synthetic(args.out,args.genflow_repo,args.genflow_checkpoint,
            steps=args.steps,batch=args.batch,n=args.points,history=args.history,device=device)
        return
    if args.command=='train-real-camera':
        path=train_real_camera(args.data,args.intrinsics,args.checkpoint,args.out,
            epochs=args.epochs,frames=args.frames,depth_scale=args.depth_scale,
            device=device,eval_data=args.eval_data)
        print('Fine-tuned camera head saved:',path)
        return
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf8'))
    model,meta=load_model(args.checkpoint,device,config=cfg)
    if int(cfg['history_length'])!=int(meta['history']):
        raise ValueError('Checkpoint history length differs from config')
    if int(cfg['points_per_frame'])!=int(meta['n']):
        raise ValueError('Checkpoint point count differs from config')
    K=load_intrinsics(args.intrinsics)
    slam=RGBDPredictiveSLAM(K,cfg,model,device,args.out)
    gt=args.gt
    if gt is None and (Path(args.data)/'groundtruth.txt').is_file():gt=str(Path(args.data)/'groundtruth.txt')
    results=slam.run(frame_stream(args.data,args.frames,args.stride,args.depth_scale,args.dynamic_mask_dir),gt)
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
