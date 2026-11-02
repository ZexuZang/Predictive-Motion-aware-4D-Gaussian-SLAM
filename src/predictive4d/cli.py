"""Command-line interface for synthetic smoke test and Bonn RGB-D sequence."""
import argparse
import json
from pathlib import Path
import yaml
from .synthetic import generate
from .data import frame_stream,load_intrinsics,find_frames
from .pipeline import RGBDPredictiveSLAM


def main():
    p=argparse.ArgumentParser(description='Predictive 4D Gaussian SLAM research reference')
    sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('make-demo',help='Create synthetic RGB-D data for smoke test')
    s.add_argument('--out',default='data/synthetic_demo');s.add_argument('--frames',type=int,default=7)
    r=sub.add_parser('run',help='Run on RGB-D dataset')
    r.add_argument('--data',required=True);r.add_argument('--config',default='configs/reference.yaml')
    r.add_argument('--intrinsics',required=True,help='YAML with fx fy cx cy or K')
    r.add_argument('--depth-scale',type=float,default=5000.,help='uint16 depth raw units per metre (Bonn/TUM often 5000)')
    r.add_argument('--frames',type=int,default=0);r.add_argument('--stride',type=int,default=1)
    r.add_argument('--out',default='outputs/run0');r.add_argument('--gt',default=None)
    r.add_argument('--dynamic-mask-dir',default=None,help='Optional PNG masks, for Pre-Ours/semantic ablations')
    args=p.parse_args()
    if args.command=='make-demo':
        intr=generate(args.out,args.frames)
        print('Generated:',args.out,'intrinsics:',intr);return
    with open(args.config,encoding='utf8') as f:cfg=yaml.safe_load(f)
    if args.depth_scale<=0:raise ValueError('--depth-scale must be positive')
    first=next(frame_stream(args.data,max_frames=1,depth_scale=args.depth_scale))
    K=load_intrinsics(args.intrinsics,first.rgb.shape[1],first.rgb.shape[0])
    slam=RGBDPredictiveSLAM(K,cfg,args.out)
    gt=args.gt
    if gt is None and (Path(args.data)/'groundtruth.txt').exists():gt=str(Path(args.data)/'groundtruth.txt')
    result=slam.run(frame_stream(args.data,max_frames=args.frames,stride=args.stride,
                    depth_scale=args.depth_scale,dynamic_mask_dir=args.dynamic_mask_dir),gt)
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__': main()
