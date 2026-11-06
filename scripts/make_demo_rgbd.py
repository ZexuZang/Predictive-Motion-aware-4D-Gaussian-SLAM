"""Generate a minimal association-based RGB-D sequence to test full inference.

Synthetic camera is stationary; this tests code integration, not estimator quality.
"""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image

p=argparse.ArgumentParser();p.add_argument('--out',default='data/demo_rgbd');p.add_argument('--frames',type=int,default=8)
a=p.parse_args();root=Path(a.out);(root/'rgb').mkdir(parents=True,exist_ok=True);(root/'depth').mkdir(parents=True,exist_ok=True)
H,W=60,80
assoc=[];gt=[]
for i in range(a.frames):
    t=1.0+i*.1
    yy,xx=np.mgrid[:H,:W]
    depth=(2.5+.002*xx+.002*yy).astype(np.float32)
    moving=(xx>(i*2+12))&(xx<(i*2+26))&(yy>20)&(yy<43)
    depth[moving]=1.6
    rgb=np.stack([xx*2+20,yy*3+20,np.full_like(xx,100)],axis=-1).clip(0,255).astype(np.uint8)
    rgb[moving]=[230,30,60]
    stem=f'{t:.6f}'
    Image.fromarray(rgb).save(root/'rgb'/f'{stem}.png')
    Image.fromarray(np.uint16(depth*5000)).save(root/'depth'/f'{stem}.png')
    assoc.append(f'{stem} rgb/{stem}.png {stem} depth/{stem}.png')
    gt.append(f'{stem} 0 0 0 0 0 0 1')
    
    # Ground-truth moving-object masks for optional controlled smoke tests only.
    (root/'dynamic_masks').mkdir(exist_ok=True)
    Image.fromarray(np.uint8(moving)*255).save(root/'dynamic_masks'/f'{stem}.png')
(root/'associations.txt').write_text('\n'.join(assoc)+'\n')
(root/'groundtruth.txt').write_text('\n'.join(gt)+'\n')
(root/'intrinsics.yaml').write_text('fx: 75\nfy: 75\ncx: 39.5\ncy: 29.5\n')
print('Demo saved to',root)
