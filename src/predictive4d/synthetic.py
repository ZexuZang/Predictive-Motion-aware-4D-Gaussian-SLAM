"""Tiny synthetic planar RGB-D sequence for SMOKE TEST only, not benchmark."""
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation


def generate(root,nframes=7,h=72,w=96,scale=5000):
    root=Path(root);(root/'rgb').mkdir(parents=True,exist_ok=True);(root/'depth').mkdir(parents=True,exist_ok=True);(root/'dynamic_masks').mkdir(parents=True,exist_ok=True)
    fx=fy=92.;cx=(w-1)/2;cy=(h-1)/2
    (root/'intrinsics.yaml').write_text(f'fx: {fx}\nfy: {fy}\ncx: {cx}\ncy: {cy}\n')
    yy,xx=np.mgrid[:h,:w];rays_x=(xx-cx)/fx;rays_y=(yy-cy)/fy
    lines=[];gt=[]
    for i in range(nframes):
        ts=float(i)/12;camx=.018*i; depth=np.full((h,w),2.6,dtype=float)
        wall_x=camx+depth*rays_x;wall_y=depth*rays_y
        # Front-facing moving rectangle at z=1.6m
        boxx=camx+1.6*rays_x;boxy=1.6*rays_y
        mask=(np.abs(boxx-(-.28+.045*i))<.22)&(np.abs(boxy+.02)<.29)
        depth[mask]=1.6
        col=np.empty((h,w,3),dtype=np.uint8)
        col[:,:,0]=np.uint8(75+50*(np.sin(8*wall_x)**2));col[:,:,1]=np.uint8(85+35*np.sin(5*wall_y)**2)
        col[:,:,2]=np.uint8(110+30*np.cos(9*wall_x+4*wall_y)**2)
        col[mask]=[220,95,65]
        stem=f'{i:06d}';Image.fromarray(col).save(root/'rgb'/f'{stem}.png')
        Image.fromarray(np.uint16(depth*scale)).save(root/'depth'/f'{stem}.png')
        Image.fromarray((mask*255).astype(np.uint8)).save(root/'dynamic_masks'/f'{stem}.png')
        lines.append(f'{ts:.6f} rgb/{stem}.png {ts:.6f} depth/{stem}.png')
        gt.append(f'{ts:.6f} {camx:.8f} 0 0 0 0 0 1')
    (root/'associations.txt').write_text('\n'.join(lines)+'\n')
    (root/'groundtruth.txt').write_text('\n'.join(gt)+'\n')
    return root/'intrinsics.yaml'
