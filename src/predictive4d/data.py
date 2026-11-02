"""RGB-D datasets. TUM/Bonn-style text associations or matching filenames."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image

@dataclass
class RGBDFrame:
    rgb:np.ndarray # H W 3 uint8
    depth:np.ndarray # H W float32 metres
    timestamp:float
    name:str
    dynamic_mask:np.ndarray|None=None  # optional external/semantic mask for experiments


def load_intrinsics(path, width, height):
    if path is None:
        # Only a demo approximation; real datasets REQUIRE calibrated intrinsics.
        fx=fy=float(max(width,height));cx=(width-1)/2;cy=(height-1)/2
        return np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],float)
    import yaml
    spec=yaml.safe_load(Path(path).read_text(encoding='utf8'))
    if 'K' in spec:
        return np.array(spec['K'],float).reshape(3,3)
    return np.array([[spec['fx'],0,spec['cx']],[0,spec['fy'],spec['cy']],[0,0,1]],float)


def _read_list(root, name):
    fp=root/name
    if not fp.exists():return []
    records=[]
    for line in fp.read_text(encoding='utf8').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):continue
        v=line.split()
        if len(v)>=2:
            try:records.append((float(v[0]),v[1]))
            except ValueError:continue
    return records


def find_frames(root, association_tolerance=.025):
    root=Path(root)
    asso=root/'associations.txt'
    if asso.exists():
        records=[]
        for line in asso.read_text(encoding='utf8').splitlines():
            if not line.strip() or line.lstrip().startswith('#'):continue
            row=line.split()
            if len(row)>=4:
                try:records.append((float(row[0]),root/row[1],root/row[3]))
                except ValueError:pass
        if records:return records
    rlist=_read_list(root,'rgb.txt');dlist=_read_list(root,'depth.txt')
    if rlist and dlist:
        pairs=[];j=0
        for tr,pr in rlist:
            while j+1<len(dlist) and abs(dlist[j+1][0]-tr)<abs(dlist[j][0]-tr):j+=1
            td,pd=dlist[j]
            if abs(td-tr)<=association_tolerance:pairs.append((tr,root/pr,root/pd))
        if pairs:return pairs
    rdir=root/'rgb';ddir=root/'depth'
    if not rdir.exists() or not ddir.exists():raise FileNotFoundError(f'Need rgb/depth folders or rgb.txt/depth.txt in {root}')
    ext={'.png','.jpg','.jpeg'}
    rgb=sorted(p for p in rdir.iterdir() if p.suffix.lower() in ext)
    depth=sorted(p for p in ddir.iterdir() if p.suffix.lower() in ext)
    depth_names={p.stem:p for p in depth}
    matches=[(float(i),p,depth_names[p.stem]) for i,p in enumerate(rgb) if p.stem in depth_names]
    if matches:return matches
    if len(rgb)!=len(depth):raise ValueError('RGB and depth counts differ; provide associations.txt')
    return [(float(i),a,b) for i,(a,b) in enumerate(zip(rgb,depth))]


def frame_stream(root,max_frames=0,stride=1,depth_scale=5000.,dynamic_mask_dir=None):
    pairs=find_frames(root)
    if not pairs:raise RuntimeError('No RGB-D frames')
    count=0
    for stamp,r,d in pairs[::stride]:
        if max_frames and count>=max_frames:break
        if not (r.exists() and d.exists()):raise FileNotFoundError(f'{r} or {d} missing')
        rgb=np.asarray(Image.open(r).convert('RGB'),dtype=np.uint8)
        raw=np.asarray(Image.open(d))
        depth=(raw.astype(np.float32)/depth_scale)
        if rgb.shape[:2]!=depth.shape:raise ValueError(f'RGB/depth shape mismatch {r} {d}')
        mask=None
        if dynamic_mask_dir is not None:
            folder=Path(dynamic_mask_dir); candidates=[folder/f'{r.stem}.png',folder/f'{r.stem}.jpg']
            path=next((p for p in candidates if p.exists()),None)
            if path is not None:
                mask=np.asarray(Image.open(path).convert('L'))>127
                if mask.shape!=depth.shape:raise ValueError(f'Dynamic mask shape mismatch {path}')
        yield RGBDFrame(rgb,depth,float(stamp),r.stem,mask)
        count+=1


def read_tum_gt(path):
    if path is None:return []
    from scipy.spatial.transform import Rotation
    ans=[]
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith('#'):continue
        a=line.split()
        if len(a)<8:continue
        ts=float(a[0]);xyz=np.array(list(map(float,a[1:4]))); q=np.array(list(map(float,a[4:8])))
        T=np.eye(4);T[:3,:3]=Rotation.from_quat(q).as_matrix();T[:3,3]=xyz
        ans.append((ts,T))
    return ans
