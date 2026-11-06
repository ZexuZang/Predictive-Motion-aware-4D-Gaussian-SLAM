"""Causal RGB-D input, TUM associations, and GT reader (GT NEVER used in tracking)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import yaml

@dataclass
class RGBDFrame:
    rgb: np.ndarray             # H W 3 RGB uint8
    depth: np.ndarray           # H W metres float32
    timestamp: float
    name: str
    dynamic_mask: np.ndarray | None = None

def load_intrinsics(path):
    spec = yaml.safe_load(Path(path).read_text(encoding="utf8"))
    K = np.asarray(spec["K"], float).reshape(3,3) if "K" in spec else np.array([
        [spec["fx"],0,spec["cx"]], [0,spec["fy"],spec["cy"]], [0,0,1]],float)
    if not np.isfinite(K).all() or np.any(np.diag(K) <= 0):
        raise ValueError("Invalid camera intrinsics")
    return K

def associations(root):
    root = Path(root)
    ap = root / "associations.txt"
    if ap.is_file():
        out=[]
        for line in ap.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"): continue
            a=line.split()
            if len(a)>=4:
                out.append((float(a[0]),root/a[1],float(a[2]),root/a[3]))
        if out: return out
    def parse(name):
        return [(float(a[0]),root/a[1]) for ln in (root/name).read_text().splitlines()
                if (a:=ln.split()) and not ln.lstrip().startswith("#") and len(a)>=2]
    rg=parse("rgb.txt"); dp=parse("depth.txt")
    from bisect import bisect_left
    dp=sorted(dp); times=[x[0] for x in dp]
    used=set();out=[]
    for t,r in sorted(rg):
        j=bisect_left(times,t)
        options=[k for k in [j-1,j] if 0<=k<len(dp) and k not in used]
        if not options: continue
        k=min(options,key=lambda k:abs(dp[k][0]-t))
        if abs(dp[k][0]-t)<=.035 and r.is_file() and dp[k][1].is_file():
            out.append((t,r,dp[k][0],dp[k][1])); used.add(k)
    return out

def frame_stream(root,max_frames=0,stride=1,depth_scale=5000.,dynamic_mask_dir=None):
    if stride<1 or depth_scale<=0: raise ValueError("stride>=1, depth_scale>0 required")
    pairs=associations(root)
    if not pairs: raise RuntimeError(f"No RGB-D pairs: {root}")
    count=0
    for t,r,td,d in pairs[::stride]:
        if max_frames and count>=max_frames: break
        if not r.is_file() or not d.is_file(): raise FileNotFoundError(f"Missing {r} or {d}")
        with Image.open(r) as im: rgb=np.asarray(im.convert("RGB"),dtype=np.uint8)
        with Image.open(d) as im: raw=np.asarray(im)
        depth=raw.astype(np.float32)/depth_scale
        if depth.shape!=rgb.shape[:2]: raise ValueError("RGB/depth size mismatch")
        mask=None
        if dynamic_mask_dir:
            for ext in (".png",".jpg"):
                path=Path(dynamic_mask_dir)/(r.stem+ext)
                if path.is_file():
                    with Image.open(path) as im: mask=np.asarray(im.convert("L"))>127
                    if mask.shape!=depth.shape:raise ValueError(f"Mask size mismatch: {path}")
                    break
        yield RGBDFrame(rgb,depth,float(t),r.stem,mask)
        count+=1

def read_gt(path):
    out=[]
    if path is None or not Path(path).is_file(): return out
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):continue
        a=line.split()
        if len(a)<8:continue
        t=float(a[0]);v=np.array(list(map(float,a[1:8])))
        T=np.eye(4);T[:3,:3]=Rotation.from_quat(v[3:7]).as_matrix();T[:3,3]=v[:3]
        out.append((t,T))
    return out
