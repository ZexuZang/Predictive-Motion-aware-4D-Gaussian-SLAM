"""RAFT-free depth reprojection inconsistency cue (not a learned BDCB).

A foreground reveal relative to previous reprojected depth may indicate dynamic
motion, but can also result from occlusion, calibration and pose errors.
"""
import numpy as np
from scipy.ndimage import minimum_filter
from .geometry import transform,project


def foreground_depth_innovation(prev_world,Tcw,points_camera,uv_current,K,image_shape,
                                threshold=.20,radius=2):
    h,w=image_shape;ref=transform(Tcw,prev_world)
    px,valid=project(ref,K)
    rc=np.rint(px).astype(int)
    good=valid&(rc[:,0]>=0)&(rc[:,0]<w)&(rc[:,1]>=0)&(rc[:,1]<h)
    zbuf=np.full((h,w),np.inf,dtype=float)
    np.minimum.at(zbuf,(rc[good,1],rc[good,0]),ref[good,2])
    zbuf=minimum_filter(zbuf,size=radius*2+1,mode='constant',cval=np.inf)
    zpred=zbuf[uv_current[:,0],uv_current[:,1]]
    # Only confidence where the previous map has actual projected depth coverage.
    return np.isfinite(zpred)&(zpred-points_camera[:,2]>threshold)
