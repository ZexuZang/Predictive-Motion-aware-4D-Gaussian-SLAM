"""Causal nearest-neighbour associations with reciprocal geometric gates.

This is a geometry-only replacement BASELINE; it is *not* GenFlow3D BDCB.
"""
import numpy as np
from scipy.spatial import cKDTree
from .geometry import transform, inverse


def match_prev_to_current(prev_world,current_camera,T_pred,flow_pred=None,max_dist=.24):
    """Predict previous points in current frame and NN associate to observed points.
    Returns prev indices, current indices, Euclidean residual distances.
    """
    n=len(prev_world)
    if n==0 or len(current_camera)==0:
        return np.empty(0,int),np.empty(0,int),np.empty(0)
    flow=np.zeros_like(prev_world) if flow_pred is None else flow_pred
    # Two candidate hypotheses: static and flow-compensated dynamic;
    # choose nearest for correspondence, not for final classification.
    pred_s=transform(T_pred,prev_world)
    pred_d=transform(T_pred,prev_world+flow)
    tree=cKDTree(current_camera)
    ds,js=tree.query(pred_s);dd,jd=tree.query(pred_d)
    choose=dd<ds
    idx=np.where(choose,jd,js);dist=np.where(choose,dd,ds)
    good=dist<max_dist
    # Unique observation assignment: retain the lowest-distance match.
    candidates=np.flatnonzero(good)
    ordered=candidates[np.argsort(dist[candidates])]
    used=set();prev=[];curr=[]
    for i in ordered:
        j=int(idx[i])
        if j in used:continue
        used.add(j);prev.append(int(i));curr.append(j)
    prev=np.array(prev,dtype=int);curr=np.array(curr,dtype=int)
    return prev,curr,dist[prev]
