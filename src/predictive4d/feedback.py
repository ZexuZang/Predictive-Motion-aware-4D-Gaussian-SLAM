"""Optional motion-compensated dynamic Gaussian camera-pose constraints.

This is a *prototype* of the proposed research contribution. It uses hard
nearest-neighbour associations, scalar uncertainty and a chi-square gate;
these should be replaced by calibrated, independently validated factors.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from .geometry import transform,pose_gn,skew


def pose_information(static_world,Tcw,weights=None,obs_variance=.045**2):
    """3D point-to-point information matrix, for rank/weak-direction checks."""
    if len(static_world)==0:return np.zeros((6,6))
    pos=transform(Tcw,static_world);w=np.ones(len(pos)) if weights is None else weights
    H=np.zeros((6,6))
    for p,weight in zip(pos,w):
        J=np.column_stack([-np.eye(3),skew(p)])
        H+=(float(weight)/obs_variance)*(J.T@J)
    return H


def refine_pose_with_dynamic_gaussians(Tcw,dynamic_pred,current_camera,
                                       static_world=None,max_match=.08,
                                       chi2_threshold=11.345,
                                       gate_when_static_weak=False):
    """Use only PRE-OBSERVATION dynamic Gaussian prediction.

    This simplification assumes independent scalar Gaussian positional errors.
    It does NOT use a future corrected map as evidence and intentionally gates
    large innovation to avoid harmful feedback.
    """
    n=len(dynamic_pred)
    if n<10 or len(current_camera)<10:
        return Tcw,{'accepted':0,'status':'insufficient_dynamic_gaussians'}
    if gate_when_static_weak and static_world is not None and len(static_world)>20:
        eig=np.linalg.eigvalsh(pose_information(static_world,Tcw))
        if eig[0]>1000:
            return Tcw,{'accepted':0,'status':'static_geometry_sufficient'}
    predicted_camera=transform(Tcw,dynamic_pred.mean)
    dist,j=cKDTree(current_camera).query(predicted_camera)
    variance=np.asarray(dynamic_pred.motion_variance)+.045**2
    nis=dist**2/np.maximum(variance,1e-9)
    ok=(dist<max_match)&(nis<chi2_threshold)&np.isfinite(dist)
    if np.sum(ok)<10:return Tcw,{'accepted':int(np.sum(ok)),'status':'innovation_gate_failed'}
    # Same current RGB-D used in static fitting, so keep a strong prior to prevent
    # double-counting/overfitting. Formal factor graph needs joint correlation model.
    weights=1/np.maximum(variance[ok],1e-4)
    Tnew=pose_gn(Tcw,dynamic_pred.mean[ok],current_camera[j[ok]],weights,
                  prior=Tcw,prior_strength=5000,iterations=2)
    diff=np.linalg.norm(Tnew[:3,3]-Tcw[:3,3])
    if diff>.04:return Tcw,{'accepted':int(np.sum(ok)),'status':'large_update_rejected'}
    return Tnew,{'accepted':int(np.sum(ok)),'status':'dynamic_feedback_applied'}
