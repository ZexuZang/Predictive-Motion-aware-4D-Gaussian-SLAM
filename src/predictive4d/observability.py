"""Static camera information matrix and gated addition of dynamic evidence."""
from __future__ import annotations
import numpy as np
from scipy.spatial import cKDTree
from .geometry import transform,se3_log,invert
from .joint_ba import skew


def static_camera_information(T,static_world,current_camera,max_dist=.20,obs_std=.045):
    H=np.zeros((6,6),float)
    if not len(static_world) or not len(current_camera):return H,0
    xyz=transform(T,static_world)
    d,j=cKDTree(current_camera).query(xyz,distance_upper_bound=max_dist)
    good=np.flatnonzero(np.isfinite(d))
    for i in good:
        p=xyz[i];J=np.column_stack((-np.eye(3),skew(p)))
        wt=1/(obs_std**2)*(1/(1+(d[i]/.1)**2))
        H+=wt*J.T@J
    return H,len(good)


def weak_direction_test(T_static,T_candidate,H_static,relative_threshold=.10):
    """Check candidate pose update overlaps a weakly observed static direction.

    In completely unanchored frames the test cannot establish useful weak
    directions and returns `no_static_anchor`. It is *not* a proof of information
    gain or calibrated uncertainty.
    """
    eig,Q=np.linalg.eigh(H_static)
    if np.max(eig)<1e-7:return False,{'status':'no_static_anchor'}
    weak=eig<max(1e-6,float(np.max(eig))*relative_threshold)
    if not np.any(weak):return False,{'status':'static_well_conditioned'}
    delta=se3_log(T_candidate@invert(T_static))
    energy=float(np.linalg.norm(Q[:,weak].T@delta)**2)
    total=float(delta@delta)
    fraction=energy/max(1e-12,total)
    return fraction>.15,{'status':'weak_direction_evaluated','weak_fraction':fraction,
                          'static_information_eigenvalues':eig.tolist()}
