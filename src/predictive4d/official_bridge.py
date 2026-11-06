"""Version-checked prediction packets to integrate with official Flow4DGS later.

This module is NOT a native Flow4DGS renderer nor a tested hot-swap in upstream
multiprocessing. It exports world-coordinate Gaussian center displacements,
pose priors, validity gates and dynamic masks for the upstream backend adapter.

Official Flow4DGS expects dynamic-Gaussian-specific keyframe offsets in the
same indexing as gaussians.dygs and _delta_xyz, while this scaffold's map
has independent sampled primitives. The mapping MUST be explicitly established
before any native update. Never blindly copy Nx3 arrays into _delta_xyz.
"""
from pathlib import Path
import json
import numpy as np

SCHEMA="predictive4d-flow4dgs-packet-v1"

def export_frame_packet(out_dir,index,timestamp,T_cw,prev_xyz_world,flow_world,
                        dynamic_probability,flow_variance):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    if prev_xyz_world.shape!=flow_world.shape or prev_xyz_world.shape[1]!=3:
        raise ValueError("World points and flow must both be [N,3]")
    if dynamic_probability.shape!=(len(prev_xyz_world),):
        raise ValueError("Dynamic probability must be [N]")
    if flow_variance.shape!=(len(prev_xyz_world),3):
        raise ValueError("Flow variance must be [N,3]")
    if not all(np.isfinite(x).all() for x in (T_cw,prev_xyz_world,flow_world,flow_variance)):
        raise ValueError("Nonfinite packet values")
    path=out/f"{index:06d}.npz"
    np.savez_compressed(path,schema=np.array(SCHEMA),index=np.array(index),
         timestamp=np.array(timestamp),T_cw=T_cw,points_world=prev_xyz_world,
         flow_world=flow_world,dynamic_probability=dynamic_probability,
         flow_variance=flow_variance)
    return path

def map_flow_to_official_gaussian_offsets(packet_path,official_xyz_at_previous_keyframe,
                                         max_distance=.15,min_dynamic_prob=.65):
    """Map predicted world displacements to EXISTING native dynamic Gaussian IDs.

    Call ONLY inside official backend with CURRENT dynamic Gaussian world positions
    from native `get_dygs_xyz + get_delta_xyz(previous_keyframe_index)`.
    Returns per-Gaussian additive offsets [M,3] plus validity [M].
    Does NOT mutate the official GaussianModel (keyframe lifecycle differs).
    """
    from scipy.spatial import cKDTree
    p=np.load(packet_path,allow_pickle=False)
    if str(p['schema'])!=SCHEMA:raise ValueError('Packet schema mismatch')
    pts=p['points_world'];flow=p['flow_world'];dyn=p['dynamic_probability'];var=p['flow_variance']
    xyz=np.asarray(official_xyz_at_previous_keyframe,float)
    if xyz.ndim!=2 or xyz.shape[1]!=3:raise ValueError('Native dynamic xyz must [M,3]')
    if not len(pts):return np.zeros_like(xyz),np.zeros(len(xyz),bool)
    d,i=cKDTree(pts).query(xyz)
    accepted=(d<max_distance)&(dyn[i]>=min_dynamic_prob)&np.isfinite(var[i]).all(axis=1)
    predicted=flow[i].copy();predicted[~accepted]=0.
    return predicted,accepted
