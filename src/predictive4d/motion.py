"""Three-hypothesis motion decomposition and per-point motion correction."""
import numpy as np
from scipy.special import logsumexp
from .geometry import transform


def motion_posterior(src_world,obs_camera,T,flow_pred,
                     obs_std=.04,flow_std=.2,p_static=.8,
                     outlier_density=1e-3,prior_static=None):
    """Independent isotropic residual model. Returns (N,3) gamma s/d/outlier.
    Prior probabilities are NOT learned calibrated probabilities.
    """
    s=obs_camera-transform(T,src_world)
    d=obs_camera-transform(T,src_world+flow_pred)
    if prior_static is None:p=np.full(len(s),p_static)
    else:p=np.clip(prior_static,.01,.98)
    var_s=obs_std**2;var_d=obs_std**2+flow_std**2
    ls=(-.5*np.sum(s*s,axis=1)/var_s - 1.5*np.log(2*np.pi*var_s)+np.log(.97*p))
    ld=(-.5*np.sum(d*d,axis=1)/var_d - 1.5*np.log(2*np.pi*var_d)+np.log(.97*(1-p)))
    lo=np.full(len(s),np.log(.03*outlier_density))
    L=np.stack([ls,ld,lo],axis=1);gam=np.exp(L-logsumexp(L,axis=1,keepdims=True))
    return gam


def estimate_pose_and_prob(src_world,obs_camera,flow_pred,T_init,prior_pose=None,
                           prior_sigma=.1,obs_std=.035,flow_std=.15,
                           p_static=.8, outer=4):
    """EM-like alternate posterior and SE(3) GN. Dynamic flows are held at prior.
    Induces a pose-only objective after marginalizing per-point flow perturbations.
    """
    from .geometry import pose_gn
    T=T_init.copy()
    prior_strength=1./max(prior_sigma,1e-4)**2 if prior_pose is not None else 0.
    gamma=np.ones((len(src_world),3))/3
    for _ in range(outer):
        gamma=motion_posterior(src_world,obs_camera,T,flow_pred,
                             obs_std,flow_std,p_static)
        # Pseudo-observations: static X and dynamically corrected X + V;
        # dynamic strength uses covariance sum, not same weight as static.
        sources=np.concatenate([src_world,src_world+flow_pred],axis=0)
        observed=np.concatenate([obs_camera,obs_camera],axis=0)
        w_s=gamma[:,0]/obs_std**2
        w_d=gamma[:,1]/(obs_std**2+flow_std**2)
        weights=np.r_[w_s,w_d]
        T=pose_gn(T,sources,observed,weights,iterations=3,prior=prior_pose,
                  prior_strength=prior_strength)
    gamma=motion_posterior(src_world,obs_camera,T,flow_pred,obs_std,flow_std,p_static)
    return T,gamma


def correct_flow(src_world,obs_camera,T,flow_pred,obs_var=.035**2,flow_var=.15**2):
    """Kalman-style flow update under isotropic Gaussian model; no GT used."""
    predicted=transform(T,src_world+flow_pred)
    innovation=obs_camera-predicted
    R=T[:3,:3]
    gain=flow_var/(flow_var+obs_var)
    return flow_pred+gain*(innovation@R)
