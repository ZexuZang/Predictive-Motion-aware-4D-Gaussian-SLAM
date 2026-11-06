"""Three-way static / dynamic / outlier posterior for paired RGB-D observations.

Coordinates: world X, V; current camera Y; T_cw world->camera.
Uncertainty: independent *proxy* diagonal world-space flow variances; before
calibration, posterior values must not be interpreted as statistical coverage.
"""
from __future__ import annotations
import numpy as np
from scipy.special import logsumexp
from .geometry import transform


def decompose(T_cw, X_world, Y_camera, V_world, p_dynamic,
              flow_variance_world, observation_std=.045, outlier_prior=.06,
              outlier_density=.0125):
    X=np.asarray(X_world,float);Y=np.asarray(Y_camera,float)
    V=np.asarray(V_world,float);p=np.asarray(p_dynamic,float)
    U=np.asarray(flow_variance_world,float)
    n=len(X)
    if not (X.shape==Y.shape==V.shape==(n,3) and p.shape==(n,) and U.shape==(n,3)):
        raise ValueError('Posterior inputs have inconsistent shapes')
    if not all(np.isfinite(z).all() for z in (X,Y,V,p,U,T_cw)):
        raise ValueError('Posterior inputs must be finite')
    if not 0<=outlier_prior<1:raise ValueError('outlier_prior invalid')
    p=np.clip(p,1e-3,1-1e-3)
    pi_s=(1-outlier_prior)*(1-p)
    pi_d=(1-outlier_prior)*p
    rs=Y-transform(T_cw,X)
    rd=Y-transform(T_cw,X+V)
    sig=np.eye(3)*observation_std**2
    R=T_cw[:3,:3]
    logp=np.empty((n,3),float)
    logp[:,0]=np.log(pi_s)-.5*np.sum(rs**2,axis=1)/observation_std**2 \
        -.5*(3*np.log(2*np.pi)+np.linalg.slogdet(sig)[1])
    for i in range(n):
        S=sig+R@np.diag(np.maximum(U[i],1e-6))@R.T
        q=rd[i]@np.linalg.solve(S,rd[i]);logdet=np.linalg.slogdet(S)[1]
        logp[i,1]=np.log(pi_d[i])-.5*(3*np.log(2*np.pi)+logdet+q)
    logp[:,2]=np.log(max(outlier_prior,1e-9))+np.log(outlier_density)
    gamma=np.exp(logp-logsumexp(logp,axis=1,keepdims=True))
    return gamma,rs,rd
