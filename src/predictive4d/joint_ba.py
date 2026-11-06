"""Motion-compensated probabilistic SE(3) optimization.

Per-point dynamic world displacement has a Gaussian prior. The 3D dynamic
nuisance variable is analytically eliminated via a Schur complement.
The module is a frame-to-frame local optimizer, NOT full sliding-window
factor-graph BA of camera/keyframe/object states.
"""
from __future__ import annotations
import numpy as np
from .geometry import transform,se3_exp,se3_log,invert,cap_pose,nearest_unique
from .probabilistic import decompose


def skew(v):
    x,y,z=v
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]],float)


def schur_dynamic_factor(T,src_world,flow_world,y_camera,flow_var,observation_std):
    """Return effective 6x6 pose Hessian and gradient, eliminating delta flow.

    residual r = y - T*(X+V), left perturbation. Jacobians:
    J_T = [-I | [p]x], J_V = -R. Flow prior: deltaV ~ N(0,U).
    """
    x=src_world+flow_world;p=transform(T,x[None])[0];r=y_camera-p
    Jt=np.concatenate([-np.eye(3),skew(p)],axis=1)
    Jv=-T[:3,:3];W=np.eye(3)/observation_std**2
    Uinv=np.diag(1/np.maximum(flow_var,1e-6))
    Htt=Jt.T@W@Jt;Htv=Jt.T@W@Jv;Hvv=Jv.T@W@Jv+Uinv
    gt=Jt.T@W@r;gv=Jv.T@W@r
    solve=np.linalg.solve(Hvv,Htv.T)
    h=Htt-Htv@solve
    g=gt-Htv@np.linalg.solve(Hvv,gv)
    S=np.eye(3)*observation_std**2+T[:3,:3]@np.diag(np.maximum(flow_var,1e-6))@T[:3,:3].T
    return (h+h.T)*.5,g,float(r@np.linalg.solve(S,r))


def optimize_joint_pose(T_prior,previous_T,X_world,V_world,p_dynamic,
                        flow_variance,current_camera,cfg):
    """Associate under two motion hypotheses, estimate posterior, optimize pose.

    Returns T, posterior on matched pairs, current assigned IDs, diagnostics.
    All matches are excluded from later independent Gaussian feedback to avoid
    double-counting a current RGB-D observation.
    """
    out={'status':'insufficient_matches','matches':0,'static_mass':0.,
         'dynamic_mass':0.,'min_static_eigen':0.,'h_condition':float('inf')}
    if len(X_world)==0 or len(current_camera)==0:
        return T_prior.copy(),np.zeros((0,3)),np.empty(0,int),out
    if not all(np.isfinite(v).all() for v in (T_prior,X_world,V_world,current_camera)):
        raise FloatingPointError('Nonfinite joint optimizer inputs')
    maxdist=float(cfg['max_match_distance'])
    obsstd=float(cfg.get('obs_std',.045))
    iterations=int(cfg.get('joint_outer_iterations',4))
    minimum=int(cfg.get('min_joint_matches',24))
    # Associate once using the same pose prior. Avoid iterative drifting to
    # arbitrary nearest surfaces. Later outer iterations update posterior only.
    src_s=transform(T_prior,X_world)
    src_d=transform(T_prior,X_world+V_world)
    selected=np.where((p_dynamic>.5)[:,None],src_d,src_s)
    i,j,d=nearest_unique(selected,current_camera,maxdist)
    out['matches']=int(len(i))
    if len(i)<minimum:return T_prior.copy(),np.zeros((0,3)),np.empty(0,int),out
    X=X_world[i];V=V_world[i];Y=current_camera[j];P=p_dynamic[i];U=flow_variance[i]
    T=T_prior.copy();lam=float(cfg.get('pose_prior_precision',2.0))
    static_H=np.zeros((6,6));gamma=np.zeros((len(i),3))
    for it in range(iterations):
        gamma,rs,rd=decompose(T,X,Y,V,P,U,observation_std=obsstd,
                       outlier_prior=float(cfg.get('outlier_prior',.06)))
        H=np.eye(6)*(lam+1e-4)
        xi_prior=se3_log(T@invert(T_prior))
        g=lam*xi_prior.copy();static_H=np.zeros((6,6))
        for k in range(len(i)):
            if gamma[k,0]>1e-3:
                p=transform(T,X[k:k+1])[0];J=np.concatenate([-np.eye(3),skew(p)],axis=1)
                w=float(gamma[k,0])/obsstd**2
                # fixed robust Cauchy suppression, not a new learned measurement
                w/=1+(np.linalg.norm(rs[k])/.15)**2
                hs=w*(J.T@J)
                H+=hs;static_H+=hs;g+=w*J.T@rs[k]
            if gamma[k,1]>1e-3:
                hd,gd,_=schur_dynamic_factor(T,X[k],V[k],Y[k],U[k],obsstd)
                w=float(gamma[k,1])/(1+(np.linalg.norm(rd[k])/.15)**2)
                H+=w*hd;g+=w*gd
        H=(H+H.T)*.5
        eig=np.linalg.eigvalsh(static_H)
        out['min_static_eigen']=float(eig[0])
        out['h_condition']=float(np.linalg.cond(H))
        if not np.isfinite(H).all() or out['h_condition']>1e11:
            out['status']='ill_conditioned';break
        try:delta=np.linalg.solve(H,-g)
        except np.linalg.LinAlgError:
            out['status']='singular';break
        if not np.isfinite(delta).all():out['status']='nonfinite_update';break
        # Guard individual GN steps; this does NOT conceal failed tracking.
        norm=float(np.linalg.norm(delta))
        if norm>.15:delta*=.15/norm
        if norm<1e-6:break
        T=se3_exp(delta)@T
    T,accepted=cap_pose(T,previous_T,
        float(cfg['max_pose_translation_m']),float(cfg['max_pose_rotation_deg']))
    out['static_mass']=float(gamma[:,0].sum())
    out['dynamic_mass']=float(gamma[:,1].sum())
    out['outlier_mass']=float(gamma[:,2].sum())
    out['status']='joint_ok' if accepted and (out['static_mass']+out['dynamic_mass'])>=minimum*.45 else 'joint_rejected'
    if out['status']!='joint_ok':return T_prior.copy(),gamma,j,out
    return T,gamma,j,out
