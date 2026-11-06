import numpy as np
from predictive4d.geometry import se3_exp,transform,is_pose
from predictive4d.probabilistic import decompose
from predictive4d.joint_ba import optimize_joint_pose,schur_dynamic_factor,skew
from predictive4d.observability import static_camera_information,weak_direction_test


def test_three_hypotheses_are_normalized():
    X=np.array([[0,0,2.],[.3,.4,3.],[.4,-.2,2.5]])
    V=np.array([[.1,0,0],[.3,0,0],[.1,0,0]])
    Y=X.copy();Y[1]+=V[1];Y[2]+=[3,0,0]
    gam,rs,rd=decompose(np.eye(4),X,Y,V,np.array([.1,.9,.8]),np.ones((3,3))*.0003)
    assert np.allclose(gam.sum(axis=1),1.)
    assert np.argmax(gam,axis=1).tolist()==[0,1,2]


def test_schur_equivalent_to_marginal_covariance():
    T=se3_exp([.03,-.01,.04,.01,-.02,.02]);X=np.array([.2,0,2.4]);V=np.array([.02,.03,0])
    Y=transform(T,(X+V)[None])[0]+np.array([.02,-.01,.01])
    U=np.array([.004,.003,.005]);sig=.045
    h,g,q=schur_dynamic_factor(T,X,V,Y,U,sig)
    p=transform(T,(X+V)[None])[0];J=np.concatenate([-np.eye(3),skew(p)],axis=1)
    S=np.eye(3)*sig**2+T[:3,:3]@np.diag(U)@T[:3,:3].T
    r=Y-p
    assert np.allclose(h,J.T@np.linalg.solve(S,J),atol=1e-7)
    assert np.allclose(g,J.T@np.linalg.solve(S,r),atol=1e-7)
    assert np.allclose(q,r@np.linalg.solve(S,r))


def test_joint_pose_uses_valid_static_and_dynamic_observations():
    rng=np.random.default_rng(77);N=200
    X=np.column_stack([rng.uniform(-.8,.8,N),rng.uniform(-.6,.6,N),rng.uniform(2,4,N)])
    flow=np.zeros((N,3));flow[:90,0]=.08
    p=np.full(N,.08);p[:90]=.95
    Tgt=se3_exp([.035,-.016,.015,.005,-.009,.003])
    Y=transform(Tgt,X+flow)+rng.normal(0,.002,(N,3))
    cfg={'max_match_distance':.15,'min_joint_matches':40,'joint_outer_iterations':5,
         'max_pose_translation_m':.3,'max_pose_rotation_deg':12.,'pose_prior_precision':1.}
    T,g,ids,diag=optimize_joint_pose(np.eye(4),np.eye(4),X,flow,p,
                                 np.ones((N,3))*.001,Y,cfg)
    assert diag['status']=='joint_ok',diag
    assert is_pose(T)
    assert np.linalg.norm(T[:3,3]-Tgt[:3,3])<.04
    assert g.shape==(len(ids),3)


def test_observability_matrix_for_3d_static_points():
    rng=np.random.default_rng(8);p=rng.uniform(-1,1,(100,3));p[:,2]+=4
    T=np.eye(4)
    H,n=static_camera_information(T,p,p)
    assert n==100
    assert np.linalg.eigvalsh(H)[0]>0
    valid,diag=weak_direction_test(T,se3_exp([.01,0,0,0,0,0]),H,relative_threshold=1e-6)
    assert not valid and diag['status']=='static_well_conditioned'
