import numpy as np
from predictive4d.geometry import se3_exp,se3_log,transform,weighted_rigid,pose_gn,inverse
from predictive4d.gaussian import GaussianMap,transfer_motion
from predictive4d.motion import motion_posterior


def test_se3_and_registration():
    xi=np.array([.1,-.04,.2,.08,.03,-.05]);T=se3_exp(xi)
    assert np.allclose(se3_log(T),xi,atol=1e-8)
    rng=np.random.default_rng(4);x=rng.normal(size=(50,3));y=transform(T,x)
    est=weighted_rigid(x,y,np.ones(len(x)))
    assert np.allclose(est,T,atol=1e-8)
    refined=pose_gn(np.eye(4),x,y,np.ones(len(x)),iterations=20)
    assert np.allclose(refined,T,atol=1e-5)
    assert np.allclose(transform(inverse(T),y),x)


def test_gaussian_affine_and_posterior():
    rng=np.random.default_rng(2);x=rng.normal(size=(100,3))*.1
    flow=np.column_stack([.05*x[:,0],-.03*x[:,1],.02*x[:,2]])+np.array([.02,.0,.01])
    ga=GaussianMap.from_points(x[:4],np.ones((4,3))*.5)
    result=transfer_motion(ga,x,flow,k=25,radius=.35)
    assert np.isfinite(result.mean).all() and np.isfinite(result.cov).all()
    assert (np.linalg.eigvalsh(result.cov)>0).all()
    gt=motion_posterior(x,x,np.eye(4),np.zeros_like(x))
    assert np.allclose(gt.sum(1),1.)
