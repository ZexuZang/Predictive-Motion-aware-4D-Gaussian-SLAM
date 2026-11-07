"""Minimal fusion tests; none require GenFlow3D or Flow4DGS CUDA."""
from pathlib import Path
import numpy as np
import torch
from predictive4d.geometry import se3_exp, invert
from predictive4d.fusion_head import (FEATURE_NAMES, ReliabilityFusionHead,
    fuse_numpy, fuse_torch, reliability_features, FusionRuntime)
from predictive4d.train_fusion import load_pairs


def test_pose_interpolation_endpoints_and_differentiability():
    t0=se3_exp([.1,-.02,.03,.01,-.01,.004]);t1=se3_exp([.22,-.04,.06,.025,.002,.01])
    assert np.allclose(fuse_numpy(t0,t1,0.),t0,atol=1e-8)
    assert np.allclose(fuse_numpy(t0,t1,1.),t1,atol=1e-7)
    from predictive4d.geometry import se3_log
    xi=se3_log(t1@invert(t0))
    w=torch.tensor([.32],requires_grad=True)
    out=fuse_torch(torch.tensor(t0[None],dtype=torch.float64),
                   torch.tensor(xi[None],dtype=torch.float64),w.double())
    assert np.allclose(out.detach().numpy()[0],fuse_numpy(t0,t1,.32),atol=1e-7)
    out[0,0,3].backward()
    assert w.grad is not None and np.isfinite(w.grad.item())


def test_feature_contract_and_runtime(tmp_path):
    p=se3_exp([.03,.0,.01,.0,.04,.0]);g=se3_exp([.02,.0,.01,.0,.01,.0])
    features=reliability_features(p,g,{'matches':80,'static_mass':70,
      'min_static_eigen':2.,'h_condition':400.,'status':'joint_ok'},
      {'pose_accepted':True,'raw_pose_translation':.03},
      np.full((100,3),.02),np.full(100,.4),90,.8)
    assert features.shape==(len(FEATURE_NAMES),)
    h=ReliabilityFusionHead()
    ckpt=tmp_path/'head.pt'
    torch.save(dict(arch='ReliabilityFusionHead-v1',feature_names=FEATURE_NAMES,
      feature_mean=[0.] * len(FEATURE_NAMES),
      feature_std=[1.] * len(FEATURE_NAMES),model=h.state_dict()),ckpt)
    w=FusionRuntime(ckpt).predict_weight(features)
    assert 0<w<1


def test_gt_anchor_first_camera_frame_and_no_train_test_leak(tmp_path):
    gt=tmp_path/'groundtruth.txt'
    gt.write_text('0.0 5 0 0 0 0 0 1\n0.1 5.1 0 0 0 0 0 1\n')
    p=tmp_path/'fusion_candidates.npz'
    t_geo=np.eye(4);t_geo[0,3]=-.1
    np.savez(p,origin_timestamp=np.array(0.),timestamps=np.array([.1]),
      T_pred=np.array([t_geo]),T_geo=np.array([t_geo]),
      features=np.ones((1,len(FEATURE_NAMES)),np.float32),valid_geo=np.array([True]))
    data=load_pairs([(p,gt)])
    assert np.allclose(data['gt'][0][:3,3],[-.1,0,0],atol=1e-8)
