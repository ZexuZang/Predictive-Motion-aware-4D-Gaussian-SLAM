from types import SimpleNamespace
import numpy as np
import pytest
import torch
from predictive4d.cv_acceleration import CVAccelerationGRU, CVAccelerationRuntime, motion_features_from_poses
from predictive4d.geometry import se3_exp, invert


def _history():
    return [SimpleNamespace(T_cw=se3_exp(np.array([0.01 * k, 0., 0., 0., 0., 0.])), timestamp=k*.1)
            for k in range(6)]


def _checkpoint(path, a):
    model=CVAccelerationGRU(np.zeros(13),np.ones(13),np.ones(6),hidden=8)
    with torch.no_grad():
        for param in model.parameters(): param.zero_()
        model.head[-1].bias[:] = torch.as_tensor(a, dtype=torch.float32)
    torch.save({'arch':'CVAccelerationGRU-v1','trained':True,'history_poses':6,
                'feature_dim':13,'hidden':8,'model':model.state_dict()},path)


def test_features_match_training_shape_and_valid_dt():
    h=_history()
    f=motion_features_from_poses([x.T_cw for x in h],[x.timestamp for x in h])
    assert f.shape==(5,13)
    assert np.allclose(f[:,6:12],0,atol=1e-5)
    assert np.allclose(f[:,-1],.1)
    with pytest.raises(ValueError):
        motion_features_from_poses([x.T_cw for x in h],[0]*6)


def test_zero_prediction_exact_cv(tmp_path):
    f=tmp_path/'c.pt';_checkpoint(f,np.zeros(6))
    r=CVAccelerationRuntime(f,linear_threshold=0,angular_threshold=1.5)
    h=_history();T,diag=r.predict(h,.1)
    expected=(h[-1].T_cw@invert(h[-2].T_cw))@h[-1].T_cw
    assert np.allclose(T,expected,atol=1e-6)
    assert diag['camera_prior_mode']=='cv_gru'


def test_angular_gate_and_long_gap_fallback(tmp_path):
    f=tmp_path/'c.pt';_checkpoint(f,[1,0,0,0,0,1.0])
    h=_history()
    r=CVAccelerationRuntime(f,linear_threshold=0,angular_threshold=1.5,max_dt_ratio=2)
    T,diag=r.predict(h,.1)
    cv=(h[-1].T_cw@invert(h[-2].T_cw))@h[-1].T_cw
    expected=se3_exp([.005,0,0,0,0,0])@cv
    assert np.allclose(T,expected,atol=1e-6)
    assert diag['linear_gate_active'] and not diag['angular_gate_active']
    T2,diag2=r.predict(h,.3)
    assert diag2['acceleration_status']=='long_gap_cv_only'
    assert np.allclose(T2,se3_exp([.03,0,0,0,0,0])@h[-1].T_cw)
