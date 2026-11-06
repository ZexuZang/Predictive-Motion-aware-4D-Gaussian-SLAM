from pathlib import Path
import numpy as np
import pytest
import torch
from predictive4d.official_bridge import export_frame_packet
from predictive4d.flow4dgs_native import apply_camera_packet

class FakeCamera:
    def __init__(self):
        self.device='cpu';self.R=torch.eye(3);self.T=torch.zeros(3)
    def update_RT(self,R,t):
        self.R,self.T=R,t


def test_native_camera_hook_reads_w2c_prior(tmp_path):
    f=tmp_path/'packets';T=np.eye(4);T[:3,3]=[.1,0,-.02]
    export_frame_packet(f,5,1.1,T,np.ones((2,3)),np.zeros((2,3)),
                        np.array([.2,.9]),np.ones((2,3))*.01)
    c=FakeCamera()
    assert apply_camera_packet({'predictive4d':{'packet_dir':str(f)},'model_params':{'flow_cam':False}},c,5)
    assert np.allclose(c.T.numpy(),[.1,0,-.02])
    with pytest.raises(RuntimeError):
        apply_camera_packet({'predictive4d':{'packet_dir':str(f)},'model_params':{'flow_cam':True}},c,5)
