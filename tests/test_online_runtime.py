from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch
from predictive4d.flow4dgs_online import OnlineNeuralRuntime

class DummyNativeCamera:
    def __init__(self,i):
        self.R=torch.eye(3);self.T=torch.tensor([i*.005,0.,0.])
        self.depth=np.ones((48,64),np.float32)*2.
        self.original_image=torch.ones((3,48,64),dtype=torch.float32)*.4
        self.intrinsic=torch.tensor([[80.,0.,32.],[0.,80.,24.],[0.,0.,1.]])

class DummyPredictor:
    def __init__(self):self.seen=[]
    def predict(self,history,dt):
        self.seen=[h.timestamp for h in history]
        xyz=history[-1].xyz_world
        return np.eye(4),np.zeros_like(xyz),np.ones(len(xyz))*.4,np.ones_like(xyz)*.01,{}


def test_online_prior_reads_only_prior_completed_frames(tmp_path):
    runtime=OnlineNeuralRuntime.__new__(OnlineNeuralRuntime)
    runtime.cfg={'history_length':3,'max_points':100,'pixel_stride':4,'seed':10}
    runtime.dt=.1;runtime.pred=DummyPredictor()
    f=SimpleNamespace(cameras={i:DummyNativeCamera(i) for i in range(4)},
       config={'predictive4d':{'packet_dir':str(tmp_path)}})
    assert runtime.predict(f,f.cameras[3],3)
    assert np.allclose(runtime.pred.seen,[0,.1,.2])
    assert (tmp_path/'000003.npz').exists()
