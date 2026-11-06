import numpy as np
import torch
from scipy.spatial.transform import Rotation
from predictive4d.network import JointMotionNet
from predictive4d.geometry import se3_exp,invert,transform,rigid_fit,is_pose
from predictive4d.gaussians import GaussianMap,propagate_gaussians
from predictive4d.tracking import dynamic_pose_feedback

def test_shared_encoder_receives_both_losses():
    net=JointMotionNet(hidden=32)
    p=torch.randn(2,3,16,7);tw=torch.randn(2,3,6)*.01;dt=torch.full((2,3,1),.1)
    v=torch.randn(2,16,3)*.01
    out=net(p,tw,dt,v)
    assert out['pose_xi'].shape==(2,6)
    assert out['flow'].shape==(2,16,3)
    assert out['dynamic_logits'].shape==(2,16)
    pose=out['pose_xi'].square().mean()
    g1=torch.autograd.grad(pose,net.encoder[0].weight,retain_graph=True)[0]
    scene=out['flow'].square().mean()
    g2=torch.autograd.grad(scene,net.encoder[0].weight)[0]
    assert g1.abs().sum()>0 and g2.abs().sum()>0

def test_exp_produces_se3():
    T=se3_exp([.15,-.04,.02,.01,-.02,.04]);assert is_pose(T)
    assert np.allclose(transform(invert(T),transform(T,np.ones((5,3)))),np.ones((5,3)),atol=1e-8)

def test_dynamic_feedback_can_adjust_camera():
    rng=np.random.default_rng(4)
    world=np.column_stack([rng.uniform(-.5,.5,100),rng.uniform(-.5,.5,100),rng.uniform(2.0,3.5,100)])
    g=GaussianMap.from_points(world,np.ones((len(world),3))*.5)
    Tgt=se3_exp([.04,-.02,.01,.001,.005,.002]);current=transform(Tgt,world)
    cfg={'feedback_enabled':True,'feedback_static_weak_below':90,
         'feedback_gate_sigma':3.,'feedback_min_dynamic_matches':8,
         'feedback_max_translation_m':.12,'feedback_max_rotation_deg':8.,
         'feedback_static_degradation':1.15,'max_match_distance':.23,
         'uncertainty_calibrated':True}
    T,info=dynamic_pose_feedback(np.eye(4),np.eye(4),current,np.zeros((0,3)),np.array([],int),g,cfg)
    assert info['accepted'],info
    assert np.linalg.norm(T[:3,3]-Tgt[:3,3])<.025

def test_gaussian_degenerate_neighbors_keep_finite_covariance():
    xyz=np.column_stack([np.linspace(0,.1,8),np.zeros(8),np.ones(8)*2])
    g=GaussianMap.from_points(xyz,np.ones((8,3)))
    r=propagate_gaussians(g,xyz,np.ones_like(xyz)*.01,np.ones_like(xyz)*.01)
    assert np.isfinite(r.mean).all() and np.isfinite(r.cov).all()

def test_differentiable_torch_splat():
    from predictive4d.torch_renderer import soft_gaussian_rgbd
    mu=torch.tensor([[0.,0.,2.],[.2,0.,2.3]],requires_grad=True)
    cov=torch.eye(3)[None].repeat(2,1,1)*.02**2
    rgb=torch.tensor([[.8,.2,.2],[.2,.2,.8]],requires_grad=True)
    K=torch.tensor([[70.,0.,39.],[0.,70.,30.],[0.,0.,1.]])
    image,depth,mass=soft_gaussian_rgbd(mu,cov,rgb,torch.ones(2)*.8,
                                        torch.eye(4),K,height=40,width=50,orig_hw=(60,80))
    loss=image.square().sum()+.01*depth.sum();loss.backward()
    assert mu.grad is not None and mu.grad.abs().sum()>0
    assert rgb.grad is not None and rgb.grad.abs().sum()>0

def test_official_adapter_uses_same_upstream_encoder():
    from torch import nn
    from predictive4d.genflow_shared import OfficialGenFlowShared
    class FakeEncoder(nn.Module):
        def __init__(self):
            super().__init__();self.proj=nn.Conv1d(3,32,1)
        def forward(self,a,b):return [],[],[self.proj(b)]
    class FakeOfficial(nn.Module):
        def __init__(self):
            super().__init__();self.unit=nn.Module();self.unit.encoder=FakeEncoder()
        def forward(self,pcs,feats,gts):
            C=torch.cat([v.transpose(1,2) for v in pcs],0)
            self.unit.encoder(C,C)
            B,N,_=pcs[-1].shape
            return [[torch.zeros((B,3,N))]],[],[]
    original=FakeOfficial();adapter=OfficialGenFlowShared(original,hidden=32)
    p=torch.randn(2,3,24,7)
    out=adapter(p,torch.zeros(2,3,6),torch.ones(2,3,1)*.1,torch.zeros(2,24,3))
    assert adapter.genflow.unit.encoder is original.unit.encoder
    assert out['pose_xi'].shape==(2,6) and out['flow'].shape==(2,24,3)
    loss=out['pose_xi'].square().mean()+out['dynamic_logits'].square().mean()
    loss.backward()
    assert adapter.camera_head[-1].weight.grad is not None
    assert original.unit.encoder.proj.weight.grad is None

def test_official_packet_roundtrip(tmp_path):
    from predictive4d.official_bridge import export_frame_packet,map_flow_to_official_gaussian_offsets
    xyz=np.array([[0.,0.,2.],[.2,0,2.],[.4,0,2.]])
    flow=np.array([[0.01,0,0],[0.02,0,0],[0.03,0,0.]])
    prob=np.array([.9,.2,.95])
    var=np.ones_like(xyz)*.01
    p=export_frame_packet(tmp_path,3,3.1,np.eye(4),xyz,flow,prob,var)
    offs,accepted=map_flow_to_official_gaussian_offsets(p,xyz)
    assert accepted.tolist()==[True,False,True]
    assert np.allclose(offs[[0,2]],flow[[0,2]]) and np.allclose(offs[1],0)
