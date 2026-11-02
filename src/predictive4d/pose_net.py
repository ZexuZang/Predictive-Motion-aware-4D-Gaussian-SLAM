"""Optional supervised research camera predictor. Requires PyTorch.

Not pretrained, not equivalent to a GenFlow3D shared feature encoder.
Input: L=3 history point clouds in camera coordinates and historical SE3 twists.
Output: next SE3 left increment and diagonal log variance (uncalibrated until tested).
"""
import torch
from torch import nn

class PoseGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.point=nn.Sequential(nn.Linear(3,64),nn.ReLU(),nn.Linear(64,96),nn.ReLU())
        self.gru=nn.GRU(96+6,128,batch_first=True)
        self.head=nn.Sequential(nn.Linear(128,128),nn.ReLU(),nn.Linear(128,12))
    def forward(self,pts,twists):
        # pts: B,L,N,3 ; twists: B,L,6
        B,L,N,_=pts.shape
        f=self.point(pts).amax(dim=2)
        z,_=self.gru(torch.cat([f,twists],dim=-1))
        y=self.head(z[:,-1])
        return y[:,:6],torch.clamp(y[:,6:],-10,3)
