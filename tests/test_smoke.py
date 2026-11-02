import numpy as np
from predictive4d.synthetic import generate
from predictive4d.data import frame_stream,load_intrinsics
from predictive4d.pipeline import RGBDPredictiveSLAM
import yaml
from pathlib import Path


def test_end_to_end_smoke(tmp_path):
    data=tmp_path/'data';intr=generate(data,nframes=4,h=48,w=64)
    c=yaml.safe_load((Path(__file__).parents[1]/'configs/reference.yaml').read_text())
    c.update({'max_points':350,'sample_stride':3,'initial_gaussians':70,'append_gaussians':30,
              'max_static_gaussians':170,'max_dynamic_gaussians':100,
              'render_width':64,'render_height':48})
    K=load_intrinsics(intr,64,48)
    runner=RGBDPredictiveSLAM(K,c,tmp_path/'outputs')
    met=runner.run(frame_stream(data,max_frames=4,dynamic_mask_dir=data/'dynamic_masks'),data/'groundtruth.txt')
    assert met['frames']==4
    assert (tmp_path/'outputs'/'trajectory_est.tum').exists()
    assert (tmp_path/'outputs'/'dynamic_maps'/'000003.npz').exists()
    assert (tmp_path/'outputs'/'previews'/'000003.png').exists()
    assert np.isfinite(np.array(runner.poses)).all()
    assert met['final_dynamic_gaussians'] > 0
