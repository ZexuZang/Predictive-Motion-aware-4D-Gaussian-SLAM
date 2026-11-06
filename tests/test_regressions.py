"""Regressions for failures found during repository audit."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from predictive4d import joint_ba
from predictive4d.geometry import se3_exp, transform
from predictive4d.gaussians import GaussianMap
from predictive4d.tracking import dynamic_pose_feedback
from predictive4d.official_bridge import export_frame_packet
from predictive4d.flow4dgs_online import OnlineNeuralRuntime, cache_completed_frame
from predictive4d.flow4dgs_native import apply_dynamic_packet
from predictive4d.pipeline import RGBDPredictiveSLAM
from predictive4d.data import RGBDFrame


def test_ill_conditioned_hessian_is_not_overwritten_as_joint_ok(monkeypatch):
    rng = np.random.default_rng(44)
    x = rng.normal(size=(80, 3)); x[:, 2] += 4.
    obs = transform(se3_exp([.008, .005, 0, 0, .005, 0]), x)
    cfg = {'max_match_distance': .2, 'min_joint_matches': 24,
           'joint_outer_iterations': 2, 'max_pose_translation_m': .3,
           'max_pose_rotation_deg': 18., 'pose_prior_precision': 2.}
    monkeypatch.setattr(joint_ba.np.linalg, 'cond', lambda _: float('inf'))
    result, post, used, diag = joint_ba.optimize_joint_pose(
        np.eye(4), np.eye(4), x, np.zeros_like(x), np.full(80, .05),
        np.full_like(x, .001), obs, cfg)
    assert diag['matches'] >= 24
    assert diag['status'] == 'ill_conditioned'
    assert np.allclose(result, np.eye(4))


def test_zero_optimizer_steps_does_not_report_tracking_success():
    rng = np.random.default_rng(13)
    x = rng.normal(size=(60, 3)); x[:, 2] += 4
    cfg = {'max_match_distance': .2, 'min_joint_matches': 24,
           'joint_outer_iterations': 0}
    _, _, _, diag = joint_ba.optimize_joint_pose(
        np.eye(4), np.eye(4), x, np.zeros_like(x), np.full(60, .05),
        np.full_like(x, .001), x, cfg)
    assert diag['status'] == 'no_iterations'


def test_uncalibrated_feedback_does_not_modify_camera():
    x = np.random.default_rng(7).normal(size=(30, 3)); x[:, 2] += 3
    g = GaussianMap.from_points(x, np.ones_like(x) * .4)
    cfg = {'feedback_enabled': True, 'uncertainty_calibrated': False}
    out, diag = dynamic_pose_feedback(
        np.eye(4), np.eye(4), x, np.zeros((0, 3)), np.array([], int), g, cfg)
    assert diag['status'] == 'uncalibrated_motion_skipped'
    assert not diag['accepted']
    assert np.allclose(out, np.eye(4))


def test_tracking_loss_does_not_contaminate_geometry_history(tmp_path):
    cfg = {'history_length': 3, 'max_points': 100, 'pixel_stride': 2,
           'seed': 1, 'max_match_distance': .1, 'min_joint_matches': 24,
           'feedback_enabled': False, 'snapshot_interval': 0, 'preview_interval': 0,
           'max_static_map': 100, 'max_dynamic_map': 100, 'append_gaussians': 30,
           'max_static_confidence': .65, 'min_dynamic_confidence': .75,
           'mapping_opt_steps': 0, 'mapping_opt_interval': 4,
           'max_pose_translation_m': .3, 'max_pose_rotation_deg': 18.}
    K = np.array([[100, 0, 24], [0, 100, 20], [0, 0, 1]], float)
    slam = RGBDPredictiveSLAM(K, cfg, network=None, device='cpu', out=tmp_path)
    rgb = np.ones((40, 48, 3), dtype=np.uint8) * 100
    first = RGBDFrame(rgb, np.full((40, 48), 2., np.float32), 0., '0')
    far = RGBDFrame(rgb, np.full((40, 48), 10., np.float32), .1, '1')
    assert slam.process(first)['status'] == 'bootstrap'
    assert slam.process(far)['status'] == 'tracking_lost'
    assert len(slam.poses) == 2
    assert len(slam.history) == 1
    assert slam.history[-1].timestamp == 0.


class DummyNativeCamera:
    def __init__(self, i):
        self.R = torch.eye(3)
        self.T = torch.tensor([i * .005, 0., 0.])
        self.depth = np.full((48, 64), 2., np.float32)
        self.original_image = torch.full((3, 48, 64), .4)
        self.intrinsic = torch.tensor([[80., 0., 32.], [0., 80., 24.], [0., 0., 1.]])

    def clean(self):
        self.depth = self.original_image = None


class DummyPredictor:
    def predict(self, history, dt):
        assert np.isfinite(history[-1].xyz_world).all()
        points = history[-1].xyz_world
        return np.eye(4), np.zeros_like(points), np.full(len(points), .9), np.full_like(points, .01), {}


def test_online_cache_survives_native_cleanup_and_pose_updates(tmp_path):
    import yaml
    yaml_path = tmp_path / 'config.yaml'
    yaml_path.write_text(yaml.safe_dump({'history_length': 3, 'max_points': 100,
                                          'pixel_stride': 4, 'seed': 42}))
    frontend = SimpleNamespace(
        cameras={i: DummyNativeCamera(i) for i in range(4)},
        config={'predictive4d': {'online': True, 'model_config': str(yaml_path),
                                  'packet_dir': str(tmp_path)}})
    for i in (0, 1):
        assert cache_completed_frame(frontend, i)
        frontend.cameras[i].clean()
    # BA pose updates AFTER raw camera-depth cleanup are reflected at inference.
    frontend.cameras[0].T = torch.tensor([.02, .0, .0])
    runtime = OnlineNeuralRuntime.__new__(OnlineNeuralRuntime)
    runtime.cfg = {'history_length': 3, 'max_points': 100, 'pixel_stride': 4, 'seed': 42}
    runtime.dt = .1
    runtime.pred = DummyPredictor()
    assert runtime.predict(frontend, frontend.cameras[3], 3)
    assert (tmp_path / '000003.npz').is_file()


class FakeGaussians:
    def __init__(self, initial):
        self._xyz = torch.as_tensor(initial.copy(), dtype=torch.float32)
        self.dygs = torch.ones(len(initial), dtype=torch.bool)
        self.kf_list = [0, 2]
        self.delta = torch.zeros_like(self._xyz)
        self.applied = None

    @property
    def get_dygs_xyz(self):
        return self._xyz

    def get_delta_xyz(self, idx):
        assert idx == 0
        return self.delta

    def update_delta_xyz(self, frame_index, previous_keyframe_id,
                         values=None, idx=None, max_motion=.1):
        self.applied = (frame_index, previous_keyframe_id,
                        values.detach().cpu().numpy(), idx.detach().cpu().numpy(), max_motion)


def test_native_flow_composes_all_frames_between_keyframes(tmp_path):
    rng = np.random.default_rng(57)
    n = 25
    p0 = np.column_stack([rng.uniform(-.4, .4, n), rng.uniform(-.4, .4, n),
                          rng.uniform(2., 3., n)])
    d1 = np.tile([.035, 0., 0.], (n, 1))
    d2 = np.tile([.04, 0., 0.], (n, 1))
    for frame_idx, prev, step_flow in ((1, p0, d1), (2, p0 + d1, d2)):
        export_frame_packet(tmp_path, frame_idx, frame_idx * .1, np.eye(4), prev,
                            step_flow, np.full(n, .95), np.full((n, 3), .001))
    gauss = FakeGaussians(p0)
    backend = SimpleNamespace(config={'predictive4d': {
        'packet_dir': str(tmp_path), 'min_gaussian_matches': 12,
        'max_gaussian_motion_m': .1}}, gaussians=gauss)
    assert apply_dynamic_packet(backend, 2, 0)
    assert gauss.applied is not None
    cur, prev, flow, indices, max_m = gauss.applied
    assert (cur, prev) == (2, 0)
    assert len(indices) == n
    assert np.allclose(flow, d1 + d2, atol=1e-5)
    assert max_m > .1


def test_native_flow_rejects_incomplete_keyframe_prediction(tmp_path):
    xyz = np.tile([0., 0., 2.], (20, 1))
    export_frame_packet(tmp_path, 2, .2, np.eye(4), xyz,
                        np.tile([.02, 0., 0.], (20, 1)),
                        np.ones(20), np.full((20, 3), .001))
    gauss = FakeGaussians(xyz)
    backend = SimpleNamespace(config={'predictive4d': {'packet_dir': str(tmp_path),
                                'min_gaussian_matches': 1}}, gaussians=gauss)
    assert not apply_dynamic_packet(backend, 2, 0)
    assert gauss.applied is None


def test_upstream_patcher_inserts_cleanup_and_tracking_hooks_together(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'patch_flow4dgs', str(__import__('pathlib').Path(__file__).parents[1] /
                               'scripts' / 'patch_flow4dgs.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / 'slam_frontend.py'
    path.write_text('class FrontEnd:\n'
                    '    def tracking(self):\n'
                    '        viewpoint.update_RT(prev.R, prev.T)\n'
                    '    def cleanup(self):\n'
                    '        self.cameras[cur_frame_idx].clean()\n')
    assert module.patch_one(path, module.FRONTEND_ANCHOR,
                            module.FRONTEND_REPLACEMENT, apply=True) != 'already patched'
    assert module.patch_one(path, module.CLEANUP_ANCHOR,
                            module.CLEANUP_REPLACEMENT, apply=True) != 'already patched'
    assert module.patch_one(path, module.CLEANUP_ANCHOR,
                            module.CLEANUP_REPLACEMENT, apply=True) == 'already patched'
    source = path.read_text()
    assert source.count('PREDICTIVE4D_FRONTEND_HOOK') == 1
    assert source.count('PREDICTIVE4D_HISTORY_CACHE') == 1
