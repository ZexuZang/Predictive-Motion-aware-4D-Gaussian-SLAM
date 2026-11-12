"""Native integration source patches must be transactional and idempotent."""
import importlib.util
import pathlib
import subprocess
import sys
from types import SimpleNamespace
import pytest

ROOT=pathlib.Path(__file__).resolve().parents[1]


def _load(script):
    spec=importlib.util.spec_from_file_location(script,ROOT/'scripts'/f'{script}.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(root):
    (root/'utils').mkdir(parents=True)
    (root/'configs/rgbd/bonn').mkdir(parents=True)
    (root/'utils/slam_frontend.py').write_text('''import torch
class Front:
    def init(self, viewpoint):
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
    def track(self, prev, viewpoint, cur_frame_idx):
        viewpoint.update_RT(prev.R, prev.T)
    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
''')
    (root/'utils/slam_backend.py').write_text('''class Back:
    def map(self, key_opt, current_window):
        self.viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in key_opt]
        current_window_set = set(key_opt)
        if True:
            if True:
                if True:
                    if True:
                        if True:
                            self.apply_flow_offset(flow, viewpoint, self.viewpoints[closest_keyframe])
''')
    (root/'configs/rgbd/bonn/bonn_ps_trk.yaml').write_text(
        'inherit_from: "configs/rgbd/bonn/base_config.yaml"\n')


def test_dry_run_and_idempotent_apply(tmp_path):
    m=_load('patch_flow4dgs');_fixture(tmp_path)
    files=[tmp_path/x for x in (m.FRONT,m.BACK,m.CONFIG)]
    before=[x.read_text() for x in files]
    dry=m.patch(tmp_path,apply=False)
    assert not dry['applied']
    assert [x.read_text() for x in files]==before
    m.patch(tmp_path,apply=True)
    after=[x.read_text() for x in files]
    assert after!=before
    assert 'predict_native_paper' in after[0]
    assert 'cache_completed_native' in after[0]
    assert 'PREDICTIVE4D_GENFLOW_APPLIED' in after[1]
    assert 'key_opt]' not in after[1]
    assert 'base_config.yaml' not in after[2]
    assert not any(m.patch(tmp_path,apply=True)['changes'].values())
    assert [x.read_text() for x in files]==after


def test_unexpected_anchor_fails_before_writing(tmp_path):
    m=_load('patch_flow4dgs');_fixture(tmp_path)
    b=tmp_path/m.BACK;b.write_text(b.read_text().replace('set(key_opt)', 'set(blah)'))
    f=tmp_path/m.FRONT;old=f.read_text()
    with pytest.raises(RuntimeError):m.patch(tmp_path,apply=True)
    assert f.read_text()==old


def test_genflow_namespace_patch(tmp_path):
    p=_load('prepare_genflow3d_imports');(tmp_path/'utils.py').write_text('class PointWarping: pass\n')
    for filename in ('models.py','units.py','layers.py'):
        (tmp_path/filename).write_text('import utils\n')
    p.patch(tmp_path,apply=True)
    assert (tmp_path/'genflow3d_utils.py').is_file()
    assert all('import genflow3d_utils as utils' in (tmp_path/x).read_text()
               for x in ('models.py','units.py','layers.py'))
    p.patch(tmp_path,apply=True)


def test_cuda_compat_patch(tmp_path):
    p=_load('patch_native_compat')
    h=tmp_path/'submodules/diff-gaussian-rasterization/cuda_rasterizer'
    h.mkdir(parents=True)
    (h/'helper_math.h').write_text('\n'.join('float lerp(float a, float b, float t);' for _ in range(4)))
    (h/'rasterizer_impl.h').write_text('#pragma once\n#include <iostream>\n')
    (tmp_path/'utils').mkdir()
    for name in ('slam_frontend.py','slam_backend.py'):
        (tmp_path/'utils'/name).write_text('def f():\n    pass\n    plt.tight_layout()\n')
    p.patch(tmp_path,apply=True)
    assert 'cuda_compat_lerp' in (h/'helper_math.h').read_text()
    assert '#include <cstdint>' in (h/'rasterizer_impl.h').read_text()
    assert 'plt.tight_layout()  # Matplotlib' in (tmp_path/'utils/slam_frontend.py').read_text()
    assert not any(p.patch(tmp_path,apply=True)['changed'].values())
