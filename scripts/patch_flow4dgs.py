"""Guarded, idempotent native CV/CV+GRU + official GenFlow3D hooks.

Usage: python scripts/patch_flow4dgs.py --upstream /content/Flow4DGS-SLAM
       python scripts/patch_flow4dgs.py --upstream /content/Flow4DGS-SLAM --apply

Modifies only a separate upstream checkout and never installs checkpoints.
Backups are recorded next to changed source files. No reference renderer.
"""
from __future__ import annotations
import argparse
import ast
from pathlib import Path
import shutil

FRONT = 'utils/slam_frontend.py'
BACK = 'utils/slam_backend.py'
CONFIG = 'configs/rgbd/bonn/bonn_ps_trk.yaml'


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    n = text.count(old)
    if n != 1:
        raise RuntimeError(f'{label}: expected one unmodified anchor; got {n}. Do not patch arbitrary upstream versions.')
    return text.replace(old, new, 1)


def plan(upstream: Path) -> dict[Path, str]:
    upstream = upstream.resolve()
    a, b, cfg = (upstream/x for x in (FRONT, BACK, CONFIG))
    for p in (a, b, cfg):
        if not p.is_file():
            raise FileNotFoundError(p)
    f, g, y = a.read_text(), b.read_text(), cfg.read_text()
    f = replace_once(f, '        viewpoint.update_RT(prev.R, prev.T)\n',
'''        viewpoint.update_RT(prev.R, prev.T)
        # PREDICTIVE4D_CAMERA_PRIOR: completed frames only; no evaluation GT.
        from predictive4d.paper_native import predict_native_paper
        predict_native_paper(self, cur_frame_idx, viewpoint)
''', 'frontend camera prior')
    f = replace_once(f, '        self.cameras[cur_frame_idx].clean()\n',
'''        # PREDICTIVE4D_HISTORY_CACHE: retain points before native cleanup.
        from predictive4d.paper_native import cache_completed_native
        cache_completed_native(self, cur_frame_idx)
        self.cameras[cur_frame_idx].clean()
''', 'frontend camera cache')
    f = replace_once(f, '        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)\n',
'''        # PREDICTIVE4D_NO_GT_INITIALIZATION: world origin = camera zero.
        viewpoint.update_RT(torch.eye(3, device=viewpoint.R.device, dtype=viewpoint.R.dtype),
                            torch.zeros(3, device=viewpoint.T.device, dtype=viewpoint.T.dtype))
''', 'initial pose')
    g = replace_once(g,
'                            self.apply_flow_offset(flow, viewpoint, self.viewpoints[closest_keyframe])\n',
'''                            # PREDICTIVE4D_SCENE_FLOW: use official GenFlow3D packets when valid.
                            from predictive4d.flow4dgs_native import apply_dynamic_packet
                            applied = apply_dynamic_packet(self, cur_frame_idx, closest_keyframe)
                            if not applied:
                                print(f'PREDICTIVE4D_NATIVE_FLOW_FALLBACK frame={cur_frame_idx}')
                                self.apply_flow_offset(flow, viewpoint, self.viewpoints[closest_keyframe])
                            else:
                                print(f'PREDICTIVE4D_GENFLOW_APPLIED frame={cur_frame_idx}')
''', 'backend scene-flow offset')
    # Tested 194-frame Colab fix: mapping sampled a shorter viewpoint_stack
    # than current_window while subsequently indexing by current_window length.
    g = replace_once(g,
        'self.viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in key_opt]',
        'self.viewpoint_stack = [self.viewpoints[kf_idx] for kf_idx in current_window]',
        'backend mapping window')
    g = replace_once(g, 'current_window_set = set(key_opt)',
                     'current_window_set = set(current_window)', 'backend random-view exclusion')
    # Official Bonn config refers to a base_config.yaml that is absent.
    y = replace_once(y, 'configs/rgbd/bonn/base_config.yaml',
                     'configs/rgbd/bonn/config.yaml', 'Bonn YAML inheritance')
    ast.parse(f, filename=str(a)); ast.parse(g, filename=str(b))
    return {a: f, b: g, cfg: y}


def patch(upstream: Path, apply: bool = False) -> dict:
    proposed = plan(upstream)  # Validate everything before writing.
    changed = {str(p): p.read_text() != updated for p, updated in proposed.items()}
    if apply:
        for path, updated in proposed.items():
            if path.read_text() == updated:
                continue
            backup = path.with_suffix(path.suffix + '.before_predictive4d')
            if not backup.exists():
                shutil.copy2(path, backup)
            path.write_text(updated)
    return {'applied': apply, 'changes': changed}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--apply', action='store_true')
    args = p.parse_args()
    print(patch(args.upstream, args.apply))
