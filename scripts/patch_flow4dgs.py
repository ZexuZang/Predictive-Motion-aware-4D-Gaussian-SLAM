"""Insert version-checked neural prediction hooks into official Flow4DGS code.

Usage:
  python scripts/patch_flow4dgs.py --upstream ../Flow4DGS-SLAM
  python scripts/patch_flow4dgs.py --upstream ../Flow4DGS-SLAM --apply

Patch requires exact anchors from the inspected upstream main branch; fails
closed on incompatible revisions. --apply writes only two upstream .py files.
"""
from pathlib import Path
import argparse,ast,difflib

FRONTEND_ANCHOR='        viewpoint.update_RT(prev.R, prev.T)\n'
FRONTEND_REPLACEMENT='''        viewpoint.update_RT(prev.R, prev.T)
        # PREDICTIVE4D_FRONTEND_HOOK (pretrained joint camera SE3 prior)
        from predictive4d.flow4dgs_native import apply_camera_packet
        from predictive4d.flow4dgs_online import predict_online_prior
        predict_online_prior(self, cur_frame_idx, viewpoint)
        apply_camera_packet(self.config, viewpoint, cur_frame_idx, self)
'''
BACKEND_ANCHOR='                            self.apply_flow_offset(flow, viewpoint, self.viewpoints[closest_keyframe])\n'
BACKEND_REPLACEMENT='''                            # PREDICTIVE4D_BACKEND_HOOK (world flow -> native dynamic Gaussian offsets)
                            from predictive4d.flow4dgs_native import apply_dynamic_packet
                            if not apply_dynamic_packet(self, cur_frame_idx, closest_keyframe):
                                self.apply_flow_offset(flow, viewpoint, self.viewpoints[closest_keyframe])
'''

def patch_one(file,anchor,replacement,apply=False):
    if not file.is_file():raise FileNotFoundError(file)
    old=file.read_text(encoding='utf-8')
    if replacement.strip() in old:return 'already patched'
    if old.count(anchor)!=1:raise RuntimeError(f'Expected exactly one patch anchor in {file}; got {old.count(anchor)}')
    new=old.replace(anchor,replacement,1)
    ast.parse(new,filename=str(file))
    diff=''.join(difflib.unified_diff(old.splitlines(keepends=True),new.splitlines(keepends=True),
                         fromfile=str(file),tofile=str(file)+' (candidate)'))
    if apply:file.write_text(new,encoding='utf8')
    return diff

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--apply',action='store_true')
    a=p.parse_args()
    root=a.upstream
    pairs=[(root/'utils'/'slam_frontend.py',FRONTEND_ANCHOR,FRONTEND_REPLACEMENT),
           (root/'utils'/'slam_backend.py',BACKEND_ANCHOR,BACKEND_REPLACEMENT)]
    # First validate all source files before ANY write.
    for f,anchor,replacement in pairs:patch_one(f,anchor,replacement,apply=False)
    for f,anchor,replacement in pairs:
        print(patch_one(f,anchor,replacement,apply=a.apply))
    print('Status:', 'patched' if a.apply else 'dry-run only')

if __name__=='__main__':main()
