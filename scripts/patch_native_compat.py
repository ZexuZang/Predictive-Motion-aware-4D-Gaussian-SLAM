"""CUDA 13/C++20 and renderer compatibility patches for pinned Flow4DGS.

Use only on a cloned upstream checkout. Textual updates are guarded and
idempotent. Run BEFORE installing native CUDA extensions. Does not change
Gaussians, losses, motion models or model parameters.
"""
from __future__ import annotations
import argparse
import ast
from pathlib import Path
import re
import shutil


def _write(path, src, dst, apply):
    if src != dst and apply:
        backup = path.with_suffix(path.suffix + '.before_predictive4d')
        if not backup.exists():
            shutil.copy2(path, backup)
        path.write_text(dst)
    return src != dst


def patch(flow: Path, apply: bool = False) -> dict:
    flow = flow.resolve()
    changes = {}
    helper = flow/'submodules/diff-gaussian-rasterization/cuda_rasterizer/helper_math.h'
    impl = flow/'submodules/diff-gaussian-rasterization/cuda_rasterizer/rasterizer_impl.h'
    for p in (helper, impl):
        if not p.is_file():
            raise FileNotFoundError(p)
    src = helper.read_text()
    dst, count = re.subn(r'\blerp(?=\s*\()', 'cuda_compat_lerp', src)
    if count not in (0, 4) or (count == 0 and dst.count('cuda_compat_lerp(') != 4):
        raise RuntimeError(f'Unexpected helper_math.h lerp overload count: {count}')
    changes[str(helper)] = _write(helper, src, dst, apply)
    src = impl.read_text()
    missing = [x for x in ('#include <cstdint>', '#include <cstddef>', '#include <stdint.h>') if x not in src]
    if missing:
        if src.count('#pragma once') != 1:
            raise RuntimeError('rasterizer_impl.h does not have expected include guard')
        dst = src.replace('#pragma once', '#pragma once\n'+ '\n'.join(missing), 1)
    else:
        dst = src
    changes[str(impl)] = _write(impl, src, dst, apply)
    # Keep Matplotlib render calls; only disable layout manager switch that
    # raises when the current figure contains a colorbar from another layout.
    for rel in ('utils/slam_frontend.py', 'utils/slam_backend.py'):
        p = flow/rel
        src = p.read_text()
        dst, _ = re.subn(r'(?m)^([ \t]*)plt\.tight_layout\(\)[ \t]*(?:#.*)?$', r'\1# plt.tight_layout()  # Matplotlib colorbar compatibility', src)
        ast.parse(dst, filename=str(p))
        changes[str(p)] = _write(p, src, dst, apply)
    return {'applied': apply, 'changed': changes}


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--apply',action='store_true')
    args=p.parse_args()
    print(patch(args.upstream,args.apply))
