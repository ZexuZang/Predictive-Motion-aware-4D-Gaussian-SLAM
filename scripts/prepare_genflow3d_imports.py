"""Unambiguously isolate GenFlow3D utils.py from Flow4DGS-SLAM utils/.

Only changes Python import spelling. It does not modify model weights, layers,
CUDA kernels or inference mathematics. Can be safely rerun after a fresh clone.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import shutil


def patch(repo: Path, apply: bool = False) -> dict:
    repo = repo.resolve()
    original = repo / 'utils.py'
    if not original.is_file():
        raise FileNotFoundError(original)
    targets = {}
    for file in ('models.py', 'layers.py', 'units.py'):
        path = repo / file
        if not path.is_file():
            raise FileNotFoundError(path)
        src = path.read_text()
        dst, n = re.subn(r'(?m)^import utils[ \t]*$', 'import genflow3d_utils as utils', src)
        if n == 0 and 'import genflow3d_utils as utils' not in src:
            raise RuntimeError(f'Expected GenFlow3D utils import in {path}')
        if n > 1:
            raise RuntimeError(f'Ambiguous utils imports in {path}')
        targets[path] = dst
    alias = repo / 'genflow3d_utils.py'
    if alias.is_file() and alias.read_bytes() != original.read_bytes():
        raise RuntimeError('GenFlow3D alias differs from upstream utils.py; inspect before overwriting')
    if apply:
        if not alias.exists():
            shutil.copy2(original, alias)
        for path, content in targets.items():
            if path.read_text() != content:
                backup = path.with_suffix(path.suffix + '.before_predictive4d')
                if not backup.exists():
                    shutil.copy2(path, backup)
                path.write_text(content)
    return {'planned': [str(x) for x in targets], 'applied': apply}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--genflow', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(patch(args.genflow, args.apply))
