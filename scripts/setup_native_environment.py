"""Apply known native compatibility fixes and verify / optionally install CUDA dependencies.

This script is designed for the existing Colab Python 3.10 + CUDA environment.
It does NOT train the GRU or download models. Source revision checks and full
194-frame functional experiments are separate; no default GT or test shortcuts.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import os
import subprocess
import sys

from patch_flow4dgs import patch as patch_native
from patch_native_compat import patch as patch_compat
from prepare_genflow3d_imports import patch as patch_genflow


def run(cmd, env=None, cwd=None):
    print('RUN:', ' '.join(map(str,cmd)),flush=True)
    subprocess.run([str(c) for c in cmd],env=env,cwd=cwd,check=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--genflow',type=Path,required=True)
    parser.add_argument('--python',type=Path,default=Path(sys.executable))
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--build-cuda',action='store_true')
    parser.add_argument('--pytorch3d-source',type=Path)
    args=parser.parse_args()
    for p in (args.upstream/'slam.py',args.genflow/'models.py',args.python):
        if not p.exists():raise FileNotFoundError(p)
    print('NATIVE PATCH:',patch_native(args.upstream,args.apply))
    print('CUDA/MATPLOTLIB PATCH:',patch_compat(args.upstream,args.apply))
    print('GENFLOW IMPORT PATCH:',patch_genflow(args.genflow,args.apply))
    if not args.build_cuda:
        print('Dry-run or source-only preparation completed. Pass --build-cuda to build extensions.')
        return
    if not args.apply:raise RuntimeError('--build-cuda requires --apply')
    capability=subprocess.check_output([str(args.python),'-c',
          'import torch; assert torch.cuda.is_available(); '
          'print(".".join(map(str,torch.cuda.get_device_capability(0))))'],text=True).strip()
    env=os.environ.copy()
    env.update(TORCH_CUDA_ARCH_LIST=capability,MAX_JOBS='1',CUDA_HOME='/usr/local/cuda')
    print('CUDA architecture:',capability)
    run([args.python,'-m','pip','install','ninja','wheel','packaging'],env=env)
    for module in ('submodules/simple-knn','submodules/diff-gaussian-rasterization'):
        run([args.python,'-m','pip','install','-v','--no-build-isolation','--no-cache-dir',
             '--no-deps',args.upstream/module],env=env)
    run([args.python,'-m','pip','install','-v','--no-build-isolation','--no-cache-dir',
         '--no-deps',args.genflow/'pointnet2'],env=env)
    check=subprocess.run([str(args.python),'-c',
          'from pytorch3d.ops import knn_points'],capture_output=True,text=True,env=env)
    if check.returncode:
        if args.pytorch3d_source is None or not (args.pytorch3d_source/'setup.py').exists():
            raise RuntimeError('PyTorch3D CUDA KNN missing. Provide --pytorch3d-source '
                               'with official source checkout; see docs for C++20 compiler notes. '
                               +check.stderr[-500:])
        run([args.python,'-m','pip','install','-v','--no-build-isolation',
             '--no-cache-dir','--no-deps',args.pytorch3d_source],env={**env,'FORCE_CUDA':'1'})
    # Import is not sufficient: actually invoke the compiled native CUDA kernels.
    verify='''import torch
import simple_knn._C, diff_gaussian_rasterization._C
from pytorch3d.ops import knn_points
from pointnet2 import pointnet2_utils
x=torch.zeros(1,2,3,device='cuda'); y=x.clone()
z=knn_points(x,y,K=1); torch.cuda.synchronize()
assert z.idx.shape==(1,2,1)
print('PASS: Native CUDA modules, PyTorch3D KNN, PointNet2 imports')'''
    run([args.python,'-c',verify],env=env)
    print('Native environment ready. Strict GenFlow and GRU weights validation requires your checkpoints.')

if __name__=='__main__': main()
