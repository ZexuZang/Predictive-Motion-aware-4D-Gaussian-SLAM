# Native CV+GRU / GenFlow3D integration — reproducible Colab patch set

## Status, source of fixes, and scientific limitation

This patch package consolidates **actual Colab troubleshooting** from the uploaded executed
`Colab_Bonn_Native_Flow4DGS_GenFlow3D_CV_GRU_Full_Run (1).ipynb`.
It installs a **trained CVAccelerationGRU-v1** architecture and a full native
Flow4DGS front/back-end hook with genuine official `GenFlow_rec_seq` inference.
It does **not** replace CUDA Gaussian rasterization or simplify native Mapping.

Historical working source revisions in that run:

- Research repo `1e21a9ab9eecd9841adfb588e21ed922c252dab2`
- Flow4DGS `ecb4e94c4a3dc3b06430bfb73c528393b3403178`
- GenFlow3D `a19e3790beb079cc87c40d890f696dc4e916b8d1`

**Scientific limitations:** Bonn `rgbd_bonn_person_tracking_stride3` was used for
GRU development/training, so it is **not an independent held-out test**. Historical
completed run: 194 frames, 276 recorded runtime inference rows (not necessarily
276 unique frames), **2** GenFlow3D dynamic Gaussian offset applications, **38**
native RAFT fallback events. These numbers are taken from the Notebook's
post-run audit, and are not claims for a fresh run. Do not claim GenFlow3D drove
all keyframes or that the full proposed probabilistic 4D SLAM is complete.

## Included fixes

| Previously observed error | Consolidated fix |
|---|---|
| Missing camera head in GitHub | `src/predictive4d/cv_acceleration.py`, `paper_native.py` |
| Late-sequence invalid SE(3) in CV-only test | `native_pose_compat.py`: bounded SO(3) projection; rejects corrupt poses |
| GenFlow3D `utils.PointWarping` collision | `scripts/prepare_genflow3d_imports.py` |
| CUDA 13 `lerp` clash | `patch_native_compat.py`: rename CUDA helper overloads |
| Undefined `uint32_t` / `std::uintptr_t` | same: add `<cstdint>`, `<cstddef>`, `<stdint.h>` |
| Matplotlib colorbar `tight_layout` failure | same: skip *layout-only* call |
| Backend `viewpoint_stack` index overflow | `patch_flow4dgs.py`: current-window stack and exclusion aligned |
| Wrong Bonn base YAML inheritance | same: inherit `configs/rgbd/bonn/config.yaml` |
| Predictive hook missing in GitHub patcher | same: complete causal cache, camera, backend hooks |
| Experiment parameters only in notebook | `build_bonn_native_config.py` creates full YAML + native parser timestamps |

Changes to official Flow4DGS source are backed up as `*.before_predictive4d`
(or previous session-specific backups). Only apply the script to a **separate
cloned upstream directory**; do not patch your original Flow4DGS baseline!

## One-time cloning (Colab terminal)

```bash
cd /content
git clone https://github.com/ZexuZang/Predictive-Motion-aware-4D-Gaussian-SLAM.git
git clone --depth 1 --recursive https://github.com/wangys16/Flow4DGS-SLAM.git
git clone --depth 1 https://github.com/ustc-hlli/GenFlow3D.git
# If the GitHub branch has not been updated, copy the patch ZIP's contents
# over /content/Predictive-Motion-aware-4D-Gaussian-SLAM first.
```

To match the original experiment, check out or fetch the pinned commits above
and verify the cloned README, configuration and algorithms before running.
Use Python 3.10 in `/content/flow4dgs_env/bin/python` (the Colab 3.13 kernel is
not the native SLAM's environment). Existing CUDA dependencies remain installed.

## Prepare native sources and GenFlow imports

```bash
cd /content/Predictive-Motion-aware-4D-Gaussian-SLAM
PY=/content/flow4dgs_env/bin/python
$PY scripts/setup_native_environment.py \
  --upstream /content/Flow4DGS-SLAM \
  --genflow /content/GenFlow3D \
  --python "$PY" --apply
$PY -m pip install --no-deps -e .
$PY -m pytest tests/test_cv_acceleration.py tests/test_native_pose_compat.py tests/test_native_patch_repro.py -q
```

If extensions are not already compiled, add `--build-cuda`. PyTorch3D additionally
requires a working official source checkout, `--pytorch3d-source /content/pytorch3d`,
when not installed. PyTorch3D with very new Torch/CUDA/host GCC may need a
compiler-flag compatibility check; never silently downgrade Torch and invalidate
already-built extensions.

Checkpoint locations from the actual experiment:

```text
/content/drive/MyDrive/Fast4D_SLAM/CameraHead_Pilot/cv_gated_acceleration_v1/cv_acceleration_gru_best.pt
/content/GenFlow3D/pretrain/GenFlow3d_nuscenes.pth
/content/Flow4DGS-SLAM/pretrained/raft-things.pth
/content/Flow4DGS-SLAM/pretrained/yolov9e-seg.pt
```

Check *file sizes* and strictly load GenFlow3D and GRU model state dictionaries.
The code never downloads weights or treats random initialization as an experiment.

## Generate native config with timestamps matching the original TUM parser

```bash
PY=/content/flow4dgs_env/bin/python
ROOT=/content/Predictive-Motion-aware-4D-Gaussian-SLAM
$PY "$ROOT/scripts/build_bonn_native_config.py" \
 --python "$PY" \
 --upstream /content/Flow4DGS-SLAM \
 --genflow /content/GenFlow3D \
 --genflow-checkpoint /content/GenFlow3D/pretrain/GenFlow3d_nuscenes.pth \
 --cv-gru-checkpoint /content/drive/MyDrive/Fast4D_SLAM/CameraHead_Pilot/cv_gated_acceleration_v1/cv_acceleration_gru_best.pt \
 --joint-config "$ROOT/configs/genflow_joint.yaml" \
 --data /content/drive/MyDrive/Datasets/Bonn/rgbd_bonn_person_tracking_stride3 \
 --output /content/drive/MyDrive/Fast4D_SLAM/Native_Paper_SLAM_CVGRU \
 --packet-dir /content/native_flow_packets_cv_gru \
 --mode cv_gru \
 --config-out /content/Flow4DGS-SLAM/configs/rgbd/bonn/bonn_native_cv_gru_genflow.yaml
```

Then execute using `/content/flow4dgs_env/bin/python -u slam.py --config <path> --eval --yolo 1 --save_results 1 --iters 50 --exp_name CVGRU_GenFlow3D_repro` with cwd `/content/Flow4DGS-SLAM`.
For CV-only ablation use **a different packet directory and config file** with
`--mode cv`; the model should load the same GenFlow weights and native backend.
Do not reuse packet directories across different experiments.

## Validation / evidence checklist

- All CUDA kernels import, PyTorch3D KNN actually executes on GPU.
- GenFlow3D weights strictly loaded; CVAccelerationGRU-v1 checkpoint verified.
- `model_params.flow_cam=false`; RAFT motion mask still enabled.
- Camera history `<t` only; first pose initialized without GT in modified run.
- Native log contains `PREDICTIVE4D_GENFLOW_APPLIED` and/or explicitly counted fallback.
- End-to-end full 194 frame output contains `plot/trj_final.json`, `psnr/before_opt/final_result.json`, `psnr/after_opt/final_result.json`.
- `runtime_diagnostics.jsonl` contains **unique frame IDs**; deduplicate old runs with repeated records.
- Baseline and predictive experiments use the same GT/initialization convention before a strict ATE attribution claim.
- GenFlow3D predicts **point motion / Scene Flow**; the adapter maps it onto existing dynamic Gaussian center offsets. Do not call this direct learned Gaussian-center output unless a new model was trained for that target.
- `uncertainty_calibrated=false` remains accurate; optional dynamic posterior / nonrigid covariance feedback is *not yet scientifically validated*.

## Why we did not turn off the failing pose check

`native_pose_compat.py` reorthogonalizes a finite 3×3 rotation matrix only if
its discrepancy from SO(3) is small (`<0.02` Frobenius norm). Nonfinite,
reflection and large drifts raise diagnostic errors rather than relying on GT,
maintaining an honest CV-only comparison. The threshold should be recorded and
can be tightened after inspecting the actual sequence.
