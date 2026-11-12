# Predictive Motion-aware 4D Gaussian SLAM

**CV+GRU Camera Trajectory Prediction and GenFlow3D-guided Dynamic 4D Gaussian Reconstruction**

A research framework extending [Flow4DGS-SLAM](https://github.com/wangys16/Flow4DGS-SLAM) with learned camera-motion prediction and multi-frame scene-flow forecasting for dynamic RGB-D reconstruction.

**Status:** Research prototype with a completed 194-frame native SLAM experiment. Further ablations and independent validation are ongoing.

---

## 1. Overview

This project explores the integration of two predictive components into a native 4D Gaussian SLAM system:

1. **CV+GRU Camera Trajectory Estimation:** Predicts the next camera pose using a constant-velocity SE(3) prior corrected by a learned acceleration GRU.
2. **GenFlow3D Dynamic Motion Prediction:** Uses historical 3D point clouds to predict scene motion and propagate dynamic Gaussian centers.

The system retains the original Flow4DGS-SLAM tracking, Gaussian mapping, and differentiable rendering backend.

### Overall Pipeline

```text
                 Historical RGB-D Frames
                           |
              Historical Camera Poses
                           |
              +------------+------------+
              |                         |
       Camera Motion History     Multi-frame Point Clouds
              |                         |
     Constant Velocity (CV)          GenFlow3D
              |                   (BDCB / STIB)
         GRU Acceleration                |
              |                  Scene Flow Forecast
              |                         |
       Predicted Camera Pose       Dynamic Point Motion
              |                         |
              |                  Gaussian Correspondence
              |                         |
              |                  Gaussian Center Offsets
              |                         |
              +------------+------------+
                           |
                 Native Flow4DGS-SLAM
                           |
              Tracking + 4D Gaussian Mapping
                           |
                   Dynamic 4D Reconstruction
```

## 2. Core Components

### 2.1 CV+GRU Camera Trajectory Estimation

The camera module combines constant-velocity extrapolation with a learned acceleration correction.

The constant-velocity prediction is:

\[
T_t^{CV}
=
\operatorname{Exp}
\left(
\frac{\Delta t}{\Delta t_{t-1}}
\operatorname{Log}(T_{t-1}T_{t-2}^{-1})
\right)T_{t-1}
\]

A GRU learns an effective 6-DoF acceleration correction from the historical camera motion:

\[
\hat{\mathbf a}_t =
f_{\mathrm{GRU}}(\mathcal H_{t-1})
\]

The final camera prior is:

\[
\hat T_t =
\operatorname{Exp}
\left(\frac{1}{2}\Delta t^2
G(\hat{\mathbf a}_t)\right)
T_t^{CV}
\]

where \(G\) applies the acceleration gates.

**Implementation:**

- `src/predictive4d/cv_acceleration.py`
- `src/predictive4d/native_pose_compat.py`
- `src/predictive4d/paper_native.py`

The current model uses six historical camera poses, a 13-dimensional motion-feature representation, and a trained `CVAccelerationGRU-v1` checkpoint.

The camera prior is causal: it does not use future RGB-D frames or evaluation ground-truth poses.

### 2.2 GenFlow3D-guided Dynamic Motion Prediction

Instead of relying exclusively on optical flow and depth-based motion propagation, the proposed system incorporates the official GenFlow3D network.

GenFlow3D processes historical point clouds and estimates future 3D point motion.

For a historical point \(\mathbf X_{i,t-1}\):

\[
\hat{\mathbf X}_{i,t}
=
\mathbf X_{i,t-1}
+
\hat{\mathbf V}_{i,t}
\]

The predicted point motion is subsequently transferred to native dynamic Gaussians through a correspondence adapter:

\[
\boldsymbol\mu_{i,t}
=
\boldsymbol\mu_{i,t-1}
+
\Delta\boldsymbol\mu_{i,t}
\]

**Implementation:**

- `src/predictive4d/paper_native.py`
- `src/predictive4d/flow4dgs_native.py`
- `src/predictive4d/official_bridge.py`

The model uses the official GenFlow3D implementation, including its temporal modeling architecture.

**Important:** GenFlow3D currently predicts point-level Scene Flow. Gaussian center updates are produced by mapping this Scene Flow onto existing dynamic Gaussians; the network has not been retrained to directly output Gaussian parameters.

### 2.3 Dynamic Mask Generation

The completed native experiment uses the original Flow4DGS motion segmentation pathway:

- YOLO-based object segmentation.
- RAFT optical flow.
- Depth-based rigid motion estimation.
- Optical-flow residual segmentation.

The resulting dynamic mask is used for dynamic point selection and Gaussian motion updates.

An independent **GenFlow3D-guided probabilistic motion mask**, together with optional Pre-Ours mask fusion, is a planned extension rather than a validated feature of the completed experiment.

### 2.4 Native 4D Gaussian Reconstruction

The original Flow4DGS backend is preserved for:

- Camera tracking and rendering-based pose refinement.
- Static and dynamic Gaussian mapping.
- Keyframe-based dynamic Gaussian center offsets.
- Time-dependent Gaussian representation.
- Native differentiable rendering and evaluation.

Predicted motion is applied when the required correspondences and validity checks succeed. Otherwise, the original RAFT-based propagation remains available as fallback.

## 3. Repository Structure

```text
Predictive-Motion-aware-4D-Gaussian-SLAM/
|
+-- configs/
|   +-- joint.yaml
|   +-- genflow_joint.yaml
|   +-- flow4dgs_native_extra.yaml
|
+-- src/predictive4d/
|   +-- cv_acceleration.py
|   +-- paper_native.py
|   +-- native_pose_compat.py
|   +-- flow4dgs_native.py
|   +-- flow4dgs_online.py
|   +-- official_bridge.py
|   +-- genflow_shared.py
|   +-- probabilistic.py
|   +-- joint_ba.py
|   +-- tracking.py
|   +-- fusion_head.py
|   +-- gaussians.py
|   +-- pipeline.py
|   +-- network.py
|   +-- torch_renderer.py
|
+-- scripts/
|   +-- setup_native_environment.py
|   +-- patch_flow4dgs.py
|   +-- patch_native_compat.py
|   +-- prepare_genflow3d_imports.py
|   +-- build_bonn_native_config.py
|   +-- audit_native_run.py
|
+-- tests/
+-- docs/
+-- README.md
+-- pyproject.toml
```

## 4. Installation

### 4.1 Local Development

Tested development workflow for VS Code and Python:

```powershell
git clone https://github.com/ZexuZang/Predictive-Motion-aware-4D-Gaussian-SLAM.git

cd Predictive-Motion-aware-4D-Gaussian-SLAM

python -m pip install -e ".[test]"

python -m pytest -q
```

Local development is intended for Python modules, configuration and unit testing.

The native CUDA SLAM implementation requires a compatible Linux/CUDA environment.

### 4.2 Native SLAM Dependencies

Clone the original repositories:

```bash
git clone --recursive https://github.com/wangys16/Flow4DGS-SLAM.git

git clone https://github.com/ustc-hlli/GenFlow3D.git
```

The original experiment used:

- NVIDIA Tesla T4.
- Python 3.10.
- PyTorch with CUDA support.
- Native Gaussian rasterization.
- PyTorch3D KNN.
- GenFlow3D PointNet2 CUDA operations.

Model checkpoints must be downloaded separately.

### 4.3 Apply Native Compatibility Patches

From the Predictive4D repository:

```bash
PY=/content/flow4dgs_env/bin/python

$PY scripts/setup_native_environment.py \
  --upstream /content/Flow4DGS-SLAM \
  --genflow /content/GenFlow3D \
  --python "$PY" --apply
```

This script installs the source-level compatibility changes and native integration hooks.

Use a separate Flow4DGS checkout for the original baseline to avoid modifying its tracking and mapping implementation.

## 5. Native SLAM Experiments

The current native experiment supports two predictive camera modes:

| Mode | Camera Prediction | Dynamic Motion |
|---|---|---|
| Original Flow4DGS | Native camera initialization | RAFT-based propagation |
| CV + GenFlow3D | Constant Velocity | GenFlow3D with native fallback |
| CV+GRU + GenFlow3D | Learned acceleration-corrected CV | GenFlow3D with native fallback |

The experimental configuration can be generated using:

`scripts/build_bonn_native_config.py`

Supported camera options:

- `--mode cv`
- `--mode cv_gru`

The experiment uses a trained CVAccelerationGRU checkpoint and official GenFlow3D pretrained weights.

For detailed native setup and commands, see [Native CV+GRU Colab Reproducibility](docs/NATIVE_CVGRU_COLAB_REPRODUCIBILITY.md).

## 6. Experimental Results

### Dataset

**Bonn RGB-D Person Tracking — stride 3**

- Frames: 194
- Camera trajectory evaluation: ATE and RPE
- Reconstruction evaluation: PSNR, SSIM, LPIPS
- Comparison: Original Flow4DGS-SLAM versus CV+GRU+GenFlow3D

### 6.1 Camera Trajectory Estimation

| Metric | Flow4DGS-SLAM | CV+GRU+GenFlow3D |
|---|---:|---:|
| ATE RMSE (m) ↓ | 0.189424 | **0.091552** |
| RPE Translation (m) ↓ | 0.053590 | **0.043623** |
| RPE Rotation (deg) ↓ | 3.738101 | **3.626036** |

The proposed configuration reduces the common-frame, SE(3)-aligned ATE RMSE by approximately **51.7%** on this sequence.

### 6.2 Rendering Quality

| Metric | Flow4DGS-SLAM | CV+GRU+GenFlow3D |
|---|---:|---:|
| PSNR Before (dB) ↑ | 19.8608 | **21.2417** |
| SSIM Before ↑ | 0.7547 | **0.8115** |
| LPIPS Before ↓ | 0.4216 | **0.3183** |
| PSNR After (dB) ↑ | 24.2283 | **26.3985** |
| SSIM After ↑ | 0.8107 | **0.8550** |
| LPIPS After ↓ | 0.3213 | **0.2376** |

These results are from the completed 194-frame experiment. They have not yet been independently reproduced using a fresh checkout of the latest consolidated code.

### 6.3 Experimental Limitations

The historical audit recorded:

- 2 successful GenFlow3D dynamic Gaussian offset applications.
- 38 native RAFT fallback events.

Therefore, the improvements cannot currently be attributed solely to GenFlow3D.

Furthermore, the Bonn stride-3 sequence was involved in CV+GRU development/training, so this comparison is not an independent held-out evaluation.

The current results are preliminary and require additional ablation studies and evaluation sequences.

## 7. Additional Research Modules

The repository also contains experimental modules for:

**Probabilistic Motion Decomposition**

`probabilistic.py`

Models static, dynamic and outlier hypotheses using geometric residuals and uncertainty proxies.

**Joint Camera–Motion Optimization**

`joint_ba.py`

Provides a local SE(3) optimization approach using a Schur-complement formulation.

**Adaptive Gaussian Motion**

`gaussians.py`

Explores local motion propagation for Gaussian geometry.

**Learned Reliability Fusion**

`fusion_head.py` and `train_fusion.py`

Investigate learned fusion between camera pose candidates.

These are research/reference modules and should not be interpreted as fully validated components of the completed native experiment.

## 8. Future Work

1. Increase GenFlow3D motion-update acceptance and reduce dependence on native RAFT fallback.
2. Evaluate CV-only, CV+GRU-only and GenFlow3D-enabled configurations independently.
3. Develop GenFlow3D-guided probabilistic dynamic mask generation.
4. Integrate optional external dynamic-mask priors with confidence-aware fusion.
5. Improve predicted point-motion to Gaussian-motion correspondence.
6. Test on independent RGB-D sequences with stronger dynamic motion.
7. Evaluate full SLAM accuracy, rendering quality, dynamic reconstruction and computational efficiency.

## 9. References

- [Flow4DGS-SLAM — Official GitHub Repository](https://github.com/wangys16/Flow4DGS-SLAM)
- [GenFlow3D — Official GitHub Repository](https://github.com/ustc-hlli/GenFlow3D)
- [Native Colab Reproducibility Guide](docs/NATIVE_CVGRU_COLAB_REPRODUCIBILITY.md)
- [Architecture and Implementation Status](docs/FINAL_NOTION_ALIGNMENT.md)

---

**Project repository:** [Predictive-Motion-aware-4D-Gaussian-SLAM](https://github.com/ZexuZang/Predictive-Motion-aware-4D-Gaussian-SLAM)

**Research status:** Native 4D Gaussian SLAM integration and preliminary experimental evaluation completed; comprehensive ablations and held-out validation remain ongoing.
