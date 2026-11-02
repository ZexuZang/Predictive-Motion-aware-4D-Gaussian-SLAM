# Predictive Motion-aware 4D Gaussian SLAM

> **Research prototype, not a reproduction of the complete Flow4DGS-SLAM + GenFlow3D research system.** The repository contains a runnable reference implementation for developing and testing the proposed pipeline. It does not currently reproduce the performance or all components of either original paper.

## 0. Implementation status and limitations (read this first)

This repository implements a runnable **RGB-D → geometric association → motion prediction → three-class motion probabilities → iterative SE(3) pose estimation → Gaussian primitive mapping → local center/covariance deformation → 2D Gaussian splat preview → trajectory/map export** reference pipeline.

Two motion-prediction backends are available:

- **`history_velocity` (default):** A geometry-based temporal velocity baseline that requires neither a neural network nor RAFT. **It is not GenFlow3D.** Use it to verify interfaces, coordinate conventions, temporal causality, and pipeline integration.
- **`genflow_official` (optional):** An adapter to the official `models.GenFlow_rec_seq` network and its pretrained weights for predicting pointwise future scene flow. It requires a compatible CUDA/PyTorch/PointNet2 environment and careful validation of the outdoor-LiDAR-to-indoor-RGB-D domain gap. **Enabling this backend does not automatically integrate the complete original Flow4DGS CUDA mapping system.**

The static and dynamic maps consist of actual Gaussian primitives (mean, covariance, color, and opacity). They can be exported and visualized with an approximate CPU alpha-splat preview. However, **this is not the official CUDA-based, differentiable 3D Gaussian Splatting trainer**. The complete Flow4DGS GMM temporal opacity/rotation model, Gaussian densification, and native differentiable CUDA rasterizer have not yet been integrated.

The default association method is 3D nearest-neighbor matching via SciPy KDTree, **not a trained Bidirectional Correlation Block (BDCB)**. The default uncertainty values are manually specified proxy variances, **not calibrated predictive covariances**. These limitations must be addressed and validated in a full research implementation.

**Do not describe this starter as a complete paper reproduction or claim state-of-the-art performance based on its current outputs.**

## 1. Project structure

```text
Predictive-Motion-aware-4D-Gaussian-SLAM/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── reference.yaml               # CPU / geometric reference pipeline
│   ├── genflow_official.yaml        # Optional official GenFlow3D GPU adapter
│   └── bonn_intrinsics_example.yaml # EXAMPLE ONLY; not calibrated values
├── src/predictive4d/
│   ├── cli.py                       # Command-line entry point
│   ├── data.py                      # Bonn/TUM-style RGB-D dataset loader
│   ├── synthetic.py                 # Synthetic end-to-end smoke-test sequence
│   ├── geometry.py                  # SE(3), projection, and Gauss-Newton utilities
│   ├── association.py               # Causal 3D nearest-neighbor baseline
│   ├── prediction.py                # Historical velocity / official GenFlow3D adapter
│   ├── pose_net.py                  # Optional trainable temporal pose GRU
│   ├── pose_prior.py                # Constant-velocity or trained pose-GRU prior
│   ├── motion.py                    # Static/dynamic/outlier probabilities and pose optimization
│   ├── gaussian.py                  # Gaussian primitives and non-rigid motion adapter
│   ├── feedback.py                  # Uncertainty-gated dynamic Gaussian pose feedback
│   ├── render.py                    # CPU ellipse alpha splatting and optional numeric refinement
│   ├── pipeline.py                  # Causal pipeline orchestration and export
│   └── evaluate.py                  # TUM trajectory format and Horn-aligned ATE
├── scripts/
│   ├── train_pose_prior.py          # Optional supervised training; needs separate GT train split
│   ├── run_colab.sh
│   └── upload_windows.ps1          # Review target URL and files before execution
├── docs/
│   └── OFFICIAL_INTEGRATION.md
└── tests/
    ├── test_geometry.py
    └── test_smoke.py
```

## 2. First run: no CUDA or GenFlow3D required

Run the following commands from the project root in Windows PowerShell or a Linux/macOS terminal.

Create a virtual environment:

```bash
python -m venv .venv
```

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

**Linux / Colab:**

```bash
source .venv/bin/activate
python -m pip install -e ".[test]"
```

Run the unit tests:

```bash
python -m pytest -q
```

Generate a small synthetic RGB-D sequence and run the complete reference pipeline:

```bash
predictive4d make-demo --out data/synthetic_demo --frames 7
predictive4d run --data data/synthetic_demo --intrinsics data/synthetic_demo/intrinsics.yaml --dynamic-mask-dir data/synthetic_demo/dynamic_masks --config configs/reference.yaml --frames 7 --out outputs/synthetic
```

Expected output structure:

```text
outputs/synthetic/
├── trajectory_est.tum          # Exported camera-to-world poses in TUM format
├── camera_poses.npz            # Internally estimated world-to-camera T_cw and timestamps
├── metrics.json                # Horn-aligned ATE when matching GT timestamps are available
├── frame_records.json          # Per-frame diagnostic information
├── previews/000000.png ...     # Approximate CPU Gaussian-splat previews
├── static_maps/000000.npz ...  # Per-frame static Gaussian primitives
└── dynamic_maps/000000.npz ... # Per-frame dynamic Gaussian primitives
```

**Important:** The demo uses foreground masks generated by the synthetic-data generator to exercise the dynamic Gaussian branch. Without an external mask, the reference pipeline relies on its geometric/depth residuals. Passing the smoke test proves that the software components execute together; it **does not** demonstrate research-level accuracy, GenFlow3D generalization to indoor scenes, or high-quality non-rigid reconstruction. Always validate camera intrinsics and depth units before testing a real sequence.

## 3. Running on Bonn `person_tracking_stride3`

Example Colab dataset directory:

```text
/content/drive/MyDrive/Datasets/Bonn/rgbd_bonn_person_tracking_stride3
```

Obtain the actual calibrated `fx`, `fy`, `cx`, and `cy` values from the dataset calibration or from your already-verified Flow4DGS configuration. Save them to `configs/bonn_intrinsics.yaml`. Do **not** use the placeholder example as if it contained valid calibration.

```yaml
fx: YOUR_CALIBRATED_FX
fy: YOUR_CALIBRATED_FY
cx: YOUR_CALIBRATED_CX
cy: YOUR_CALIBRATED_CY
```

Example command:

```bash
predictive4d run \
  --data /content/drive/MyDrive/Datasets/Bonn/rgbd_bonn_person_tracking_stride3 \
  --intrinsics configs/bonn_intrinsics.yaml \
  --config configs/reference.yaml \
  --depth-scale 5000 \
  --frames 20 \
  --out outputs/bonn_20
```

**Verify `--depth-scale` for your files.** A `uint16` depth image storing millimeters typically requires `1000`, whereas TUM/Bonn-style depth encoding sometimes uses `5000` units per meter. The correct setting depends on the actual dataset files and calibration. Inspect valid raw depth values and verify that converted depths are in a plausible range of meters.

To run an **external-mask ablation** with your existing Pre-Ours masks, append:

```bash
--dynamic-mask-dir /path/to/your/mask_pngs
```

The mask PNG filename stems must match those of the corresponding RGB images. This option does **not** validate an autonomous, RAFT-free motion-segmentation method.

If `associations.txt` follows the four-column TUM convention (`rgb_timestamp rgb_path depth_timestamp depth_path`), the loader uses it. Otherwise, it attempts timestamp association from `rgb.txt`/`depth.txt` or filename-based matching.

Start with a 10–20-frame smoke test before running the entire sequence. Inspect `metrics.json`, `frame_records.json`, and the generated previews.

> **Performance caveat:** The CPU Gaussian renderer in this reference implementation is primarily a map-state visualization tool, not a substitute for the complete differentiable Flow4DGS-SLAM renderer. `render_pose_refinement: true` performs slow numerical differentiation; leave it disabled for the initial 194-frame experiments.

## 4. Connecting the official GenFlow3D network (GPU)

1. On a CUDA-capable machine, clone the [official GenFlow3D repository](https://github.com/ustc-hlli/GenFlow3D) separately.
2. Follow its README to establish a compatible PyTorch/CUDA environment and compile the PointNet2 extension. Verify that the official `GenFlow_rec_seq` model runs and loads its checkpoint in the upstream repository **before** integrating it here.
3. Place the authorized `GenFlow3d_nuscenes.pth` checkpoint in the upstream repository's `pretrain/` directory. **Do not commit pretrained model weights to GitHub.**
4. Set `genflow_repo`, `genflow_checkpoint`, and `genflow_device` in `configs/genflow_official.yaml`.
5. Run a small test:

```bash
predictive4d run \
  --data /path/to/bonn_sequence \
  --intrinsics /path/to/real_intrinsics.yaml \
  --config configs/genflow_official.yaml \
  --frames 10 \
  --out outputs/genflow_smoke
```

The adapter prepares **three historical frames**, **8,192 sampled points per frame**, and `feats=xyz`. In evaluation mode, it uses the final predicted future-flow tensor, `flow_preds[0][-1]`, and transfers that flow to working points by nearest-neighbor interpolation. The upstream `forward(pcs, feats, gts)` signature still expects a `gts` argument; the adapter supplies correctly shaped zero placeholders during evaluation, where actual flow ground truth is not used as training supervision.

**Manually validate** the output coordinate frame, motion direction/sign, temporal step, spatial scale, and indoor-RGB-D versus outdoor-LiDAR distribution shift. Neither correct tensor shapes nor a successful forward pass guarantee meaningful motion predictions on Bonn.

If loading the official network fails, this backend should report the error instead of silently falling back to `history_velocity`.

## 5. Learning-based camera pose head (optional)

The default pose predictor is a training-free constant-velocity model. If you have an **independent training sequence** and valid camera-pose ground truth, you can train the included lightweight PointNet + GRU pose head:

```bash
python -m pip install torch
PYTHONPATH=src python scripts/train_pose_prior.py \
   --data /path/to/TRAINING_sequence \
   --intrinsics /path/to/TRAIN_calibration.yaml \
   --gt /path/to/TRAIN_groundtruth.txt \
   --out checkpoints/pose_gru_train_only.pt \
   --epochs 30
```

The `PYTHONPATH=src` syntax above is for Linux/macOS. In Windows PowerShell, set the variable first with `$env:PYTHONPATH="src"`, then run `python scripts/train_pose_prior.py` with the same arguments.

Enable it in YAML:

```yaml
pose_backend: pose_gru
pose_checkpoint: /absolute/path/to/checkpoints/pose_gru_train_only.pt
pose_device: cuda
```

**Avoid evaluation leakage:** Do not train the pose head on Bonn ground-truth test trajectories and then report ATE on those same frames. The included pose GRU has not been formally trained or validated and does **not** share weights with the official GenFlow3D Encoder. A genuinely shared feature encoder requires a dedicated architectural and training integration; see `docs/OFFICIAL_INTEGRATION.md`.

## 6. Enabling reliability-gated dynamic pose feedback (research option)

Set the following fields in a configuration file:

```yaml
dynamic_feedback_enabled: true
feedback_only_weak_static: false
```

This is a **reference implementation**, not the complete proposed method. It associates the dynamic Gaussians **predicted from historical information** with current points, applies an approximate normalized-innovation gating test, retains a strong pose-initialization prior, and rejects excessive pose jumps. It is disabled by default so that tracking can be compared with and without this component under otherwise identical conditions.

Still to be implemented and validated: calibrated per-point anisotropic motion covariance, a complete static/dynamic factor graph, rigorous handling of correlated RGB-D evidence, and information-gain gating along weakly observable camera-pose directions. Do not claim that the current switch implements all of the proposed research contributions.

## 7. Proposed research contributions (not yet validated)

1. **Joint Prediction and Probabilistic Motion Decomposition.** Shared geometric representation with separate camera-motion and scene-flow heads; static/dynamic/outlier posterior estimation and iterative SE(3) refinement.
2. **Non-rigid Gaussian Motion Adapter.** Local-affine scene-flow fitting, $F=I+\nabla V$, to propagate Gaussian center, covariance, rotation, and scale, with fallbacks for degenerate geometric neighborhoods.
3. **Reliability-gated Dynamic Pose Feedback.** Use high-confidence, motion-compensated dynamic Gaussian predictions as potential camera-pose constraints when static geometry is weak.

There is related prior art, including **EgoFlowNet, DynoSAM, DynaGSLAM, GP-4DGS, and USplat4D**. These are **candidate contributions**, not demonstrated novelty or verified first-of-their-kind claims. Their exact distinction from previous methods requires systematic literature review and controlled experiments.

## 8. Uploading to your GitHub repository

The intended repository name shown in the GitHub creation screen is `Predictive Motion-aware 4D Gaussian SLAM`. GitHub may produce a slug such as `Predictive-Motion-aware-4D-Gaussian-SLAM`. **After creating the repository, copy its actual remote URL from the green Code menu. Do not infer that the repository already exists solely from the creation form.**

Open the extracted project root in VS Code. Then, in PowerShell:

```powershell
git init
git branch -M main
git add .
git status
git commit -m "Initial research pipeline"
git remote add origin https://github.com/ZexuZang/Predictive-Motion-aware-4D-Gaussian-SLAM.git
git push -u origin main
```

Replace the URL after `git remote add origin` if the actual GitHub repository address is different. If the remote repository already contains an initial README or any other commit, avoid pushing unrelated histories: clone the repository and copy the project files into it, or deliberately reconcile the histories before pushing.

**Never commit** datasets, pretrained checkpoints, private filesystem paths, credentials, or generated `.npz` outputs. The provided `.gitignore` excludes many such files, but always inspect `git status` before committing.

## 9. Recommended development milestones

- **M0:** Pass unit tests and the synthetic demo; complete a 10-frame Bonn smoke test.
- **M1:** Integrate the official GenFlow3D network and verify coordinate conventions, scale, and temporal alignment.
- **M2:** Evaluate geometric correspondence reliability and static/dynamic classification.
- **M3:** Compare center-only Gaussian propagation against local-affine covariance propagation.
- **M4:** Train a pose head on separate data, or establish that constant velocity is sufficient.
- **M5:** Integrate the official Flow4DGS CUDA Gaussian backend and differentiable mapping.
- **M6:** Calibrate pose/motion uncertainty; extend feedback to a joint factor graph and information-gain-based gating.
- **M7:** Evaluate ATE/RPE, dynamic reconstruction metrics, FPS, motion forecasting, non-rigid generalization, and the full ablation matrix.

For more implementation details, see [`docs/OFFICIAL_INTEGRATION.md`](docs/OFFICIAL_INTEGRATION.md).

## 10. References and licensing

- [Flow4DGS-SLAM](https://github.com/wangys16/Flow4DGS-SLAM)
- [GenFlow3D](https://github.com/ustc-hlli/GenFlow3D)
- [DynoSAM](https://github.com/ACFR-RPG/DynOSAM)

This repository does not vendor third-party source code, pretrained checkpoints, or CUDA extensions. When integrating upstream projects, comply with their respective licenses and model-distribution terms. Until a project license is selected, do not claim that this repository is released under a specific open-source license.
