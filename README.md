# Predictive Motion-aware 4D Gaussian SLAM — Final Architecture Scaffold

**Final research ARCHITECTURE, not a validated full-paper implementation.**

The project follows the Notion research plan: an official GenFlow3D neural encoder + recurrent flow decoder, a separate **learned camera SE(3) head receiving the same encoder features**, probabilistic static/dynamic/outlier decomposition, local Schur-complement joint geometric optimization, observability-aware non-rigid Gaussian propagation, guarded dynamic-Gaussian-to-camera feedback, and native Flow4DGS tracking/mapping hooks.

The default `configs/joint.yaml` is a runnable **compact neural test configuration**. `configs/genflow_joint.yaml` chooses the real upstream GenFlow3D shared model when its trained checkpoint and CUDA PointNet2 are available. The native integration uses the actual Flow4DGS-SLAM code; unlike the compact path, it is **not tested end-to-end here**.

**Design authority:** `docs/FINAL_NOTION_ALIGNMENT.md` is the mapping between the Notion formulas and implemented module boundaries.

## 0. Project layout

```text
src/predictive4d/
    genflow_shared.py       # REAL upstream GenFlow3D encoder+decoder + neural camera head
    network.py              # fast compact neural mock/baseline, not paper method
    tracking.py             # causal neural prior, gated dynamic feedback
    probabilistic.py        # static / dynamic / outlier posterior
    joint_ba.py             # local SE(3) + per-point flow Schur elimination
    observability.py        # static information matrix, weak directions
    gaussians.py            # rank-aware local nonrigid center/covariance propagation
    pipeline.py             # standalone Torch Gaussian reference smoke runner
    flow4dgs_online.py      # one-pass native frontend: use frames < t only
    flow4dgs_native.py      # native camera/BA/feedback and Gaussian offset hooks
    official_bridge.py      # explicit packet schema and native Gaussian ID matching
    torch_renderer.py       # compact differentiable renderer, not native CUDA
scripts/patch_flow4dgs.py # guarded patch for official frontend + backend
configs/genflow_joint.yaml
configs/flow4dgs_native_extra.yaml
```

## 1. VS Code (Windows) — develop/test

Use a VS Code terminal opened at this project's folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest -q
```

These tests do not require the upstream CUDA repos. The official CUDA modules must run on compatible **Linux + CUDA** (AutoDL / a compatible remote server), ideally through VS Code Remote SSH. Do not overwrite a previously working Flow4DGS CUDA environment.

## 2. Quick pipeline check (NOT evidence of final model accuracy)

```powershell
predictive4d train-synthetic --steps 200 --points 256 --history 3 --out checkpoints/compact_synthetic.pt
python scripts/make_demo_rgbd.py --out data/demo_rgbd --frames 8
predictive4d run --data data/demo_rgbd --intrinsics data/demo_rgbd/intrinsics.yaml --checkpoint checkpoints/compact_synthetic.pt --config configs/joint.yaml --frames 8 --out outputs/demo
```

The synthetic checkpoint is **not** a real indoor neural camera checkpoint. The small demonstration is only to validate API, causal timing, geometry, mapping, output format and stability. No baseline ATE/FPS claims should be drawn from it.

## 3. Official GenFlow3D neural camera + dynamic prediction (Linux CUDA)

Clone the ORIGINAL neural architecture next to this repo:

```bash
git clone https://github.com/ustc-hlli/GenFlow3D.git
```

Follow its original PointNet2/CUDA install and obtain the correct GenFlow3D checkpoint. Edit `configs/genflow_joint.yaml` to use exact absolute paths; its network must use `n=8192` or the compatible original configuration.

The separate Camera Head **has no official pretrained weights**. Train the extra camera/dynamic heads to test wiring (synthetic supervision only):

```bash
predictive4d train-official-head-synthetic \
  --genflow-repo /absolute/path/GenFlow3D \
  --genflow-checkpoint /absolute/path/GenFlow3D/pretrain/GenFlow3d_nuscenes.pth \
  --out checkpoints/official_camera_head.pt \
  --device cuda --points 8192 --history 3 --steps 50
```

**This synthetic head training is only a functional smoke.** For your actual research, you must train/calibrate the Camera Head on real, independent motion-labeled RGB-D training scenes; `train-real-camera` is currently available for the compact head but real-data training of the official head still needs an additional dataset adapter/training loop. Use no evaluation GT in online SLAM.

## 4. Official Flow4DGS-SLAM tracking + mapping integration

Clone the original CUDA backend separately and install its dependencies in a compatible Linux CUDA environment:

```bash
git clone https://github.com/wangys16/Flow4DGS-SLAM.git
```

Run a DRY RUN, inspect the diff, then patch only if upstream source matches verified anchors:

```bash
python scripts/patch_flow4dgs.py --upstream ../Flow4DGS-SLAM
python scripts/patch_flow4dgs.py --upstream ../Flow4DGS-SLAM --apply
```

Add the `predictive4d:` section from `configs/flow4dgs_native_extra.yaml` to the native Flow4DGS experiment YAML. Set `model_params.flow_cam: false`, otherwise its original camera-flow initialization overwrites the neural pose. Set **real seconds per processed frame** and absolute paths for `checkpoint`, `model_config`, and `packet_dir`.

Set `online: true`: after warm-up, the native frontend reads **only completed historical frames** and uses the official GenFlow3D shared network to predict the new camera prior and world Scene Flow. With `native_joint_refine: true`, current RGB-D is then used to compute a static/dynamic/outlier posterior and Schur pose correction, with optional historical-native-dynamic-Gaussian feedback, before native Gaussian rendering refinement. The native backend can use predicted flow in place of its optical-flow-derived per-keyframe **center** offsets after validating native dynamic indices and frame times. Native GMM/rotation/opacity/rendering remain those of upstream Flow4DGS.

**Do not assume this has passed native multiprocess CUDA validation.** The native hooks are compiled and tested against mock interfaces, and code anchor patterns were checked against the upstream source, but actual downstream tracker/mapper behavior requires tests with upstream versions and GPU binaries. If an optional hook finds absent/mismatched native state it will not silently invent arrays; correct the integration before reporting results.

The current offset injection does **not** remove all remaining upstream RAFT operations, nor implement a full optimized native Gaussian rotation/scale prior. See `docs/FINAL_NOTION_ALIGNMENT.md`.

## 4.1. Tracking and native integration safety fixes (2026-11-06)

The repository now applies the following fail-closed behavior:

- Ill-conditioned, singular or nonfinite local SE(3) solves **cannot** be relabeled `joint_ok` at the end of the optimizer. These frames do not update the map.
- Per-frame motion posteriors cannot be downgraded by a weaker neural-only motion hint. A production probabilistic fusion model remains a research task.
- A `tracking_lost` frame is output as a held pose but is **not inserted into world-anchored training/prediction history**; robust relocalization after prolonged loss is still unimplemented.
- `uncertainty_calibrated: false` disables motion-prior contributions to camera pose (both Schur dynamic terms and dynamic Gaussian feedback). It still permits dynamic mapping hypotheses. Flip to `true` **only after** validating uncertainty calibration on independent data; setting the flag alone does not calibrate a model.
- Native Flow4DGS `FrontEnd.cleanup()` must preserve sampled historical RGB-D point clouds before `Camera.clean()` removes depth and image; the patcher now inserts a cache hook in addition to the tracking and backend hooks.
- Native Gaussian center offsets are composed across **all intermediate per-frame prediction packets** between keyframes, with time-order, motion, and correspondence gates. A missing intermediate packet causes native fallback rather than applying only the last-frame displacement.

These issues have regression tests, but the actual official CUDA + native multiprocessing integration remains unverified. The native patcher must be re-applied to a clean upstream copy; existing one-hook patches should be carefully reviewed before adding new hooks.

## 5. Test and evaluation safeguards

- All model-predicted world flow is in meters for the specified `dt`, not pixels or camera flow.
- Native camera transforms use `T_cw` (world to camera), not `T_wc`.
- Predictions are made without seeing RGB-D at the target timestamp. Target-frame observations arrive only in the subsequent tracker/mapper.
- Our local joint optimizer marginalizes dynamic flow via a Schur complement. It is **not** equivalent to a full multi-frame camera/object/point factor graph.
- Feedback uses *historical* dynamic Gaussians, and re-used current observation IDs are removed from separate dynamic matches. Uncertainty values remain proxy until calibrated.
- Before quantitative claims, obtain real trained Camera Head, domain-adapted Scene Flow, calibrated uncertainty, and independent ATE/RPE/EPE/PSNR evaluations.

## References

Official GenFlow3D: https://github.com/ustc-hlli/GenFlow3D

Official Flow4DGS-SLAM: https://github.com/wangys16/Flow4DGS-SLAM
