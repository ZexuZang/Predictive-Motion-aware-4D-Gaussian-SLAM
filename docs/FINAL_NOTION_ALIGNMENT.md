# Final design contracts (Notion-aligned) and honest implementation status

## Intended mathematical flow

```
RGB-D_{<=t-1} -> shared GenFlow3D hierarchical encoder + STIB history
                           |                       |
                           v                       v
                    Camera SE(3) Head      Scene Flow Diffusion Head
                           |                       |
                           +---- neural priors -----+
                                       |
RGB-D_t -> static / dynamic / outlier E step -> local joint SE(3) + Schur
                                       |
                             Gaussian deformation adapter
                                       |
                         historical Gaussian -> pose feedback
                                       |
                       Flow4DGS native differentiable tracking & mapping
```

The network prior at time **t** must contain only frames **< t**. Current
RGB-D is used only *after* the prior has been generated, in data association,
optimization, and mapping. Camera pose is always `T_cw` (world -> camera).
Predicted flow is `V_world` in meters per *actual* frame interval.

| Notion requirement | Implementation file / behavior | Validated status |
| --- | --- | --- |
| Shared Hierarchical Encoder + BDCB/STIB | `genflow_shared.py` loads one *real* upstream `GenFlow_rec_seq`, reuses `unit.encoder` features during its original scene decoder forward; camera/dyn heads added | API tested with surrogate only; real checkpoint/CUDA required |
| Neural Camera SE(3) | `genflow_shared.py` + `tracking.NeuralPredictor`; `se3_exp` from 6D | Structural tests; only synthetic/optional camera GT training available |
| Generative Scene Flow | unmodified upstream GenFlow decoder (in official configuration) | Not validated on indoor RGB-D; pretraining domain gap |
| Three-hypothesis posterior | `probabilistic.decompose`: Gaussian log-likelihood+outlier, logsumexp | Unit tested |
| Joint SE(3)/Flow optimization | `joint_ba.optimize_joint_pose`, per-point motion nuisance elimination (`schur_dynamic_factor`) | Unit tested, **local single-frame GN**, not full sliding-window factor-graph BA |
| Static Gaussian rendering for tracking | `flow4dgs_native.apply_camera_packet` sets native pose; native FrontEnd retains original CUDA rendering optimizer | Hook available; upstream end-to-end CUDA NOT tested |
| Non-rigid Gaussian Motion Adapter | `gaussians.propagate_gaussians`: rank-aware local affine, center+shape; native offset transfer uses native keyframe Gaussian indices | Local rank tests; native full rotation/scale/GMM integration is not complete |
| 4D Gaussian mapping | Native `Flow4DGS-SLAM` backend via patch script, retain native GaussianModel & rasterizer | Integration anchor checked against inspected upstream; CUDA/runtime NOT tested |
| Dynamic-to-camera | `tracking.dynamic_pose_feedback` and native optional joint refinement using native historical dynamic Gaussians; NIS-like gate + static weak-direction test | Tested on synthetic geometric example; covariance calibration incomplete |
| Online causality | `flow4dgs_online.OnlineNeuralRuntime` reads only native completed frame IDs less than t, writes packet before t's native tracking | Mock native online test; real timestamps must be configured |

## Important not-yet-completed SCIENTIFIC requirements

1. Full joint camera/point training with aligned indoor RGB-D scene-flow labels or carefully designed self-supervised motion consistency. Upstream GenFlow weights do *not* train a Camera Head.
2. Train/calibrate pose and flow uncertainty on data independent of evaluation. Native dynamic gating uses covariance proxies until validated; reported confidence is not calibrated.
3. Final production multi-frame factor graph with correlated camera, scene, and map uncertainties. Local Schur GN does not replace true sliding-window BA.
4. Native Gaussian `_delta_xyz` integrates predicted *center offsets*; full native quaternion/scale/GMM parameter priors and covariance-to-native parameter mapping require further code and CUDA validation.
5. Assess the RAFT dependency in the upstream native backend. Applying predicted Flow offset does not automatically remove every RAFT call used by other branches. Only claim RAFT-free once all native paths are audited.
6. Check upstream commit version. A source patch uses specific anchors and intentionally refuses to modify newer upstream files if those anchors change.
7. Validate all native process/queue state, keyframe index alignment, rotation conventions, and physical timestamps on a real GPU. A smoke test in a separate CPU reference pipeline is NOT a full native integration test.

## Experiments needed before paper-level claims

Camera: ATE/RPE, lost-frame %, relocalization. Flow: EPE3D and fast motion/occlusion subsets. Mapping: dynamic PSNR/SSIM/LPIPS, depth and temporal consistency. Robustness: NIS, coverage, singular values of static information, false dynamic feedback, latency/FPS. Ablate: camera head, scene flow head, motion adapter, Schur dynamics, native Gaussian feedback, uncertainty gate, and full native rendering.
