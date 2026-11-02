# Predictive Motion-aware 4D Gaussian SLAM

> Research code / 研究原型，**不是已经复现完整 Flow4DGS-SLAM + GenFlow3D 的最终论文系统**。

## 0. 先看清楚实现边界（很重要）

本仓库包含完整、可运行的 **RGB-D → 几何匹配 → 预测 → 三类运动概率 → SE(3) 迭代 → Gaussian primitive map → 局部形变中心/协方差 → 2D Gaussian splat preview → 轨迹/地图输出** 参考流程。

支持两个运动预测器：

- **`history_velocity`（默认）**：不需要神经网络，不使用 RAFT。它是经过时序历史传播的**几何基线**，绝对**不是 GenFlow3D**。仅供验证模块接口、坐标约定与因果数据流。
- **`genflow_official`（可选）**：通过官方 `models.GenFlow_rec_seq` + 官方预训练权重计算点级未来 Scene Flow。必须在正确的 GPU/CUDA/PointNet2 环境中运行，并解决室外 LiDAR → 室内 RGB-D 的领域差异。此模式只是将网络的输出接入本项目的几何/地图前端，**并不把整个 Flow4DGS 原版 CUDA backend 自动替换完成**。

静态与动态 Gaussian 是真实的 mean/cov/color/opacity primitive，具有输出和 CPU 近似 alpha splat 预览，但**不是**官方 CUDA 3DGS 训练器；尚未包含 Flow4DGS 原版 GMM 时间 opacity/rotation、densification、可微 CUDA rasterization 的完整集成。因此本项目**不应使用“完整论文效果已复现”“实现了 SOTA”**的表述。

此外，默认匹配器是 SciPy KDTree 的三维最近邻，不是训练好的 BDCB。正式复现时需要专门验证并替换匹配器，或使用可信第三方 3D matching 模型。默认预测不确定性是手动设定的代理方差，**尚未校准**。

## 1. 完整研究结构

```text
Predictive-Motion-aware-4D-Gaussian-SLAM/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── configs/
│   ├── reference.yaml               # CPU / geometric reference
│   ├── genflow_official.yaml        # optional official GenFlow3D GPU adapter
│   └── bonn_intrinsics_example.yaml # ⚠ EXAMPLE ONLY
├── src/predictive4d/
│   ├── cli.py                       # command line entry
│   ├── data.py                      # RGB-D Bonn/TUM reader
│   ├── synthetic.py                 # end-to-end synthetic smoke sequence
│   ├── geometry.py                  # SE3 + point projection + GN
│   ├── association.py               # causal 3D nearest-neighbour matching baseline
│   ├── prediction.py                # history velocity or official GenFlow3D
│   ├── pose_net.py                  # optional trainable temporal pose GRU
│   ├── pose_prior.py                # constant-velocity or pretrained pose GRU
│   ├── motion.py                    # static/dynamic/outlier E-like step + M-step
│   ├── gaussian.py                  # Gaussian primitive + non-rigid motion adapter
│   ├── feedback.py                  # uncertainty-gated dynamic Gaussian pose factor
│   ├── render.py                    # CPU ellipse alpha splat + optional numeric refine
│   ├── pipeline.py                  # complete causal orchestration + outputs
│   └── evaluate.py                  # TUM trajectory & Horn aligned ATE
├── scripts/
│   ├── train_pose_prior.py          # optional, supervised; requires separate GT training split
│   ├── run_colab.sh
│   └── upload_windows.ps1          # review URL / contents before executing
├── docs/
│   └── OFFICIAL_INTEGRATION.md
└── tests/
    ├── test_geometry.py
    └── test_smoke.py
```

## 2. 第一次验证：不需要 CUDA、不需要 GenFlow3D

在 Windows PowerShell 或 Linux/macOS Terminal 中，从项目根目录执行：

```bash
python -m venv .venv
```

Windows：

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
```

Linux/Colab：

```bash
source .venv/bin/activate
python -m pip install -e ".[test]"
```

运行单元测试：

```bash
python -m pytest -q
```

创建小型合成数据并完整运行：

```bash
predictive4d make-demo --out data/synthetic_demo --frames 7
predictive4d run --data data/synthetic_demo --intrinsics data/synthetic_demo/intrinsics.yaml --dynamic-mask-dir data/synthetic_demo/dynamic_masks --config configs/reference.yaml --frames 7 --out outputs/synthetic
```

输出：

```text
outputs/synthetic/
├── trajectory_est.tum          # world-to-camera 内部估计，输出为 camera-to-world TUM poses
├── camera_poses.npz            # T_cw, timestamps
├── metrics.json                # 如果 GT 时间戳匹配，输出 Horn aligned ATE
├── frame_records.json          # 每帧诊断
├── previews/000000.png ...     # CPU 近似 Gaussian splat rendering
├── static_maps/000000.npz ...  # 每帧静态 Gaussian primitives
└── dynamic_maps/000000.npz ... # 每帧动态 Gaussian primitives
```

**重要**：合成演示使用合成器生成的前景 mask 来验证动态 Gaussian 分支（默认不加载 mask 则只使用几何/深度残差）。合成数据测试证明的是代码链路可运行，**不证明**学术精度、GenFlow3D 的室内泛化能力、非刚性真实重建效果。运行真实 Bonn 时要先检查深度单位和内参。

## 3. 在 Bonn person_tracking_stride3 上执行

已有数据示例（Colab）：

`/content/drive/MyDrive/Datasets/Bonn/rgbd_bonn_person_tracking_stride3`

请从该数据集提供的相机标定文件或你已验证的 Flow4DGS 配置中提取真实 `fx/fy/cx/cy`，保存为 `configs/bonn_intrinsics.yaml`（该文件不是仓库内的示例占位参数）。

```yaml
fx: YOUR_CALIBRATED_FX
fy: YOUR_CALIBRATED_FY
cx: YOUR_CALIBRATED_CX
cy: YOUR_CALIBRATED_CY
```

示例命令：

```bash
predictive4d run \
  --data /content/drive/MyDrive/Datasets/Bonn/rgbd_bonn_person_tracking_stride3 \
  --intrinsics configs/bonn_intrinsics.yaml \
  --config configs/reference.yaml \
  --depth-scale 5000 \
  --frames 20 \
  --out outputs/bonn_20
```

检查 `depth_scale`：深度若是 uint16 以毫米为单位应设为 1000；若是 TUM/Bonn 风格 1m=5000 通常使用 5000，**以实际标定与文件为准**。可用 `numpy` 打印深度中位数，确认转换后在合理米数范围。

如果需要单独测试你已有的 Pre-Ours Mask，可以增加 `--dynamic-mask-dir /path/to/your/mask_pngs`，Mask PNG 的 stem 必须与 RGB 文件名 stem 对齐。该模式是**外部 Mask 消融**，不是端到端自主运动分离。

如果 dataset 的 `associations.txt` 有 TUM 标准 `timestamp rgb/file timestamp depth/file` 四列，代码会读取；否则尝试 `rgb.txt/depth.txt` 时间关联或按文件名配对。

真实序列可能更长，先 10–20 帧 smoke，之后再逐渐提高帧数；处理完成后查看 `metrics.json`、`frame_records.json` 及输出预览。

> ⚠ 此 reference pipeline 的 CPU Gaussian renderer 主要用于显示地图状态；它不是 Flow4DGS-SLAM 完整 differentiable rendering 的性能基准。`render_pose_refinement: true` 是非常慢的数值微分参考，先不要在 194 帧实验中启用。

## 4. 接入真正的 GenFlow3D（GPU）

1. 在一台具备 CUDA GPU 的机器上单独克隆 [GenFlow3D](https://github.com/ustc-hlli/GenFlow3D)。
2. 按官方 README 建立兼容的 PyTorch / PointNet2 CUDA 环境，并编译 PointNet2 扩展。先在原仓库通过官方测试，确认 `GenFlow_rec_seq` 能加载权重。
3. 将可信的 `GenFlow3d_nuscenes.pth` 放到原仓库的 `pretrain/` 目录；**不要把权重提交 GitHub**。
4. 编辑 `configs/genflow_official.yaml` 中的 `genflow_repo`、`genflow_checkpoint`、`genflow_device`。
5. 执行：

```bash
predictive4d run \
  --data /path/to/bonn_sequence \
  --intrinsics /path/to/real_intrinsics.yaml \
  --config configs/genflow_official.yaml \
  --frames 10 \
  --out outputs/genflow_smoke
```

适配器准备了 `L=3` 帧输入、`8192` 点，`feats=xyz`，并使用官方模型在 eval 模式下最后一个 `flow_preds[0][-1]` 作为未来 Scene Flow，然后 KNN 转移到工作点。官方 `forward(pcs,feats,gts)` 的 `gts` 在 eval 下不会作为真实流监督使用，适配器传入形状占位零张量。

**必须人工确认**：模型输出三维运动坐标系/符号、每帧步长、尺度、室外→室内 domain gap、LiDAR vs RGB-D 采样分布。这里的几何输出不能直接当作已经有效的 Scene Flow GT。

如果加载模型失败，不会静默退回 history_velocity；应先修复官方环境。

## 5. 学习式 Camera Pose Head（可选）

默认使用无需训练的恒速度位姿预测。若确有**独立训练序列**和真实 GT，可训练此仓库中的简单 PointNet + GRU Pose Head：

```bash
python -m pip install torch
PYTHONPATH=src python scripts/train_pose_prior.py \
   --data /path/to/TRAINING_sequence \
   --intrinsics /path/to/TRAIN_calibration.yaml \
   --gt /path/to/TRAIN_groundtruth.txt \
   --out checkpoints/pose_gru_train_only.pt \
   --epochs 30
```

然后在 YAML 中启用：

```yaml
pose_backend: pose_gru
pose_checkpoint: /absolute/path/to/checkpoints/pose_gru_train_only.pt
pose_device: cuda
```

**不要用 Bonn 测试轨迹的 GT 训练 Pose Head 后又在相同的 Bonn 帧上报告 ATE**。这个简易 Pose GRU 未经过正式训练与验证，也不共享 GenFlow3D 官方 Encoder 的权重。真正的 Shared Feature Encoder 需要新增训练方案与适配点云网络，见 `docs/OFFICIAL_INTEGRATION.md`。

## 6. 打开可靠性门控的动态 Pose Feedback（研究功能）

在配置文件中：

```yaml
dynamic_feedback_enabled: true
feedback_only_weak_static: false
```

这是参考级动态 Gaussian 反馈：使用历史预测的动态 Gaussian 对当前点云进行 NN 关联和 NIS 近似门控，保留强相机初值先验并拒绝大幅位姿跳跃。默认关闭，可比较相同 RGB-D 数据和参数下的 Tracking 差异。

**未完成部分**：逐点各向异性校准协方差、与静态 Gaussian 联合因子图、对同源 RGB-D 测量的严格相关建模、完整弱方向信息增益门控。不要把此参考开关称为已实现论文全部创新。

## 7. 三项拟验证创新

1. **Joint Prediction & Probabilistic Decomposition**：共享点云特征与独立 Camera/Scene Head；静态/动态/异常值后验与 SE(3) 交替估计。
2. **Non-rigid Gaussian Motion Adapter**：逐点 Scene Flow 的局部仿射拟合 $F=I+\nabla V$，中心/旋转/尺度/协方差传播，退化邻域回退。
3. **Reliability-gated Dynamic Pose Feedback**：利用运动补偿后的可信动态 Gaussian 预测，为静态几何弱约束提供潜在位姿信息。

前人已有相近思想：EgoFlowNet、DynoSAM、DynaGSLAM、GP-4DGS、USplat4D。上述三个是**待实证并检索首创性边界的候选创新点**。

## 8. 上传到自己的 GitHub（建议你亲自完成）

你在网页选择的仓库名称是 `Predictive Motion-aware 4D Gaussian SLAM`。GitHub 自动生成的仓库 slug 可能是：`Predictive-Motion-aware-4D-Gaussian-SLAM`。**请在真正创建完成后，从页面绿色 Code 菜单复制实际 URL，不能只凭截图推测远程仓库已创建。**

在 VS Code 中打开解压后的项目根目录，打开 Terminal（PowerShell）：

```powershell
git init
git branch -M main
git add .
git status
git commit -m "Initial research pipeline"
git remote add origin https://github.com/ZexuZang/Predictive-Motion-aware-4D-Gaussian-SLAM.git
git push -u origin main
```

执行 `git remote add` 之前，请检查项目页面实际 URL 与 GitHub 用户名是否一致。若远程仓库建立时已经有 README 或其他提交，不能直接推送到不相关历史，请先 `git clone` 新建仓库并把代码复制进去，或使用 `git pull --rebase` 处理历史。

**不要提交** datasets、pretrained weights、私人路径、真实凭据、实验输出 `.npz`；本项目 `.gitignore` 已屏蔽大部分相关文件，但上传前仍要运行 `git status` 逐项检查。

## 9. 推荐的阶段顺序

- M0：合成 Demo、单元测试、真实 Bonn 的 10 帧 smoke。
- M1：适配官方 GenFlow3D、检查预测方向/尺度及时间对齐。
- M2：完成几何对应/静态分类的可靠性消融。
- M3：验证 local-affine covariance adapter 与 center-only 对照。
- M4：独立训练 Pose Head（或证明恒速度够用）。
- M5：实现官方 Flow4DGS CUDA Gaussian Backend 的真正可微渲染接口。
- M6：校准运动/位姿不确定性，添加因子图动态反馈与信息增益门控。
- M7：ATE/RPE、动态重建指标、FPS、外推精度、non-rigid 泛化与公开消融。

详见 [`docs/OFFICIAL_INTEGRATION.md`](docs/OFFICIAL_INTEGRATION.md)。

## 10. 参考论文与许可

- [Flow4DGS-SLAM](https://github.com/wangys16/Flow4DGS-SLAM)
- [GenFlow3D](https://github.com/ustc-hlli/GenFlow3D)
- [DynoSAM](https://github.com/ACFR-RPG/DynOSAM)

本仓库不复制第三方源代码、预训练权重或其 CUDA 扩展。使用上游项目时应遵循各自许可证与模型发布条款。本仓库在正式选择许可证前不对外宣称特定开源授权。
