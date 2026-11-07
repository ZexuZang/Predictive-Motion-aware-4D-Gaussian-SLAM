# Learned Reliability Fusion Head — 第一阶段可复现实验

## 目的与适用范围

本补丁只增加 **Camera SE(3) 标量权重的监督式自适应融合**：

`Neural camera proposal + geometric camera proposal -> feature normalization -> MLP -> w_geo -> SE(3) interpolation`。

- 使用已经训练好的 `JointMotionNet` 或 `OfficialSharedGenFlow` 产生神经预测；**不**重新训练 GenFlow3D。
- `joint` 来源是 `optimize_joint_pose()` 的 **神经初值优化后** 位姿，不是独立 Geometry Expert。
- `independent` 来源是 `static_pose_correction()` 从上一个可信相机位姿开始、只依据当前/上一帧几何和可选外部 Mask 计算的候选；不使用神经预测位姿和 Flow 作为初值。与原生 Flow4DGS 的几何求解器仍不是同一算法。
- 第一版只融合 **Camera Pose**。**不训练 Scene Flow 权重**，**不接入原生 Flow4DGS**，也**不实现双失败 Reject Head**。
- 在 `uncertainty_calibrated: false` 下保持原有 Schur 动态反馈保护开关，**不要因为加载 Fusion Head 而手动改为 true**。

## 更新的文件

- **新增** `src/predictive4d/fusion_head.py`：18 个可靠性特征、64→32→1 的 MLP、可微 SE(3) 左插值、冻结推理加载。
- **新增** `src/predictive4d/train_fusion.py`：基于真实 Camera GT 的训练、独立序列验证、测试序列基线比较。
- **修改** `src/predictive4d/pipeline.py`：在预测与几何优化之后导出两类候选与可靠性特征；可选应用 Frozen Fusion Head；再进入反馈和地图更新。
- **修改** `src/predictive4d/cli.py`：增加四个 CLI 选项/入口。
- **新增** `tests/test_fusion.py`：Lie 插值、梯度、特征契约、GT 第一帧坐标对齐。

## 前提

在项目根目录安装：

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
```

确保 **Camera Head checkpoint 已在真实训练序列上训练有效**。合成 checkpoint 只可用于检查流程，不能用于科研结论。

## Step 1：按 *序列* 生成候选文件（无需 GT）

对每一条 Bonn 训练/验证/测试序列分别运行，例如：

```powershell
predictive4d run `
  --data "D:\Datasets\Bonn\seq_train_A" `
  --intrinsics "D:\Datasets\Bonn\intrinsics.yaml" `
  --checkpoint "checkpoints\camera_head.pt" `
  --config "configs\genflow_joint.yaml" `
  --frames 0 `
  --out "outputs\fusion_train_A" `
  --export-fusion-candidates
```

> `--frames 0` 表示不截断；`--checkpoint` 与 YAML 必须匹配，同样需要 Linux/CUDA 才能运行官方 GenFlow3D。Windows 的轻量模型使用 `configs/joint.yaml` 和对应轻量 checkpoint。

输出 `outputs/fusion_train_A/fusion_candidates.npz`，包含：

- `timestamps`, `origin_timestamp`
- `T_pred`, `T_geo`, `features`, `valid_geo`
- `T_geo_independent`, `features_independent`, `valid_geo_independent`

数据包 **不包含 GT**。只有网络历史长度满足时才能输出候选。

## Step 2：用不同序列的相机轨迹 GT 训练权重

```powershell
predictive4d train-fusion `
  --train-pair "outputs\fusion_train_A\fusion_candidates.npz" "D:\Datasets\Bonn\seq_train_A\groundtruth.txt" `
  --train-pair "outputs\fusion_train_B\fusion_candidates.npz" "D:\Datasets\Bonn\seq_train_B\groundtruth.txt" `
  --val-pair "outputs\fusion_val_C\fusion_candidates.npz" "D:\Datasets\Bonn\seq_val_C\groundtruth.txt" `
  --geometry-source independent `
  --out "checkpoints\fusion_independent.pt" `
  --epochs 150
```

`--geometry-source independent` 更适合研究神经与独立几何提议的融合；`joint` 更接近原先的 `joint_ba.py` 优化结果，但存在共用先验相关性。两种来源必须分别训练，不能混用同一个 Head。

训练时通过 GT `T_wc,t` 对齐第一帧相机世界坐标：`T_cw,t^GT = inv(T_wc,t^GT) @ T_wc,0^GT`。GT **不输入网络推理**。

保存 checkpoint 包括训练集特征均值/标准差和在验证集选出的 **最佳固定权重**。学习权重使用端到端 SE(3) 融合位置+旋转损失；无需人为给出权重标签。

## Step 3：只在真正留出的测试序列评估

```powershell
predictive4d eval-fusion `
  --checkpoint "checkpoints\fusion_independent.pt" `
  --pair "outputs\fusion_test_D\fusion_candidates.npz" "D:\Datasets\Bonn\seq_test_D\groundtruth.txt"
```

输出四项对比：`neural_only`, `geometry_only`, `fixed_half`, `validation_tuned_fixed`, `learned`。注意这里的 `translation_rmse_m` 为**第一帧锚定、未进行 Horn 全局拟合**的相机中心误差，不能直接等同 `evaluate.py` 中的 Horn-aligned ATE。

**预先按完整 sequence** 划分 train/validation/test；不能把一条视频相邻帧随机拆开。测试文件不允许与 checkpoint 中记录的训练/验证文件完全同路径，但研究者还应避免内容重复和对测试集反复调参。

## Step 4：可选闭环推理

```powershell
predictive4d run `
  --data "D:\Datasets\Bonn\seq_test_D" `
  --intrinsics "D:\Datasets\Bonn\intrinsics.yaml" `
  --checkpoint "checkpoints\camera_head.pt" `
  --config "configs\genflow_joint.yaml" `
  --out "outputs\fusion_closed_loop_test_D" `
  --fusion-checkpoint "checkpoints\fusion_independent.pt"
```

`frame_records.json` 新增 `fusion.status`, `fusion.weight_geometry`, `fusion.weight_prediction` 和 `fusion.source`。没有 `--fusion-checkpoint` 时，原算法完全保持默认。

现阶段，只在几何候选 `tracking_ok` 时运行融合；如果几何分支失败，保留原先的谨慎跟踪丢失逻辑。因此这个 **MVP 不声称已经解决极端静态约束缺失场景**。应在未来增加网络独立 Pose Acceptance / Reject Head，并以 Failure Rate 和地图污染率进行严格验证。

## 后续阶段（尚未在本补丁实现）

1. 训练 6DoF / Hessian 本征空间权重，评估可观测性特征是否提供增益。
2. 增加双失败 Reject Gate / Calibration；几何完全退化时的预测保持、恢复与重定位。
3. 引入 Point-level Scene Flow Adaptive Fusion，需要真实 Flow GT 或严格验证的自监督。
4. 在 **原生 Flow4DGS-SLAM** frontend/backend 中建立经过真实 CUDA 测试的集成；此补丁只接入独立参考 `pipeline.py`。
