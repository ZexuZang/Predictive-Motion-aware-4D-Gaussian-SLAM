# VS Code 最终研究架构：操作手册

## 先分清两个运行路径

- **快速验证路径**：`configs/joint.yaml` + `JointMotionNet`，只用于测试三假设概率分解、Schur 位姿更新、Gaussian Adapter、动态反馈是否执行。它不是最终 GenFlow3D 模型。
- **论文目标路径**：`configs/genflow_joint.yaml` + 官方 GenFlow3D CUDA Encoder/BDCB/STIB/Diffusion + 新训练的 Camera Head + 官方 Flow4DGS-SLAM。原始 CUDA 后端通过 `flow4dgs_online.py` 和 `flow4dgs_native.py` 接入。

## 1. VS Code 中替换旧仓库

1. 解压 `Predictive4D-Final-Framework-VSCode.zip`。
2. VS Code 中选择 **File → Open Folder**，打开解压目录。
3. 若继续使用原 GitHub 仓库：在旧仓库 VS Code 终端中，把新包的 `src/`、`configs/`、`scripts/`、`tests/`、`docs/`、`README.md`、`pyproject.toml` 复制到相应路径，按需要移除旧模块。**不要把 ZIP 中的 `.git` 覆盖到任何 Git 仓库**。
4. 因你明确表示不需要保留旧版，可将更新后的工作树提交到当前 Git 分支。但建议仍用 `git status` 查看待提交内容，避免意外覆盖本地数据集、checkpoint。

## 2. 本地 Python 测试

Windows PowerShell：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
python -m pytest -q
```

当前发布包中共有 13 项自动测试，覆盖共享特征的反向传播、SE(3)、三假设后验、动态 Schur、Gaussian 退化、动态反馈、Torch 渲染，以及原生接口的模拟测试。

## 3. 神经网络参考路径运行

```powershell
predictive4d train-synthetic --steps 200 --points 256 --history 3 --out checkpoints/compact_synthetic.pt
python scripts/make_demo_rgbd.py --out data/demo_rgbd --frames 8
predictive4d run --data data/demo_rgbd --intrinsics data/demo_rgbd/intrinsics.yaml --checkpoint checkpoints/compact_synthetic.pt --config configs/joint.yaml --frames 8 --out outputs/demo
```

此路径只证明数据流和数值逻辑可运行。合成 checkpoint 不能用来报告 TUM/Bonn 上的论文性能。

## 4. Linux GPU 上进行最终架构集成

在 VS Code Remote SSH / AutoDL / CUDA Linux：

```bash
git clone https://github.com/ustc-hlli/GenFlow3D.git
git clone https://github.com/wangys16/Flow4DGS-SLAM.git
```

按照两个官方仓库 README 安装所需的 CUDA/PointNet2、Gaussian Rasterizer 和依赖，**不要直接覆盖原来可工作的 PyTorch/CUDA 环境**。

在 `configs/genflow_joint.yaml` 设置官方 GenFlow3D 路径与 checkpoint，完成新增 Camera Head 的独立训练。提供的 `train-official-head-synthetic` 只用于验证模型结构能训练，**不能替代在真实且独立的 RGB-D 数据上训练相机 Head**。

之后在自己的项目根目录执行：

```bash
python scripts/patch_flow4dgs.py --upstream ../Flow4DGS-SLAM
python scripts/patch_flow4dgs.py --upstream ../Flow4DGS-SLAM --apply
```

第一条命令只打印 diff，第二条才修改 Flow4DGS 的 `utils/slam_frontend.py` 与 `utils/slam_backend.py`。它只会在当前源代码匹配经过核查的插入点时生效，否则明确报错，禁止盲目替换。

把 `configs/flow4dgs_native_extra.yaml` 的 `predictive4d` 部分合并到官方 Flow4DGS 的 **RGB-D 运行配置**，并设置：

```yaml
model_params:
  flow_cam: false
```

注意这只是配置片段；请保留官方原 YAML 的其他 `Dataset` / `Training` / `Results` 参数。修改 `seconds_per_frame` 为所使用的**处理后序列实际时间间隔（秒）**；stride3 的 TUM 大约 0.1s 仅是示例，最好使用真实逐帧 timestamp。

接着使用官方 Flow4DGS 仓库自带的入口与原有任务配置运行。前端在当前帧 Tracking 之前，借助已完成的历史帧生成神经预测。当前帧的 RGB-D 之后用于联合 SE(3) 校正和原生 Gaussian 渲染。后端通过原生 dynamic Gaussian ID + keyframe 对齐来转换预测 Flow。

## 5. 需要你实测和继续完善的最后部分

- 真正的 GenFlow3D 权重 + PointNet2 CUDA 推理是否与当前 RGB-D 世界坐标输入匹配（时间间隔、采样点数和域偏移）。
- 新相机 Head 的真实监督训练、未见过序列的准确度、预测协方差校准。
- 官方 Flow4DGS 多进程/关键帧同步与不同 commit 的兼容性；当前没有在此环境中完成完整 CUDA 端到端测试。
- 完整原生高斯 Rotation/Scale/GMM 参数的联合预测先验，以及联合多帧窗口因子图（目前有局部 Schur SE(3)，并非全窗口 BA）。
- 上游 RAFT 的其他调用路径；替换 offset 不等于彻底删掉所有 RAFT。

**研究实现状态请以 `docs/FINAL_NOTION_ALIGNMENT.md` 为准。没有 CUDA 实测与真实数据量化之前，不要称为论文最终可复现结果。**
