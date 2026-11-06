# VS Code 直接替换旧版：按顺序操作

这是**重构版完整代码框架**，不需要将旧版 `constant_velocity` 代码保留在本地运行目录。

## 1. 修改旧 GitHub 仓库

1. 在 Windows 上将本 ZIP 解压到例如 `D:\\Projects\\Predictive4D-Joint`。不要把项目解压在旧仓库内部。
2. 进入旧 GitHub 仓库，在 VS Code **Terminal / PowerShell** 中执行：

```powershell
# 在旧 GitHub 仓库根目录执行。不要删除 .git 文件夹。
# 因为不保留旧版本，连同旧 tests/configs/scripts 一并替换，避免混用。
$base = 'D:\Projects\Predictive4D-Joint'  # 改成实际解压路径
foreach ($path in @('src\predictive4d', 'tests', 'configs', 'scripts', 'docs', '.vscode')) {
    if (Test-Path $path) { Remove-Item -Recurse -Force $path }
}
New-Item -ItemType Directory -Force '.\src' | Out-Null
Copy-Item -Recurse -Force "$base\src\predictive4d" '.\src\'
foreach ($path in @('tests', 'configs', 'scripts', 'docs', '.vscode')) {
    Copy-Item -Recurse -Force "$base\$path" '.\'
}
foreach ($name in @('pyproject.toml', 'requirements.txt', '.gitignore', 'README.md')) {
    Copy-Item -Force "$base\$name" '.\'
}
```

**替换前请确认** `D:\Projects\Predictive4D-Joint` 是你实际解压的位置。这些删除命令会删除旧版的 `src\predictive4d`、`tests`、`configs`、`scripts`、`docs`、`.vscode`。你已经明确表示不需要保留旧代码，但依然请先确认工作目录无误。`.git` 仓库历史不会被删除，但会记录文件替换。

如果你不想直接操作旧仓库，也可以直接用 VS Code 打开解压后的新项目，先验证运行再复制文件。

## 2. Windows VS Code Python 环境

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ".[test]"
python -m pytest -q
```

在 VS Code 按 Ctrl+Shift+P → **Python: Select Interpreter** → 选择 `.venv` 下的 Python。

## 3. 本地训练和端到端 Smoke Test

```powershell
python scripts/make_demo_rgbd.py --out data/demo_rgbd --frames 8
predictive4d train-synthetic --steps 200 --out checkpoints/joint_synthetic.pt
predictive4d run --data data/demo_rgbd --intrinsics data/demo_rgbd/intrinsics.yaml --dynamic-mask-dir data/demo_rgbd/dynamic_masks --checkpoint checkpoints/joint_synthetic.pt --config configs/joint.yaml --frames 8 --out outputs/demo_joint
```

用 Ctrl+Shift+D 可以选择 `Run joint SLAM (local demo)` 调试入口。

## 4. 将代码推送到旧 GitHub 地址

```powershell
git status
git add -A
git commit -m "Replace baseline with neural joint motion SLAM scaffold"
git push origin main
```

不要提交数据集和网络 checkpoint。先检查 `git status` 无意外的大文件。

## 5. Colab 上测试 TUM

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
!git clone https://github.com/ZexuZang/Predictive-Motion-aware-4D-Gaussian-SLAM.git /content/joint-slam
%cd /content/joint-slam
%pip install -e '.[test]'
!python -m pytest -q
!predictive4d train-synthetic --steps 200 --device cuda --out checkpoints/joint_synthetic.pt
!predictive4d run --data /content/drive/MyDrive/Datasets/DynamicSLAM/Processed/TUM/freiburg3_walking_halfsphere_stride3 --intrinsics configs/tum_freiburg3_intrinsics.yaml --checkpoint checkpoints/joint_synthetic.pt --config configs/joint.yaml --frames 20 --stride 1 --out /content/outputs/tum_joint20
```

如果 Colab 没有 CUDA，请在训练命令中使用 `--device cpu`。

**重要：** 合成数据训练的网络尚未证明能在室内 TUM/Bonn 正确预测。测试真实数据的结果只用于检查程序是否运行和接口是否匹配，不能当作方法精度结论。

## 6. 使用真正的官方 GenFlow3D Encoder + Decoder

首先在可编译 PointNet2 的 CUDA 环境中准备官方仓库、官方权重；检查 `configs/genflow_joint.yaml` 中的两个路径。

```bash
predictive4d train-official-head-synthetic \
  --genflow-repo /content/GenFlow3D \
  --genflow-checkpoint /content/GenFlow3D/pretrain/GenFlow3d_nuscenes.pth \
  --points 8192 --batch 1 --steps 50 --device cuda \
  --out checkpoints/official_joint_heads.pt

predictive4d run \
  --data /path/to/your/sequence \
  --intrinsics configs/tum_freiburg3_intrinsics.yaml \
  --checkpoint checkpoints/official_joint_heads.pt \
  --config configs/genflow_joint.yaml --frames 20 --device cuda \
  --out outputs/official_joint20
```

这个可选分支是真正复用 GenFlow3D 的 Encoder + Scene Flow Decoder；不过新 Camera Head 与动态置信度 Head 依然必须训练，且官方 GPU/PointNet2 运行环境并未在这份交付中验证。

## 7. 独立真实训练序列的 Camera Head 微调

```bash
predictive4d train-real-camera \
  --data /path/to/SEPARATE_TRAIN_sequence \
  --intrinsics /path/to/train_intrinsics.yaml \
  --checkpoint checkpoints/joint_synthetic.pt \
  --eval-data /path/to/TEST_sequence \
  --epochs 15 --out checkpoints/joint_real_camera.pt
```

这个命令只支持轻量共享网络；**训练序列不能与报告 ATE 的测试序列相同**。
