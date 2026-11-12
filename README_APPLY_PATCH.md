# Bring the successful Colab changes into the GitHub repository

The ZIP root mirrors the repository root. Apply it over a **clean clone** of
`ZexuZang/Predictive-Motion-aware-4D-Gaussian-SLAM` after taking a new branch.
No trained weights or 194-frame result files are included.

```bash
cd /content
git clone https://github.com/ZexuZang/Predictive-Motion-aware-4D-Gaussian-SLAM.git
cd Predictive-Motion-aware-4D-Gaussian-SLAM
git switch -c fix/native-cvgru-colab
# Unzip predictive4d_github_fixes.zip into this repository's root
# (overwrites scripts/patch_flow4dgs.py with the tested native integration patcher)
git status --short
python -m pytest -q tests/test_cv_acceleration.py tests/test_native_pose_compat.py tests/test_native_patch_repro.py
git add scripts src tests docs README_APPLY_PATCH.md
git commit -m 'Integrate native CV+GRU and consolidated Flow4DGS/GenFlow fixes'
git push -u origin fix/native-cvgru-colab
```

Authenticate via your own GitHub session/token/credential manager if push asks.
A fresh Colab runtime can use the next cloned branch and run the commands in
`docs/NATIVE_CVGRU_COLAB_REPRODUCIBILITY.md` without notebook-only Python files.
