# Native Integration

See the authoritative architecture and implementation contract in [FINAL_NOTION_ALIGNMENT.md](FINAL_NOTION_ALIGNMENT.md), and the VS Code commands in `../VSCode_最终框架_使用说明.md`.

The official GenFlow3D and native Flow4DGS backends are independent upstream repositories with their own compiled CUDA components. The current project contains guarded source integration hooks and an online runtime, but a full real-GPU run of both native components together remains the required final validation milestone.
