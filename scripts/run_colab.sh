#!/usr/bin/env bash
set -euo pipefail
# Run from project root. Change DATA and INTRINSICS.
DATA="/content/drive/MyDrive/Datasets/Bonn/rgbd_bonn_person_tracking_stride3"
INTRINSICS="configs/bonn_intrinsics_example.yaml"
python -m pip install -e .
predictive4d run --data "$DATA" --intrinsics "$INTRINSICS" --config configs/reference.yaml --frames 20 --out outputs/bonn_smoke
