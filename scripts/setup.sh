#!/usr/bin/env bash
# One-time setup on a Linux + NVIDIA GPU machine. LIBERO only installs on Linux.
set -euo pipefail
source "$(dirname "$0")/common.sh"

pip install "lerobot[smolvla,libero]==0.6.1" matplotlib
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU'; print(torch.cuda.get_device_name(0))"

# Pull the dataset (1.9 GB) and both checkpoints now so training and eval start immediately.
hf download "$DATASET" --repo-type dataset
hf download "$BASE_POLICY"
hf download "$OFFICIAL_POLICY"
