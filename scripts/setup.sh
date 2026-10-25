#!/usr/bin/env bash
# One-time setup on a Linux + NVIDIA GPU machine. LIBERO only installs on Linux.
# In mainland China: export HF_ENDPOINT=https://hf-mirror.com before running.
set -euo pipefail
source "$(dirname "$0")/common.sh"

# egl_probe compiles C++ at install time (needs cmake); torchcodec needs FFmpeg's shared libs;
# headless MuJoCo needs EGL (GPU) or OSMesa (CPU fallback).
if command -v apt-get >/dev/null && ! command -v cmake >/dev/null; then
  DEBIAN_FRONTEND=noninteractive apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y cmake build-essential libegl1 libgl1 libosmesa6 libglvnd-dev ffmpeg
fi

pip install "lerobot[smolvla,libero]==0.6.1" matplotlib
python -c "import torch; assert torch.cuda.is_available(), 'no CUDA GPU'; print(torch.cuda.get_device_name(0))"

# LIBERO asks an interactive Y/N question on first import, which hangs headless runs. Answer "no custom path".
echo n | python -c "import libero.libero" > /dev/null

# Pull the dataset (1.9 GB), both checkpoints and LIBERO's scene assets now so training and eval start immediately.
# LeRobotDataset reads from its own cache (HF_LEROBOT_HOME/hub), not the default HF hub cache.
hf download "$DATASET" --repo-type dataset --cache-dir "${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}/hub"
hf download "$BASE_POLICY"
# smolvla_base loads its VLM backbone by name at policy creation, with no progress output.
hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct
hf download "$OFFICIAL_POLICY"
hf download lerobot/libero-assets --repo-type dataset --local-dir ~/.cache/libero/assets
