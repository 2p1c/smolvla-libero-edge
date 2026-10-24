#!/usr/bin/env bash
# Fine-tune smolvla_base on LIBERO. Extra args go straight to lerobot-train.
#   STEPS=20000 BATCH=32 scripts/train.sh
# Check it/s in the first minute, then set STEPS = it/s x seconds you can afford.
set -euo pipefail
source "$(dirname "$0")/common.sh"

STEPS="${STEPS:-20000}"
BATCH="${BATCH:-32}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
RUN="${RUN:-smolvla_libero_ft}"

# lerobot-train refuses an existing output_dir, so only the parent is created here.
mkdir -p "$OUTPUTS_DIR"

lerobot-train \
  --policy.path="$BASE_POLICY" \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.scheduler_decay_steps="$STEPS" \
  --dataset.repo_id="$DATASET" \
  --dataset.video_backend=torchcodec \
  --rename_map="$RENAME_MAP" \
  --batch_size="$BATCH" \
  --steps="$STEPS" \
  --save_freq="$SAVE_FREQ" \
  --wandb.enable=false \
  --job_name="$RUN" \
  --output_dir="$OUTPUTS_DIR/$RUN" \
  "$@" 2>&1 | tee "$OUTPUTS_DIR/$RUN.log"
