#!/usr/bin/env bash
# Eval every saved checkpoint of a run to get the success-vs-training-steps curve.
#   RUN=smolvla_libero_ft EPISODES=10 scripts/eval_ckpts.sh
# Fine-tuned from smolvla_base, the checkpoint keeps n_action_steps=50; set N_ACTION_STEPS to compare.
set -euo pipefail
source "$(dirname "$0")/common.sh"

RUN="${RUN:-smolvla_libero_ft}"

for ckpt in "$OUTPUTS_DIR/$RUN"/checkpoints/[0-9]*/pretrained_model; do
  step="$(basename "$(dirname "$ckpt")")"
  "$(dirname "$0")/eval.sh" "$ckpt" "${RUN}_step${step}"
done
