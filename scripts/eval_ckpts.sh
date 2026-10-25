#!/usr/bin/env bash
# Eval saved checkpoints of a run to get the success-vs-training-steps curve.
#   RUN=smolvla_libero_ft EPISODES=5 scripts/eval_ckpts.sh                       # every checkpoint
#   CKPTS="002500 005000 010000 020000" EPISODES=5 scripts/eval_ckpts.sh        # only these steps
# Fine-tuned from smolvla_base, the checkpoint keeps n_action_steps=50; set N_ACTION_STEPS to compare.
set -euo pipefail
source "$(dirname "$0")/common.sh"

RUN="${RUN:-smolvla_libero_ft}"
CKPTS="${CKPTS:-}"

for ckpt in "$OUTPUTS_DIR/$RUN"/checkpoints/[0-9]*/pretrained_model; do
  step="$(basename "$(dirname "$ckpt")")"
  if [[ -n "$CKPTS" && " $CKPTS " != *" $step "* ]]; then
    continue
  fi
  # Skip checkpoints already evaluated, so the script can be rerun while training is still saving new ones.
  if [[ -f "$RESULTS_DIR/${RUN}_step${step}/eval_info.json" ]]; then
    echo "skip $step (already evaluated)"
    continue
  fi
  "$(dirname "$0")/eval.sh" "$ckpt" "${RUN}_step${step}"
done
