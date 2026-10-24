#!/usr/bin/env bash
# The core exercise: same policy, vary how many actions of each chunk run before replanning.
#   scripts/sweep_n_action_steps.sh                         # official checkpoint
#   POLICY=outputs/.../pretrained_model scripts/sweep_n_action_steps.sh
# chunk_size is 50, so n must be <= 50. Inference must then run at >= 20 Hz / n.
set -euo pipefail
source "$(dirname "$0")/common.sh"

POLICY="${POLICY:-$OFFICIAL_POLICY}"
NAME="${NAME:-official}"
EPISODES="${EPISODES:-5}"
VALUES="${VALUES:-1 5 10 25 50}"

for n in $VALUES; do
  EPISODES="$EPISODES" N_ACTION_STEPS="$n" "$(dirname "$0")/eval.sh" "$POLICY" "sweep_${NAME}_n${n}"
done

python "$ROOT/scripts/collect_results.py" --plot-sweep "sweep_${NAME}_"
