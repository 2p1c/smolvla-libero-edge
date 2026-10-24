#!/usr/bin/env bash
# Closed-loop eval of one policy in LIBERO (MuJoCo). Writes results/<tag>/eval_info.json + meta.json.
#   scripts/eval.sh <policy path or hub id> <tag> [extra lerobot-eval args]
#   EPISODES=10 N_ACTION_STEPS=10 NUM_STEPS=5 scripts/eval.sh outputs/.../pretrained_model ft_n10
# EPISODES is per task; libero_spatial has 10 tasks, so EPISODES=10 means 100 episodes.
set -euo pipefail
source "$(dirname "$0")/common.sh"

POLICY="$1"
TAG="$2"
shift 2
EPISODES="${EPISODES:-10}"
OUT="$RESULTS_DIR/$TAG"

ARGS=(
  --policy.path="$POLICY"
  --env.type=libero
  --env.task="$SUITE"
  --eval.batch_size=1
  --eval.n_episodes="$EPISODES"
  --env.max_parallel_tasks=1
  --seed=1000
  --output_dir="$OUT"
)
[[ -n "${N_ACTION_STEPS:-}" ]] && ARGS+=(--policy.n_action_steps="$N_ACTION_STEPS")
[[ -n "${NUM_STEPS:-}" ]] && ARGS+=(--policy.num_steps="$NUM_STEPS")
RENAME="$(rename_flag "$POLICY")"
[[ -n "$RENAME" ]] && ARGS+=("$RENAME")

rm -rf "$OUT"
mkdir -p "$OUT"
lerobot-eval "${ARGS[@]}" "$@" 2>&1 | tee "$OUT/eval.log"

python - "$OUT/meta.json" "$POLICY" "$SUITE" "$EPISODES" "${N_ACTION_STEPS:-}" "${NUM_STEPS:-}" <<'EOF'
import json, sys
path, policy, suite, episodes, n_action_steps, num_steps = sys.argv[1:]
json.dump({
    "policy": policy,
    "suite": suite,
    "episodes_per_task": int(episodes),
    "n_action_steps": int(n_action_steps) if n_action_steps else None,
    "num_steps": int(num_steps) if num_steps else None,
}, open(path, "w"), indent=2)
EOF
