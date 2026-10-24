# Shared settings. Sourced by the other scripts; override any of these from the environment.

export MUJOCO_GL="${MUJOCO_GL:-egl}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-$ROOT/results}"
OUTPUTS_DIR="${OUTPUTS_DIR:-$ROOT/outputs}"

DATASET="${DATASET:-lerobot/libero}"
BASE_POLICY="${BASE_POLICY:-lerobot/smolvla_base}"
OFFICIAL_POLICY="${OFFICIAL_POLICY:-HuggingFaceVLA/smolvla_libero}"
SUITE="${SUITE:-libero_spatial}"

# smolvla_base was pretrained with cameras named camera1..3; LIBERO calls them image / image2.
RENAME_MAP='{"observation.images.image": "observation.images.camera1", "observation.images.image2": "observation.images.camera2"}'

# Prints the --rename_map flag when the checkpoint expects camera1/camera2 keys.
rename_flag() {
  local policy="$1"
  if [[ -f "$policy/config.json" ]] && grep -q 'observation.images.camera1' "$policy/config.json"; then
    printf -- '--rename_map=%s' "$RENAME_MAP"
  fi
}
