#!/usr/bin/env bash
set -euo pipefail

# First real run wrapper. Keep this intentionally short.
export EXECUTE="${EXECUTE:-True}"
export MAX_CONTROL_STEPS="${MAX_CONTROL_STEPS:-5}"
export DEBUG_DIR="${DEBUG_DIR:-/tmp/cosmos_lerobot_so101_debug}"
export ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM1}"
export ROBOT_ID="${ROBOT_ID:-follower_arm}"
export FRONT_CAMERA_INDEX="${FRONT_CAMERA_INDEX:-2}"
export WRIST_CAMERA_INDEX="${WRIST_CAMERA_INDEX:-4}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_cosmos_so101_client_thor.sh" \
  --execute="${EXECUTE}" \
  --max_control_steps="${MAX_CONTROL_STEPS}" \
  --debug_dir="${DEBUG_DIR}"
