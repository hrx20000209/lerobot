#!/usr/bin/env bash
set -euo pipefail

# PI0.5 front+wrist continuous async client with real motor execution enabled.
# Set MAX_RELATIVE_TARGET explicitly when you want SOFollower.send_action() to
# clip each motor goal relative to the present position.

REPO_DIR="${REPO_DIR:-/home/hrx/Projects/lerobot}"

export SHADOW_MODE="${SHADOW_MODE:-false}"
export ENABLE_ROBOT_EXECUTION="${ENABLE_ROBOT_EXECUTION:-true}"
export DISPLAY_DATA="${DISPLAY_DATA:-false}"
export TIMELINE_SAVE_IMAGES="${TIMELINE_SAVE_IMAGES:-none}"
export TIMELINE_LOG_PATH="${TIMELINE_LOG_PATH:-${REPO_DIR}/outputs/continuous_async/pi05_real_client_timeline.jsonl}"
export TIMELINE_PLOT_PATH="${TIMELINE_PLOT_PATH:-${REPO_DIR}/outputs/continuous_async/pi05_real_timeline.png}"

exec "${REPO_DIR}/scripts/inference/run_pi05_continuous_async_shadow_client.sh"
