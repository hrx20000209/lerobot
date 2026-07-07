#!/usr/bin/env bash
set -euo pipefail

# PI0.5 front+wrist continuous async client in shadow mode.
# Defaults match the local Three Cubes PI0.5 front+wrist checkpoint and the
# requested hardware mapping: /dev/ttyACM1, camera index 4, camera index 2.

REPO_DIR="${REPO_DIR:-/home/hrx/Projects/lerobot}"

export SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8080}"
export ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"
export ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM1}"
export ROBOT_ID="${ROBOT_ID:-follower_arm}"
export POLICY_TYPE="${POLICY_TYPE:-pi05}"
export PRETRAINED_NAME_OR_PATH="${PRETRAINED_NAME_OR_PATH:-/home/hrx/Projects/models/three_cubes_1/pi05_front_wrist}"
export POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
export CLIENT_DEVICE="${CLIENT_DEVICE:-cpu}"
export ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-50}"
export FPS="${FPS:-30}"
export CONTINUOUS_OBS_FPS="${CONTINUOUS_OBS_FPS:-30}"
export TASK="${TASK:-go to red cube. take the red cube. go to box. put the red cube in box.}"

export CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
export CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
export FRONT_CAMERA_INDEX="${FRONT_CAMERA_INDEX:-4}"
export WRIST_CAMERA_INDEX="${WRIST_CAMERA_INDEX:-2}"
if [[ -z "${ROBOT_CAMERAS:-}" ]]; then
  ROBOT_CAMERAS="{ front: {type: opencv, index_or_path: ${FRONT_CAMERA_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${FPS}, fourcc: \"MJPG\"}, wrist: {type: opencv, index_or_path: ${WRIST_CAMERA_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${FPS}, fourcc: \"MJPG\"} }"
fi
export ROBOT_CAMERAS

export SHADOW_MODE="${SHADOW_MODE:-true}"
export ENABLE_ROBOT_EXECUTION="${ENABLE_ROBOT_EXECUTION:-false}"
export DISPLAY_DATA="${DISPLAY_DATA:-true}"
export RECORD_TIMELINE="${RECORD_TIMELINE:-true}"
export TIMELINE_SAVE_IMAGES="${TIMELINE_SAVE_IMAGES:-key}"
export TIMELINE_LOG_PATH="${TIMELINE_LOG_PATH:-${REPO_DIR}/outputs/continuous_async/pi05_client_timeline.jsonl}"
export TIMELINE_PLOT_PATH="${TIMELINE_PLOT_PATH:-${REPO_DIR}/outputs/continuous_async/pi05_timeline.png}"

export AGGREGATION_FN="${AGGREGATION_FN:-splice_by_timestamp}"
export MAX_PENDING_OBSERVATIONS="${MAX_PENDING_OBSERVATIONS:-1}"
export STALE_INFERENCE_MAX_AGE="${STALE_INFERENCE_MAX_AGE:-2.0}"
export BLEND_HORIZON="${BLEND_HORIZON:-5}"
export BLEND_ALPHA="${BLEND_ALPHA:-0.5}"
export EMERGENCY_STOP="${EMERGENCY_STOP:-false}"

exec "${REPO_DIR}/scripts/inference/run_continuous_async_robot_client.sh"
