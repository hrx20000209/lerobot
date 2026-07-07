#!/usr/bin/env bash
set -euo pipefail

# Async inference client for a Three Cubes Giga World checkpoint.
#
# The matching training script used front+wrist camera keys:
#   observation.images.front
#   observation.images.wrist

REPO_DIR="${REPO_DIR:-/home/hrx/Projects/lerobot}"
PYTHON_BIN="${PYTHON_BIN:-/home/hrx/miniconda3/envs/lerobot/bin/python}"

SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8080}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM1}"
ROBOT_ID="${ROBOT_ID:-follower_arm}"

# Stable USB-port paths; /dev/videoN changes whenever a camera reconnects.
# FRONT_CAMERA="${FRONT_CAMERA:-/dev/v4l/by-path/platform-a80aa10000.usb-usb-0:4.2.2:1.0-video-index0}"
# WRIST_CAMERA="${WRIST_CAMERA:-/dev/v4l/by-path/platform-a80aa10000.usb-usb-0:4.2.4:1.0-video-index0}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
FPS="${FPS:-30}"
FOURCC="${FOURCC:-MJPG}"

POLICY_TYPE="${POLICY_TYPE:-giga_world}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
CLIENT_DEVICE="${CLIENT_DEVICE:-cpu}"

PRETRAINED_NAME_OR_PATH="${PRETRAINED_NAME_OR_PATH:-/home/hrx/Projects/models/three_cubes_1/giga_world}"

# At 30 Hz, all 48 actions cover 1.6 seconds. Warm GigaWorld inference takes
# roughly 1.3-2.0 seconds on this Jetson Thor, so use the complete model chunk.
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-48}"
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0.0}"
AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-weighted_average}"

TASK="${TASK:-go to red cube. take the red cube. go to box. put the red cube in box.}"
DEBUG_VISUALIZE_QUEUE_SIZE="${DEBUG_VISUALIZE_QUEUE_SIZE:-True}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"
RECORD_TIMELINE="${RECORD_TIMELINE:-true}"
TIMELINE_LOG_DIR="${TIMELINE_LOG_DIR:-/home/hrx/Projects/lerobot/logs/async_timeline/giga_world}"
TIMELINE_SAVE_IMAGES="${TIMELINE_SAVE_IMAGES:-key}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

if [[ ! -f "${PRETRAINED_NAME_OR_PATH}/config.json" ]]; then
  echo "Missing Giga World checkpoint config: ${PRETRAINED_NAME_OR_PATH}/config.json" >&2
  echo "Set PRETRAINED_NAME_OR_PATH to the checkpoint's pretrained_model directory." >&2
  exit 2
fi

mkdir -p "${TIMELINE_LOG_DIR}"

cd "${REPO_DIR}"

ROBOT_SAFETY_ARGS=()
if [[ -n "${MAX_RELATIVE_TARGET:-}" ]]; then
  ROBOT_SAFETY_ARGS+=(--robot.max_relative_target="${MAX_RELATIVE_TARGET}")
fi

exec "${PYTHON_BIN}" -m lerobot.async_inference.robot_client \
  --server_address="${SERVER_ADDRESS}" \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  "${ROBOT_SAFETY_ARGS[@]}" \
  --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${FPS}, fourcc: \"${FOURCC}\"}, wrist: {type: opencv, index_or_path: 4, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${FPS}, fourcc: \"YUYV\"} }" \
  --task="${TASK}" \
  --policy_type="${POLICY_TYPE}" \
  --pretrained_name_or_path="${PRETRAINED_NAME_OR_PATH}" \
  --policy_device="${POLICY_DEVICE}" \
  --client_device="${CLIENT_DEVICE}" \
  --actions_per_chunk="${ACTIONS_PER_CHUNK}" \
  --chunk_size_threshold="${CHUNK_SIZE_THRESHOLD}" \
  --aggregate_fn_name="${AGGREGATE_FN_NAME}" \
  --fps="${FPS}" \
  --record_timeline="${RECORD_TIMELINE}" \
  --timeline_log_dir="${TIMELINE_LOG_DIR}" \
  --timeline_save_images="${TIMELINE_SAVE_IMAGES}" \
  --display_data="${DISPLAY_DATA}" \
  --debug_visualize_queue_size="${DEBUG_VISUALIZE_QUEUE_SIZE}"
