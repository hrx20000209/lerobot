#!/usr/bin/env bash
set -euo pipefail

# Async inference client for the Three Cubes VLA-JEPA LoRA checkpoint.
#
# The checkpoint was trained with dataset camera keys:
#   observation.images.exterior_1_left
#   observation.images.exterior_2_left
#
# The async robot client currently has no CLI-level rename_map option, so the
# camera names below intentionally match the checkpoint keys directly.

REPO_DIR="${REPO_DIR:-/home/hrx/Projects/lerobot}"
PYTHON_BIN="${PYTHON_BIN:-/home/hrx/miniconda3/envs/lerobot/bin/python}"

SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8080}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM1}"
ROBOT_ID="${ROBOT_ID:-follower_arm}"

EXTERIOR_1_LEFT_CAMERA_INDEX="${EXTERIOR_1_LEFT_CAMERA_INDEX:-4}"
EXTERIOR_2_LEFT_CAMERA_INDEX="${EXTERIOR_2_LEFT_CAMERA_INDEX:-2}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
FPS="${FPS:-30}"
FOURCC="${FOURCC:-MJPG}"

POLICY_TYPE="${POLICY_TYPE:-vla_jepa}"
PRETRAINED_NAME_OR_PATH="${PRETRAINED_NAME_OR_PATH:-/home/hrx/Projects/models/three_cubes_1/vla_jepa_lora/}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
CLIENT_DEVICE="${CLIENT_DEVICE:-cpu}"

# VLA-JEPA checkpoint config has chunk_size=7. Keeping this default aligned with
# the checkpoint avoids implying that the model can produce a 50-step chunk.
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-7}"
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0.5}"
AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-conservative}"

TASK="${TASK:-go to red cube. take the red cube. go to box. put the red cube in box.}"
DEBUG_VISUALIZE_QUEUE_SIZE="${DEBUG_VISUALIZE_QUEUE_SIZE:-True}"
RECORD_TIMELINE="${RECORD_TIMELINE:-true}"
TIMELINE_LOG_DIR="${TIMELINE_LOG_DIR:-/home/hrx/Projects/lerobot/logs/async_timeline}"
TIMELINE_SAVE_IMAGES="${TIMELINE_SAVE_IMAGES:-key}"

cd "${REPO_DIR}"

exec "${PYTHON_BIN}" -m lerobot.async_inference.robot_client \
  --server_address="${SERVER_ADDRESS}" \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.cameras="{ exterior_1_left: {type: opencv, index_or_path: ${EXTERIOR_1_LEFT_CAMERA_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${FPS}, fourcc: \"${FOURCC}\"}, exterior_2_left: {type: opencv, index_or_path: ${EXTERIOR_2_LEFT_CAMERA_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${FPS}, fourcc: \"${FOURCC}\"} }" \
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
  --debug_visualize_queue_size="${DEBUG_VISUALIZE_QUEUE_SIZE}"
