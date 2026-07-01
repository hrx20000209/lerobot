#!/usr/bin/env bash
set -euo pipefail

# Async inference client for a Three Cubes Giga World checkpoint.
#
# The matching training script used front+wrist camera keys:
#   observation.images.front
#   observation.images.wrist

REPO_DIR="${REPO_DIR:-/home/rxhuang/Projects/lerobot}"
PYTHON_BIN="${PYTHON_BIN:-/home/rxhuang/anaconda3/envs/lerobot/bin/python}"

SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8080}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM1}"
ROBOT_ID="${ROBOT_ID:-follower_arm}"

FRONT_CAMERA_INDEX="${FRONT_CAMERA_INDEX:-4}"
WRIST_CAMERA_INDEX="${WRIST_CAMERA_INDEX:-2}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
FPS="${FPS:-30}"
FOURCC="${FOURCC:-MJPG}"

POLICY_TYPE="${POLICY_TYPE:-giga_world}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
CLIENT_DEVICE="${CLIENT_DEVICE:-cpu}"

if [[ -z "${PRETRAINED_NAME_OR_PATH:-}" ]]; then
  CANDIDATE_CHECKPOINTS=(
    "/home/rxhuang/Projects/models/lerobot_train/three_cubes/giga_world_front_wrist_r64_w01_b8_padmask_ft/checkpoints/last/pretrained_model"
    "/home/rxhuang/Projects/models/lerobot_train/three_cubes/giga_world_front_wrist_r64_w01_b8/checkpoints/last/pretrained_model"
  )
  PRETRAINED_NAME_OR_PATH="${CANDIDATE_CHECKPOINTS[0]}"
  for candidate in "${CANDIDATE_CHECKPOINTS[@]}"; do
    if [[ -f "${candidate}/config.json" ]]; then
      PRETRAINED_NAME_OR_PATH="${candidate}"
      break
    fi
  done
fi

# At 30 Hz, all 48 actions cover 1.6 seconds. GigaWorld inference takes roughly
# 1-2 seconds on a 4090D, so returning only 16 actions would starve the client queue.
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-48}"
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-1.0}"
AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-conservative}"

TASK="${TASK:-go to red cube. take the red cube. go to box. put the red cube in box.}"
DEBUG_VISUALIZE_QUEUE_SIZE="${DEBUG_VISUALIZE_QUEUE_SIZE:-True}"
RECORD_TIMELINE="${RECORD_TIMELINE:-true}"
TIMELINE_LOG_DIR="${TIMELINE_LOG_DIR:-/home/rxhuang/Projects/lerobot/logs/async_timeline}"
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

cd "${REPO_DIR}"

exec "${PYTHON_BIN}" -m lerobot.async_inference.robot_client \
  --server_address="${SERVER_ADDRESS}" \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.cameras="{ front: {type: opencv, index_or_path: ${FRONT_CAMERA_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${FPS}, fourcc: \"${FOURCC}\"}, wrist: {type: opencv, index_or_path: ${WRIST_CAMERA_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${FPS}, fourcc: \"${FOURCC}\"} }" \
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
