#!/usr/bin/env bash
set -euo pipefail

# Front+wrist async inference client for Three Cubes.
#
# Override any value from the shell, for example:
#   POLICY_TYPE=vla_jepa PRETRAINED_NAME_OR_PATH=/path/to/checkpoint/pretrained_model \
#     scripts/inference/run_front_wrist.sh

SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8080}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM1}"
ROBOT_ID="${ROBOT_ID:-follower_arm}"
FRONT_CAMERA_INDEX="${FRONT_CAMERA_INDEX:-4}"
WRIST_CAMERA_INDEX="${WRIST_CAMERA_INDEX:-2}"
POLICY_TYPE="${POLICY_TYPE:-pi05}"
PRETRAINED_NAME_OR_PATH="${PRETRAINED_NAME_OR_PATH:-/home/rxhuang/Projects/lerobot/output_lerobot_train/three_cubes/pi05_front_wrist_r64_full_expert_pad01_b24_after_smolvla/checkpoints/last/pretrained_model}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
CLIENT_DEVICE="${CLIENT_DEVICE:-cpu}"
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-50}"
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-1.0}"
AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-conservative}"
FPS="${FPS:-30}"
TASK="${TASK:-go to red cube. take the red cube. go to box. put the red cube in box.}"
RECORD_TIMELINE="${RECORD_TIMELINE:-true}"
TIMELINE_LOG_DIR="${TIMELINE_LOG_DIR:-/home/rxhuang/Projects/lerobot/logs/async_timeline}"
TIMELINE_SAVE_IMAGES="${TIMELINE_SAVE_IMAGES:-key}"
DEBUG_VISUALIZE_QUEUE_SIZE="${DEBUG_VISUALIZE_QUEUE_SIZE:-True}"
PYTHON_BIN="${PYTHON_BIN:-/home/rxhuang/anaconda3/envs/lerobot/bin/python}"

cd /home/rxhuang/Projects/lerobot

"${PYTHON_BIN}" -m lerobot.async_inference.robot_client \
  --server_address="${SERVER_ADDRESS}" \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.cameras="{ front: {type: opencv, index_or_path: ${FRONT_CAMERA_INDEX}, width: 640, height: 480, fps: ${FPS}, fourcc: \"MJPG\"}, wrist: {type: opencv, index_or_path: ${WRIST_CAMERA_INDEX}, width: 640, height: 480, fps: ${FPS}, fourcc: \"MJPG\"}}" \
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
