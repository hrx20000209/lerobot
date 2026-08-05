#!/usr/bin/env bash
set -euo pipefail

# HIL shadow client for the Cosmos 13K checkpoint on Jetson Thor.
#
# Reads the SO101 follower on /dev/ttyACM0 (joint positions + 3 cameras) and
# streams observations to the policy server, but NEVER writes a goal position:
# cosmos_shadow_client_no_write.py blocks send_action() and any Goal_* register
# write at the motor-bus level.
#
# Camera mapping matches the training dataset features:
#   observation.images.front -> index 4
#   observation.images.right -> index 2   (server's "left_wrist" slot)
#   observation.images.wrist -> index 0   (server's "right_wrist" slot)
# /dev/video0 is YUYV-only, hence the different fourcc.

REPO_DIR="${REPO_DIR:-/home/hrx/Projects/lerobot}"
PYTHON_BIN="${PYTHON_BIN:-/home/hrx/miniconda3/envs/lerobot/bin/python}"

SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8082}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
ROBOT_ID="${ROBOT_ID:-follower_arm}"

FRONT_CAMERA_INDEX="${FRONT_CAMERA_INDEX:-4}"
RIGHT_CAMERA_INDEX="${RIGHT_CAMERA_INDEX:-2}"
WRIST_CAMERA_INDEX="${WRIST_CAMERA_INDEX:-0}"
FRONT_FOURCC="${FRONT_FOURCC:-MJPG}"
RIGHT_FOURCC="${RIGHT_FOURCC:-MJPG}"
WRIST_FOURCC="${WRIST_FOURCC:-YUYV}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"

FPS="${FPS:-30}"
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-16}"
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0.7}"
AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-conservative}"
RUN_SECONDS="${RUN_SECONDS:-60}"
TASK="${TASK:-go to red cube. take the red cube. go to box. put the red cube in box.}"

LOG_DIR="${LOG_DIR:-/home/hrx/Projects/models/three_cubes_1/cosmos_policy_step13000/shadow_logs}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
TIMELINE_LOG_DIR="${TIMELINE_LOG_DIR:-${LOG_DIR}/timeline_13k_${RUN_TAG}}"
SHADOW_JSONL_PATH="${SHADOW_JSONL_PATH:-${LOG_DIR}/shadow_actions_13k_${RUN_TAG}.jsonl}"
CLIENT_LOG="${CLIENT_LOG:-${LOG_DIR}/client_13k_${RUN_TAG}.log}"

# `max_relative_target` is a belt-and-braces clamp only. The shadow client never
# reaches send_action(), so this value is never actually exercised.
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-0.05}"

mkdir -p "${LOG_DIR}" "${TIMELINE_LOG_DIR}"

CAMERAS="{ front: {type: opencv, index_or_path: ${FRONT_CAMERA_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, fourcc: \"${FRONT_FOURCC}\"}, \
right: {type: opencv, index_or_path: ${RIGHT_CAMERA_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, fourcc: \"${RIGHT_FOURCC}\"}, \
wrist: {type: opencv, index_or_path: ${WRIST_CAMERA_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, fourcc: \"${WRIST_FOURCC}\"} }"

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/home/hrx/.cache/huggingface}"

echo "Cosmos 13K SHADOW client -> ${CLIENT_LOG}"
echo "  robot=${ROBOT_PORT} (READ-ONLY: no Goal_Position will be written)"
echo "  cameras front=${FRONT_CAMERA_INDEX} right=${RIGHT_CAMERA_INDEX} wrist=${WRIST_CAMERA_INDEX}"
echo "  shadow records -> ${SHADOW_JSONL_PATH}"

cd "${REPO_DIR}"

exec "${PYTHON_BIN}" -u scripts/inference/cosmos_shadow_client_no_write.py \
  --server_address="${SERVER_ADDRESS}" \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.use_degrees=true \
  --robot.max_relative_target="${MAX_RELATIVE_TARGET}" \
  --robot.cameras="${CAMERAS}" \
  --task="${TASK}" \
  --policy_type=cosmos_policy \
  --pretrained_name_or_path=/home/hrx/Projects/models/three_cubes_1/cosmos_policy_step13000/model/model \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk="${ACTIONS_PER_CHUNK}" \
  --chunk_size_threshold="${CHUNK_SIZE_THRESHOLD}" \
  --aggregate_fn_name="${AGGREGATE_FN_NAME}" \
  --fps="${FPS}" \
  --run_seconds="${RUN_SECONDS}" \
  --record_timeline=true \
  --timeline_log_dir="${TIMELINE_LOG_DIR}" \
  --timeline_save_images=key \
  --shadow_jsonl_path="${SHADOW_JSONL_PATH}" \
  --debug_visualize_queue_size=false \
  "$@" 2>&1 | tee "${CLIENT_LOG}"
