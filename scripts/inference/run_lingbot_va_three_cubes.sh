#!/usr/bin/env bash
set -euo pipefail

# Async inference client for the three_cubes_1 LingBot-VA LoRA adapter on a real SO-101.
#
# Camera keys must match the adapter's training config
# (examples/lingbot_va_so101/common.py): front + right + wrist, concatenated on width.
#
# FPS matters twice: stalling AND world-model quality
# ---------------------------------------------------
# LingBot-VA predicts frame_chunk_size(4) x action_per_frame(4) = 16 actions per chunk.
# Measured on this Jetson Thor (bf16, 3 cameras, adapter attached, real preprocessed
# observations and real KV feedback), one chunk costs, by
# (num_inference_steps / action_num_inference_steps):
#
#     20/50 (training default)  20.89 s      ->  0.8 Hz
#     10/20                      8.82 s      ->  1.8 Hz
#      5/10 (checkpoint default) 4.98 s      ->  3.2 Hz
#      3/6                       3.22 s      ->  5.0 Hz
#      2/4                       2.50 s      ->  6.4 Hz
#
# 1. Stalling. Async inference only overlaps while a chunk covers more wall-clock time than
#    the next chunk takes to compute. At FPS=30 a 16-action chunk is 0.53 s of motion against
#    a ~5 s inference: the arm moves for half a second, then holds for ~4.5 s, repeatedly.
#
# 2. World-model quality (this is the part that is easy to miss). LingBot-VA is
#    autoregressive: after each chunk the *observed* frames are fed back into its KV cache,
#    and the next chunk is predicted from that history. The server hands back the frames
#    observed during the execution window (PolicyServer._push_observation_history). When FPS
#    is far above what inference can sustain, that window is 0.53 s of a 5 s cycle, so the
#    model's view of its own history is a sequence of brief twitches -- nothing like the
#    continuous 30 Hz motion it was trained on. Matching FPS to the chunk rate (16 actions /
#    chunk time, i.e. ~3 Hz at 5/10 steps) makes the fed-back history contiguous.
#
# The trajectory is executed slower than demonstrated; actions are absolute joint targets,
# so the path is preserved. Raise FPS only together with a lower step count (and re-run the
# build script with --video_steps / --action_steps to match).

REPO_DIR="${REPO_DIR:-/home/hrx/Projects/lerobot}"
PYTHON_BIN="${PYTHON_BIN:-/home/hrx/miniconda3/envs/lerobot/bin/python}"

SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8080}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM1}"
ROBOT_ID="${ROBOT_ID:-follower_arm}"

# This host exposes three capture devices: /dev/video{0,2,4} (the odd-numbered nodes are
# their metadata companions). front=2 / wrist=4 follow run_giga_world_three_cubes.sh;
# right=0 is the remaining one -- VERIFY the physical mapping before the first run, e.g.
#   python -m lerobot.scripts.lerobot_find_cameras opencv
# A swapped camera order silently degrades the policy: the three views are concatenated on
# width in obs_cam_keys order and the adapter was trained as front, right, wrist.
FRONT_CAMERA="${FRONT_CAMERA:-2}"
RIGHT_CAMERA="${RIGHT_CAMERA:-0}"
WRIST_CAMERA="${WRIST_CAMERA:-4}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"
FOURCC="${FOURCC:-MJPG}"
WRIST_FOURCC="${WRIST_FOURCC:-YUYV}"

POLICY_TYPE="${POLICY_TYPE:-lingbot_va}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
CLIENT_DEVICE="${CLIENT_DEVICE:-cpu}"

PRETRAINED_NAME_OR_PATH="${PRETRAINED_NAME_OR_PATH:-/home/hrx/Projects/models/three_cubes_1/lingbot_va_async}"

# The model's own chunk is 16 actions (the first chunk returns 12: frame 0 is the
# conditioning frame and its actions are dropped). PolicyServer truncates to
# min(ACTIONS_PER_CHUNK, 16), so anything above 16 has no effect.
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-16}"
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0.0}"
AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-conservative}"
FPS="${FPS:-30}"

TASK="${TASK:-go to red cube. take the red cube. go to box. put the red cube in box.}"
DEBUG_VISUALIZE_QUEUE_SIZE="${DEBUG_VISUALIZE_QUEUE_SIZE:-True}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"
RECORD_TIMELINE="${RECORD_TIMELINE:-true}"
TIMELINE_LOG_DIR="${TIMELINE_LOG_DIR:-/home/hrx/Projects/lerobot/logs/async_timeline/lingbot_va}"
TIMELINE_SAVE_IMAGES="${TIMELINE_SAVE_IMAGES:-key}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

if [[ ! -f "${PRETRAINED_NAME_OR_PATH}/config.json" ]]; then
  echo "Missing LingBot-VA checkpoint config: ${PRETRAINED_NAME_OR_PATH}/config.json" >&2
  echo "Build it with scripts/inference/build_lingbot_va_three_cubes_checkpoint.py" >&2
  exit 2
fi

if [[ ! -f "${PRETRAINED_NAME_OR_PATH}/adapter_model.safetensors" ]]; then
  echo "Missing adapter weights in ${PRETRAINED_NAME_OR_PATH}" >&2
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
  --robot.cameras="{ front: {type: opencv, index_or_path: ${FRONT_CAMERA}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, fourcc: \"${FOURCC}\"}, right: {type: opencv, index_or_path: ${RIGHT_CAMERA}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, fourcc: \"${FOURCC}\"}, wrist: {type: opencv, index_or_path: ${WRIST_CAMERA}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, fourcc: \"${WRIST_FOURCC}\"} }" \
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
