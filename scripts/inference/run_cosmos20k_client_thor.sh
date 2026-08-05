#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# REAL CLOSED-LOOP RUN -- THIS WRITES GOAL POSITIONS TO /dev/ttyACM0.
# The arm moves. Stay next to it, ready to physically e-stop or cut power.
#
# Start run_cosmos20k_server_thor.sh first and wait for the port to open.
# Set SHADOW=true to make the motor bus read-only for a dry check.
# ============================================================================
#
# Records, for every executed action: the commanded joints, the joint positions
# measured at that instant, the resulting tracking error, queue depth, how stale
# the observation was, the server-side stage breakdown, and periodic servo
# load/temperature. Screenshots are written from a background thread so JPEG
# encoding stays out of the control loop.
#
# fps must match the server's: the server stamps action timestamps with 1/fps.

REPO_DIR="${REPO_DIR:-/home/hrx/Projects/lerobot}"
PYTHON_BIN="${PYTHON_BIN:-/home/hrx/miniconda3/envs/lerobot/bin/python}"

SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8082}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
ROBOT_ID="${ROBOT_ID:-follower_arm}"
CKPT_ROOT="${CKPT_ROOT:-/home/hrx/Projects/models/three_cubes_1/cosmos_policy_step20000}"

# Camera names must match the server's *_camera_key settings.
# /dev/video0 is YUYV-only, hence the different fourcc.
FRONT_CAMERA_INDEX="${FRONT_CAMERA_INDEX:-4}"
RIGHT_CAMERA_INDEX="${RIGHT_CAMERA_INDEX:-2}"
WRIST_CAMERA_INDEX="${WRIST_CAMERA_INDEX:-0}"
FRONT_FOURCC="${FRONT_FOURCC:-MJPG}"
RIGHT_FOURCC="${RIGHT_FOURCC:-MJPG}"
WRIST_FOURCC="${WRIST_FOURCC:-YUYV}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"

FPS="${FPS:-8}"
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-16}"
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0.7}"
AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-conservative}"
RUN_SECONDS="${RUN_SECONDS:-120}"
SHADOW="${SHADOW:-false}"
SCREENSHOT_HZ="${SCREENSHOT_HZ:-2.0}"
ACTUATOR_SAMPLE_EVERY="${ACTUATOR_SAMPLE_EVERY:-4}"
TASK="${TASK:-go to red cube. take the red cube. go to box. put the red cube in box.}"

# Client-side per-command clamp, in degrees. Empty = disabled. The server
# already bounds the chunk; this is only a backstop, so keep it above the
# server's max_step_delta or it will fight it.
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-}"

# Default true: on shutdown the stock setting drops torque, so a raised arm
# falls. Holding is safer.
DISABLE_TORQUE_ON_DISCONNECT="${DISABLE_TORQUE_ON_DISCONNECT:-false}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
BASE_DIR="${BASE_DIR:-${CKPT_ROOT}/profiling}"
TRACE_DIR="${TRACE_DIR:-${BASE_DIR}/${RUN_TAG}}"
CLIENT_LOG="${CLIENT_LOG:-${TRACE_DIR}/client.log}"

mkdir -p "${TRACE_DIR}"

_cam() { echo "$1: {type: opencv, index_or_path: $2, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, fourcc: \"$3\"}"; }
CAMERAS="{ $(_cam front "${FRONT_CAMERA_INDEX}" "${FRONT_FOURCC}"), \
$(_cam right "${RIGHT_CAMERA_INDEX}" "${RIGHT_FOURCC}"), \
$(_cam wrist "${WRIST_CAMERA_INDEX}" "${WRIST_FOURCC}") }"

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/home/hrx/.cache/huggingface}"

if ! ss -ltn 2>/dev/null | grep -q ":${SERVER_ADDRESS##*:}\b"; then
  echo "ERROR: no policy server on ${SERVER_ADDRESS}." >&2
  echo "Start run_cosmos20k_server_thor.sh and wait for it to finish loading." >&2
  exit 3
fi

EXTRA=()
[[ -n "${MAX_RELATIVE_TARGET}" ]] && EXTRA+=(--robot.max_relative_target="${MAX_RELATIVE_TARGET}")

echo "=========================================================="
if [[ "${SHADOW}" == "true" ]]; then
  echo " SHADOW RUN -- motor bus read-only, the arm will NOT move."
else
  echo " REAL RUN -- THE ARM WILL MOVE. Stay at the e-stop."
fi
echo "   ckpt     : ${CKPT_ROOT##*/}"
echo "   robot    : ${ROBOT_PORT}   fps=${FPS}   run_seconds=${RUN_SECONDS}"
echo "   cameras  : front=${FRONT_CAMERA_INDEX} right=${RIGHT_CAMERA_INDEX} wrist=${WRIST_CAMERA_INDEX}"
if [[ -n "${MAX_RELATIVE_TARGET}" ]]; then
  echo "   clamp    : ${MAX_RELATIVE_TARGET} deg per command"
else
  echo "   clamp    : client-side clamp DISABLED (server clamps still apply)"
fi
echo "   traces   : ${TRACE_DIR}"
echo "=========================================================="

cd "${REPO_DIR}"

exec "${PYTHON_BIN}" -u scripts/inference/cosmos_profiled_client.py \
  --server_address="${SERVER_ADDRESS}" \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.use_degrees=true \
  --robot.disable_torque_on_disconnect="${DISABLE_TORQUE_ON_DISCONNECT}" \
  "${EXTRA[@]}" \
  --robot.cameras="${CAMERAS}" \
  --task="${TASK}" \
  --policy_type=cosmos_policy \
  --pretrained_name_or_path="${CKPT_ROOT}" \
  --policy_device=cuda \
  --client_device=cpu \
  --actions_per_chunk="${ACTIONS_PER_CHUNK}" \
  --chunk_size_threshold="${CHUNK_SIZE_THRESHOLD}" \
  --aggregate_fn_name="${AGGREGATE_FN_NAME}" \
  --fps="${FPS}" \
  --run_seconds="${RUN_SECONDS}" \
  --shadow="${SHADOW}" \
  --trace_dir="${TRACE_DIR}" \
  --screenshot_hz="${SCREENSHOT_HZ}" \
  --actuator_sample_every="${ACTUATOR_SAMPLE_EVERY}" \
  --record_timeline=true \
  --timeline_log_dir="${TRACE_DIR}" \
  --timeline_save_images=off \
  --debug_visualize_queue_size=false \
  "$@" 2>&1 | tee "${CLIENT_LOG}"
