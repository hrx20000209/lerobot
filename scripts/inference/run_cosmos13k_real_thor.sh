#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# REAL CLOSED-LOOP RUN -- THIS SCRIPT WRITES GOAL POSITIONS TO THE MOTOR BUS.
# The arm will move.  A human must be next to it, ready to physically e-stop
# or cut power.  This is the first closed-loop test of the step-13000
# checkpoint; its real-robot behaviour is unknown.
#
# For validation without motion use run_cosmos13k_shadow_client_thor.sh, which
# cannot write to the bus at all.
# ============================================================================
#
# Conservative first-run parameters:
#   fps=8                   -- measured: inference is ~842 ms for 16 actions, so
#                              at 15 fps a chunk covers only 1.07 s and most of
#                              it is already stale on arrival; the run at 15 fps
#                              executed only 220 of 464 actions and drifted to an
#                              effective 8.3 Hz with p50 observation age 1.2 s.
#                              8 fps makes a chunk span 2 s.  Must match the
#                              server's fps: it stamps action timestamps 1/fps.
#   run_seconds=30          -- short, supervised.  Still enforced.
#   max_relative_target     -- DISABLED by default at the user's request, so the
#                              raw model action reaches the motors unmodified.
#                              Set MAX_RELATIVE_TARGET=<deg> to re-enable.
#   server clamps           -- also disabled by default in the server script.
#                              Exposure: chunk-boundary jumps up to ~20 deg on
#                              shoulder_lift in a single step for this
#                              checkpoint.  Re-enable with the env vars listed
#                              in run_cosmos13k_shadow_server_thor.sh.
#   disable_torque_on_disconnect=false
#                           -- default true makes the arm go limp on shutdown,
#                              i.e. drop from wherever it is.  Hold instead.

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

FPS="${FPS:-8}"
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-16}"
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0.7}"
AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-conservative}"
RUN_SECONDS="${RUN_SECONDS:-300}"
# Empty = no client-side clamp; the model's action goes to the motors unmodified.
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-}"
TASK="${TASK:-go to red cube. take the red cube. go to box. put the red cube in box.}"

LOG_DIR="${LOG_DIR:-/home/hrx/Projects/models/three_cubes_1/cosmos_policy_step13000/real_logs}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
TIMELINE_LOG_DIR="${TIMELINE_LOG_DIR:-${LOG_DIR}/timeline_${RUN_TAG}}"
CLIENT_LOG="${CLIENT_LOG:-${LOG_DIR}/client_real_${RUN_TAG}.log}"

mkdir -p "${LOG_DIR}" "${TIMELINE_LOG_DIR}"

CAMERAS="{ front: {type: opencv, index_or_path: ${FRONT_CAMERA_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, fourcc: \"${FRONT_FOURCC}\"}, \
right: {type: opencv, index_or_path: ${RIGHT_CAMERA_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, fourcc: \"${RIGHT_FOURCC}\"}, \
wrist: {type: opencv, index_or_path: ${WRIST_CAMERA_INDEX}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, fourcc: \"${WRIST_FOURCC}\"} }"

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/home/hrx/.cache/huggingface}"

# Omit the flag entirely when empty: passing it is what enables the clamp.
CLAMP_ARGS=()
if [[ -n "${MAX_RELATIVE_TARGET}" ]]; then
  CLAMP_ARGS+=(--robot.max_relative_target="${MAX_RELATIVE_TARGET}")
fi

echo "=========================================================="
echo " REAL RUN: the arm WILL move. Stay next to the e-stop."
echo "   port=${ROBOT_PORT}  fps=${FPS}  run_seconds=${RUN_SECONDS}"
if [[ -n "${MAX_RELATIVE_TARGET}" ]]; then
  echo "   max_relative_target=${MAX_RELATIVE_TARGET} deg per command"
else
  echo "   max_relative_target=DISABLED -- raw model action goes to the motors"
fi
echo "   cameras: front=${FRONT_CAMERA_INDEX} right=${RIGHT_CAMERA_INDEX} wrist=${WRIST_CAMERA_INDEX}"
echo "   log -> ${CLIENT_LOG}"
echo "=========================================================="

cd "${REPO_DIR}"

exec "${PYTHON_BIN}" -u -m lerobot.async_inference.robot_client \
  --server_address="${SERVER_ADDRESS}" \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.use_degrees=true \
  "${CLAMP_ARGS[@]}" \
  --robot.disable_torque_on_disconnect=false \
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
  --debug_visualize_queue_size=false \
  "$@" 2>&1 | tee "${CLIENT_LOG}"
