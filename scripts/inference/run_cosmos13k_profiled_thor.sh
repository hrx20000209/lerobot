#!/usr/bin/env bash
set -euo pipefail

# Instrumented Cosmos 13K run: full latency trace + async screenshots.
#
#   SHADOW=true  (default) -- motor bus read-only, the arm does NOT move.
#   SHADOW=false           -- REAL run, the arm moves.
#
# Pair with run_cosmos13k_shadow_server_thor.sh started with PROFILE_STAGES=true
# so the server attaches its per-stage breakdown to every action chunk.

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
RUN_SECONDS="${RUN_SECONDS:-60}"
SHADOW="${SHADOW:-true}"
SCREENSHOT_HZ="${SCREENSHOT_HZ:-2.0}"
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-}"
TASK="${TASK:-go to red cube. take the red cube. go to box. put the red cube in box.}"

BASE_DIR="${BASE_DIR:-/home/hrx/Projects/models/three_cubes_1/cosmos_policy_step13000/profiling}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
TRACE_DIR="${TRACE_DIR:-${BASE_DIR}/${RUN_TAG}}"
CLIENT_LOG="${CLIENT_LOG:-${TRACE_DIR}/client.log}"

mkdir -p "${TRACE_DIR}"

# View-ablation runs must also stop capturing the camera the server no longer
# consumes, otherwise the saving is only the VAE half of it.  CAMERA_SET picks
# which views the client opens; it has to agree with the server's
# use_wrist_image / num_wrist_images / left_wrist_camera_key.
#   all         -> front + right + wrist   (server: num_wrist_images=2)
#   front_wrist -> front + wrist           (server: num_wrist_images=1, left_wrist_camera_key=wrist)
#   front_right -> front + right           (server: num_wrist_images=1, left_wrist_camera_key=right)
#   front       -> front only              (server: use_wrist_image=false)
CAMERA_SET="${CAMERA_SET:-all}"
_cam() { echo "$1: {type: opencv, index_or_path: $2, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, fourcc: \"$3\"}"; }
_front="$(_cam front "${FRONT_CAMERA_INDEX}" "${FRONT_FOURCC}")"
_right="$(_cam right "${RIGHT_CAMERA_INDEX}" "${RIGHT_FOURCC}")"
_wrist="$(_cam wrist "${WRIST_CAMERA_INDEX}" "${WRIST_FOURCC}")"
case "${CAMERA_SET}" in
  all)         CAMERAS="{ ${_front}, ${_right}, ${_wrist} }" ;;
  front_wrist) CAMERAS="{ ${_front}, ${_wrist} }" ;;
  front_right) CAMERAS="{ ${_front}, ${_right} }" ;;
  front)       CAMERAS="{ ${_front} }" ;;
  *) echo "Unknown CAMERA_SET=${CAMERA_SET}" >&2; exit 2 ;;
esac

export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/home/hrx/.cache/huggingface}"

EXTRA=()
if [[ -n "${MAX_RELATIVE_TARGET}" ]]; then
  EXTRA+=(--robot.max_relative_target="${MAX_RELATIVE_TARGET}")
fi

echo "=========================================================="
if [[ "${SHADOW}" == "true" ]]; then
  echo " PROFILED SHADOW RUN -- the arm will NOT move."
else
  echo " PROFILED REAL RUN -- the arm WILL move. Stay at the e-stop."
fi
echo "   fps=${FPS} run_seconds=${RUN_SECONDS} screenshots=${SCREENSHOT_HZ} Hz"
echo "   cameras: front=${FRONT_CAMERA_INDEX} right=${RIGHT_CAMERA_INDEX} wrist=${WRIST_CAMERA_INDEX}"
echo "   traces -> ${TRACE_DIR}"
echo "=========================================================="

cd "${REPO_DIR}"

exec "${PYTHON_BIN}" -u scripts/inference/cosmos_profiled_client.py \
  --server_address="${SERVER_ADDRESS}" \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  --robot.use_degrees=true \
  --robot.disable_torque_on_disconnect=false \
  "${EXTRA[@]}" \
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
  --shadow="${SHADOW}" \
  --trace_dir="${TRACE_DIR}" \
  --screenshot_hz="${SCREENSHOT_HZ}" \
  --actuator_sample_every="${ACTUATOR_SAMPLE_EVERY:-4}" \
  --record_timeline=true \
  --timeline_log_dir="${TRACE_DIR}" \
  --timeline_save_images=off \
  --debug_visualize_queue_size=false \
  "$@" 2>&1 | tee "${CLIENT_LOG}"
