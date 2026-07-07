#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-/home/hrx/Projects/lerobot}"
LEROBOT_DIR="${LEROBOT_DIR:-${REPO_DIR}/src}"
PYTHON_BIN="${PYTHON_BIN:-/home/hrx/miniconda3/envs/lerobot/bin/python}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
SERVER_ADDRESS="${HOST}:${PORT}"

ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM1}"
ROBOT_ID="${ROBOT_ID:-follower_arm}"

FRONT_CAMERA="${FRONT_CAMERA:-2}"
RIGHT_CAMERA="${RIGHT_CAMERA:-0}"
WRIST_CAMERA="${WRIST_CAMERA:-4}"
CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
CAMERA_FPS="${CAMERA_FPS:-30}"
CAMERA_FOURCC="${CAMERA_FOURCC:-MJPG}"

POLICY_TYPE="${POLICY_TYPE:-smolvla}"
PRETRAINED_NAME_OR_PATH="${PRETRAINED_NAME_OR_PATH:-/home/hrx/Projects/models/three_cubes_1/smolvla/}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
CLIENT_DEVICE="${CLIENT_DEVICE:-cpu}"
TASK="${TASK:-go to red cube. take the red cube. go to box. put the red cube in box.}"

ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-50}"
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-1.0}"
AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-conservative}"
CONTINUOUS_AGGREGATION_FN="${CONTINUOUS_AGGREGATION_FN:-latency_aligned_blend}"
FPS="${FPS:-30}"
CONTINUOUS_OBS_FPS="${CONTINUOUS_OBS_FPS:-30}"
OBS_QUEUE_TIMEOUT="${OBS_QUEUE_TIMEOUT:-0.05}"
INFERENCE_LATENCY="${INFERENCE_LATENCY:-0}"

# Safety default: do not move motors. To execute, run:
#   SHADOW_MODE=False ENABLE_ROBOT_EXECUTION=True RUN_SECONDS=5 ./scripts/inference/run_so101_smolvla_continuous_async.sh
SHADOW_MODE="${SHADOW_MODE:-True}"
ENABLE_ROBOT_EXECUTION="${ENABLE_ROBOT_EXECUTION:-False}"
RUN_SECONDS="${RUN_SECONDS:-8}"

MAX_JOINT_DELTA="${MAX_JOINT_DELTA:-15}"
MAX_GRIPPER_DELTA="${MAX_GRIPPER_DELTA:-15}"
MAX_JOINT_DELTA_PER_STEP="${MAX_JOINT_DELTA_PER_STEP:-8}"
MAX_GRIPPER_DELTA_PER_STEP="${MAX_GRIPPER_DELTA_PER_STEP:-12}"
STALE_INFERENCE_MAX_AGE="${STALE_INFERENCE_MAX_AGE:-2.0}"
MIN_USABLE_ACTIONS="${MIN_USABLE_ACTIONS:-5}"
MAX_CONTROL_STEPS="${MAX_CONTROL_STEPS:-}"
if [[ -z "${MAX_CONTROL_STEPS}" && "${ENABLE_ROBOT_EXECUTION}" =~ ^([Tt]rue|1|yes|YES)$ ]]; then
  MAX_CONTROL_STEPS="5"
fi

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-/tmp/so101_smolvla_continuous_async}"
RUN_DIR="${RUN_DIR:-${LOG_ROOT}/${RUN_ID}}"
SERVER_STDOUT_LOG="${RUN_DIR}/server_stdout.log"
CLIENT_STDOUT_LOG="${RUN_DIR}/client_stdout.log"
TIMELINE_JSONL="${RUN_DIR}/timeline.jsonl"
TIMELINE_PNG="${RUN_DIR}/timeline.png"

export PYTHONPATH="${LEROBOT_DIR}:${PYTHONPATH:-}"

mkdir -p "${RUN_DIR}"
cd "${REPO_DIR}"

require_path() {
  local path="$1"
  if [[ ! -e "${path}" ]]; then
    echo "Missing required path: ${path}" >&2
    exit 1
  fi
}

require_path "${PYTHON_BIN}"
require_path "${ROBOT_PORT}"
require_path "/dev/video${FRONT_CAMERA}"
require_path "/dev/video${RIGHT_CAMERA}"
require_path "/dev/video${WRIST_CAMERA}"
require_path "${PRETRAINED_NAME_OR_PATH}"

if "${PYTHON_BIN}" - <<PY
import socket
s = socket.socket()
s.settimeout(0.2)
raise SystemExit(0 if s.connect_ex(("${HOST}", ${PORT})) == 0 else 1)
PY
then
  echo "Port ${PORT} is already in use. Stop the old async server first." >&2
  exit 1
fi

SERVER_PID=""
cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Stopping continuous async server pid=${SERVER_PID}"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "Starting continuous async policy server: ${SERVER_ADDRESS}"
"${PYTHON_BIN}" -u -m lerobot.async_inference.continuous_policy_server \
  --host="${HOST}" \
  --port="${PORT}" \
  --fps="${FPS}" \
  --inference_latency="${INFERENCE_LATENCY}" \
  --obs_queue_timeout="${OBS_QUEUE_TIMEOUT}" \
  --record_timeline=True \
  --timeline_log_dir="${RUN_DIR}" \
  --timeline_log_path="${TIMELINE_JSONL}" \
  >"${SERVER_STDOUT_LOG}" 2>&1 &
SERVER_PID="$!"

echo "Waiting for server..."
for _ in $(seq 1 120); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Continuous async server exited during startup. Last log lines:" >&2
    tail -80 "${SERVER_STDOUT_LOG}" >&2 || true
    exit 1
  fi
  if "${PYTHON_BIN}" - <<PY
import socket
s = socket.socket()
s.settimeout(0.2)
raise SystemExit(0 if s.connect_ex(("${HOST}", ${PORT})) == 0 else 1)
PY
  then
    break
  fi
  sleep 1
done

ROBOT_CAMERAS="{ front: {type: opencv, index_or_path: ${FRONT_CAMERA}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, fourcc: \"${CAMERA_FOURCC}\"}, right: {type: opencv, index_or_path: ${RIGHT_CAMERA}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, fourcc: \"${CAMERA_FOURCC}\"}, wrist: {type: opencv, index_or_path: ${WRIST_CAMERA}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${CAMERA_FPS}, fourcc: \"${CAMERA_FOURCC}\"} }"

CLIENT_CMD=(
  "${PYTHON_BIN}" -u -m lerobot.async_inference.continuous_robot_client
  --server_address="${SERVER_ADDRESS}"
  --robot.type=so101_follower
  --robot.port="${ROBOT_PORT}"
  --robot.id="${ROBOT_ID}"
  --robot.cameras="${ROBOT_CAMERAS}"
  --task="${TASK}"
  --policy_type="${POLICY_TYPE}"
  --pretrained_name_or_path="${PRETRAINED_NAME_OR_PATH}"
  --policy_device="${POLICY_DEVICE}"
  --client_device="${CLIENT_DEVICE}"
  --actions_per_chunk="${ACTIONS_PER_CHUNK}"
  --chunk_size_threshold="${CHUNK_SIZE_THRESHOLD}"
  --aggregate_fn_name="${AGGREGATE_FN_NAME}"
  --aggregation_fn="${CONTINUOUS_AGGREGATION_FN}"
  --fps="${FPS}"
  --continuous_obs_fps="${CONTINUOUS_OBS_FPS}"
  --record_timeline=True
  --timeline_log_dir="${RUN_DIR}"
  --timeline_log_path="${TIMELINE_JSONL}"
  --timeline_plot_path="${TIMELINE_PNG}"
  --stale_inference_max_age="${STALE_INFERENCE_MAX_AGE}"
  --min_usable_actions="${MIN_USABLE_ACTIONS}"
  --max_joint_delta="${MAX_JOINT_DELTA}"
  --max_gripper_delta="${MAX_GRIPPER_DELTA}"
  --max_joint_delta_per_step="${MAX_JOINT_DELTA_PER_STEP}"
  --max_gripper_delta_per_step="${MAX_GRIPPER_DELTA_PER_STEP}"
  --shadow_mode="${SHADOW_MODE}"
  --enable_robot_execution="${ENABLE_ROBOT_EXECUTION}"
  --debug_visualize_queue_size=True
)

if [[ -n "${MAX_CONTROL_STEPS}" ]]; then
  CLIENT_CMD+=(--max_control_steps="${MAX_CONTROL_STEPS}")
fi

echo "Starting continuous async robot client. shadow_mode=${SHADOW_MODE} enable_robot_execution=${ENABLE_ROBOT_EXECUTION}"
if [[ "${RUN_SECONDS}" == "0" ]]; then
  "${CLIENT_CMD[@]}" 2>&1 | tee "${CLIENT_STDOUT_LOG}"
else
  timeout --signal=INT "${RUN_SECONDS}" "${CLIENT_CMD[@]}" 2>&1 | tee "${CLIENT_STDOUT_LOG}" || {
    status=$?
    if [[ "${status}" != "124" && "${status}" != "130" ]]; then
      exit "${status}"
    fi
  }
fi

if [[ -s "${TIMELINE_JSONL}" ]]; then
  PLOT_WINDOW_DURATION="${RUN_SECONDS}"
  if [[ "${PLOT_WINDOW_DURATION}" == "0" ]]; then
    PLOT_WINDOW_DURATION="8"
  fi
  "${PYTHON_BIN}" -m lerobot.scripts.plot_async_timeline \
    --log_path "${TIMELINE_JSONL}" \
    --output_path "${TIMELINE_PNG}" \
    --window_start 0 \
    --window_duration "${PLOT_WINDOW_DURATION}" \
    --summary_path "${RUN_DIR}/timeline_summary.json" \
    >"${RUN_DIR}/timeline_plot_stdout.log"
fi

echo "Run directory: ${RUN_DIR}"
echo "Timeline JSONL: ${TIMELINE_JSONL}"
echo "Timeline PNG: ${TIMELINE_PNG}"
echo "Server log: ${SERVER_STDOUT_LOG}"
echo "Client log: ${CLIENT_STDOUT_LOG}"
