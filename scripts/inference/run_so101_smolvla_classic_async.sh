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
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0.5}"
AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-conservative}"
FPS="${FPS:-30}"
INFERENCE_LATENCY="${INFERENCE_LATENCY:-0.033}"
OBS_QUEUE_TIMEOUT="${OBS_QUEUE_TIMEOUT:-2}"

# The classic async client has no shadow mode: if it runs, it sends actions to
# the robot. Keep an explicit guard so this script is not accidentally dangerous.
ENABLE_ROBOT_EXECUTION="${ENABLE_ROBOT_EXECUTION:-False}"
RUN_SECONDS="${RUN_SECONDS:-8}"
CLIENT_TIMEOUT_GRACE="${CLIENT_TIMEOUT_GRACE:-180}"
DISPLAY_DATA="${DISPLAY_DATA:-False}"
TIMELINE_SAVE_IMAGES="${TIMELINE_SAVE_IMAGES:-key}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="${LOG_ROOT:-/tmp/so101_smolvla_classic_async}"
RUN_DIR="${RUN_DIR:-${LOG_ROOT}/${RUN_ID}}"
TIMELINE_LOG_DIR="${RUN_DIR}/timeline_logs"
SERVER_STDOUT_LOG="${RUN_DIR}/server_stdout.log"
CLIENT_STDOUT_LOG="${RUN_DIR}/client_stdout.log"
TIMELINE_PNG="${RUN_DIR}/timeline.png"

PLOT_SECONDS_PER_INCH="${PLOT_SECONDS_PER_INCH:-1.4}"
PLOT_MIN_FIG_WIDTH="${PLOT_MIN_FIG_WIDTH:-24}"
PLOT_FIG_HEIGHT="${PLOT_FIG_HEIGHT:-9}"
PLOT_DPI="${PLOT_DPI:-220}"
PLOT_OBS_LABEL_STRIDE="${PLOT_OBS_LABEL_STRIDE:-1}"
PLOT_INFERENCE_LABEL_STRIDE="${PLOT_INFERENCE_LABEL_STRIDE:-1}"
PLOT_ACTION_LABEL_STRIDE="${PLOT_ACTION_LABEL_STRIDE:-5}"
PLOT_QUEUE_LABEL_STRIDE="${PLOT_QUEUE_LABEL_STRIDE:-3}"

export PYTHONPATH="${LEROBOT_DIR}:${PYTHONPATH:-}"

mkdir -p "${RUN_DIR}" "${TIMELINE_LOG_DIR}"
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

if [[ ! "${ENABLE_ROBOT_EXECUTION}" =~ ^([Tt]rue|1|yes|YES)$ ]]; then
  echo "Classic async has no shadow mode and will command motors." >&2
  echo "Set ENABLE_ROBOT_EXECUTION=True to run it." >&2
  exit 2
fi

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
    echo "Stopping classic async server pid=${SERVER_PID}"
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

echo "Starting classic async policy server: ${SERVER_ADDRESS}"
"${PYTHON_BIN}" -u -m lerobot.async_inference.policy_server \
  --host="${HOST}" \
  --port="${PORT}" \
  --fps="${FPS}" \
  --inference_latency="${INFERENCE_LATENCY}" \
  --obs_queue_timeout="${OBS_QUEUE_TIMEOUT}" \
  --record_timeline=True \
  --timeline_log_dir="${TIMELINE_LOG_DIR}" \
  >"${SERVER_STDOUT_LOG}" 2>&1 &
SERVER_PID="$!"

echo "Waiting for server..."
for _ in $(seq 1 120); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Classic async server exited during startup. Last log lines:" >&2
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
  "${PYTHON_BIN}" -u -m lerobot.async_inference.robot_client
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
  --fps="${FPS}"
  --run_seconds="${RUN_SECONDS}"
  --record_timeline=True
  --timeline_log_dir="${TIMELINE_LOG_DIR}"
  --timeline_save_images="${TIMELINE_SAVE_IMAGES}"
  --display_data="${DISPLAY_DATA}"
  --debug_visualize_queue_size=False
)

echo "Starting classic async robot client. enable_robot_execution=${ENABLE_ROBOT_EXECUTION}"
CLIENT_TIMEOUT_SECONDS="$("${PYTHON_BIN}" - <<PY
import math
print(int(math.ceil(float("${RUN_SECONDS}") + float("${CLIENT_TIMEOUT_GRACE}"))))
PY
)"
timeout --signal=INT "${CLIENT_TIMEOUT_SECONDS}" "${CLIENT_CMD[@]}" 2>&1 | tee "${CLIENT_STDOUT_LOG}" || {
  status=$?
  if [[ "${status}" != "124" && "${status}" != "130" ]]; then
    exit "${status}"
  fi
}

LOG_PATHS="$(find "${TIMELINE_LOG_DIR}" -type f -name '*_latency_*.jsonl' | sort | tr '\n' ' ')"
env \
  LOG_PATH="${LOG_PATHS}" \
  OUTPUT_DIR="${RUN_DIR}" \
  OUTPUT_PATH="${TIMELINE_PNG}" \
  SUMMARY_PATH="${RUN_DIR}/timeline_summary.json" \
  WINDOW_DURATION="${RUN_SECONDS}" \
  SECONDS_PER_INCH="${PLOT_SECONDS_PER_INCH}" \
  MIN_FIG_WIDTH="${PLOT_MIN_FIG_WIDTH}" \
  FIG_HEIGHT="${PLOT_FIG_HEIGHT}" \
  DPI="${PLOT_DPI}" \
  OBS_LABEL_STRIDE="${PLOT_OBS_LABEL_STRIDE}" \
  INFERENCE_LABEL_STRIDE="${PLOT_INFERENCE_LABEL_STRIDE}" \
  ACTION_LABEL_STRIDE="${PLOT_ACTION_LABEL_STRIDE}" \
  QUEUE_LABEL_STRIDE="${PLOT_QUEUE_LABEL_STRIDE}" \
  "${REPO_DIR}/scripts/inference/plot_latest_async_timeline.sh" \
  >"${RUN_DIR}/timeline_plot_stdout.log" \
  2>"${RUN_DIR}/timeline_plot_stderr.log"

echo "Run directory: ${RUN_DIR}"
echo "Timeline logs: ${TIMELINE_LOG_DIR}"
echo "Timeline PNG: ${TIMELINE_PNG}"
echo "Timeline PDF: ${TIMELINE_PNG%.*}.pdf"
echo "Server log: ${SERVER_STDOUT_LOG}"
echo "Client log: ${CLIENT_STDOUT_LOG}"
