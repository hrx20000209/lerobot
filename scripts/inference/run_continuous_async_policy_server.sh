#!/usr/bin/env bash
set -euo pipefail

# Fully asynchronous continuous inference policy server.
#
# This is separate from run_policy_server.sh and does not change the original
# threshold-triggered LeRobot async server.

REPO_DIR="${REPO_DIR:-/home/hrx/Projects/lerobot}"
PYTHON_BIN="${PYTHON_BIN:-/home/hrx/miniconda3/envs/lerobot/bin/python}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
FPS="${FPS:-30}"
OBS_QUEUE_TIMEOUT="${OBS_QUEUE_TIMEOUT:-2}"
CONTINUOUS_INFERENCE_WORKERS="${CONTINUOUS_INFERENCE_WORKERS:-1}"
MAX_PENDING_OBSERVATIONS="${MAX_PENDING_OBSERVATIONS:-1}"
TIMELINE_LOG_PATH="${TIMELINE_LOG_PATH:-${REPO_DIR}/outputs/continuous_async/server_timeline.jsonl}"

RECORD_SYSTEM_RESOURCES="${RECORD_SYSTEM_RESOURCES:-false}"
SYSTEM_RESOURCE_INTERVAL_S="${SYSTEM_RESOURCE_INTERVAL_S:-1.0}"
SYSTEM_RESOURCE_SAMPLE_NVIDIA_SMI="${SYSTEM_RESOURCE_SAMPLE_NVIDIA_SMI:-false}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

if [[ ! -d "${REPO_DIR}/src/lerobot" ]]; then
  echo "Missing LeRobot checkout at: ${REPO_DIR}" >&2
  exit 2
fi

mkdir -p "$(dirname "${TIMELINE_LOG_PATH}")"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

exec "${PYTHON_BIN}" -m lerobot.async_inference.continuous_policy_server \
  --async_mode=continuous \
  --host="${HOST}" \
  --port="${PORT}" \
  --fps="${FPS}" \
  --obs_queue_timeout="${OBS_QUEUE_TIMEOUT}" \
  --continuous_inference_workers="${CONTINUOUS_INFERENCE_WORKERS}" \
  --max_pending_observations="${MAX_PENDING_OBSERVATIONS}" \
  --timeline_log_path="${TIMELINE_LOG_PATH}" \
  --record_system_resources="${RECORD_SYSTEM_RESOURCES}" \
  --system_resource_interval_s="${SYSTEM_RESOURCE_INTERVAL_S}" \
  --system_resource_sample_nvidia_smi="${SYSTEM_RESOURCE_SAMPLE_NVIDIA_SMI}"
