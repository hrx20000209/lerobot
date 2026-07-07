#!/usr/bin/env bash
set -euo pipefail

# Policy server wrapper for Giga World async inference.
#
# Giga World loads the local giga-world-policy checkout inside the policy_server
# process, so GIGA_WORLD_POLICY_ROOT must be set before starting the server.

REPO_DIR="${REPO_DIR:-/home/hrx/Projects/lerobot}"
PYTHON_BIN="${PYTHON_BIN:-/home/hrx/miniconda3/envs/lerobot/bin/python}"
# Jetson Thor exposes one CUDA device. Keep this overridable for other hosts.
GPU="${GPU:-0}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
FPS="${FPS:-30}"
INFERENCE_LATENCY="${INFERENCE_LATENCY:-0.033}"
OBS_QUEUE_TIMEOUT="${OBS_QUEUE_TIMEOUT:-2}"
RECORD_TIMELINE="${RECORD_TIMELINE:-true}"
TIMELINE_LOG_DIR="${TIMELINE_LOG_DIR:-/home/hrx/Projects/lerobot/logs/async_timeline/giga_world}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export GIGA_WORLD_POLICY_ROOT="${GIGA_WORLD_POLICY_ROOT:-/home/hrx/Projects/giga-world-policy}"
export GIGA_WORLD_MODEL_CACHE_DIR="${GIGA_WORLD_MODEL_CACHE_DIR:-/home/hrx/Projects/models/giga-world-policy}"
export HF_HOME="${HF_HOME:-/home/hrx/.cache/huggingface}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

if ! "${PYTHON_BIN}" -c 'from diffusers.models import AutoencoderKLWan; from diffusers.models._modeling_parallel import ContextParallelInput' >/dev/null 2>&1; then
  echo "Giga World requires diffusers>=0.36.0,<0.37.0 in ${PYTHON_BIN}." >&2
  echo "Install the pinned LeRobot environment requirements before starting the server." >&2
  exit 2
fi

if [[ ! -d "${GIGA_WORLD_POLICY_ROOT}/world_action_model" ]]; then
  echo "Missing giga-world-policy checkout at: ${GIGA_WORLD_POLICY_ROOT}" >&2
  echo "Set GIGA_WORLD_POLICY_ROOT to a checkout containing world_action_model/." >&2
  exit 2
fi

if [[ ! -d "${REPO_DIR}/src/lerobot" ]]; then
  echo "Missing LeRobot checkout at: ${REPO_DIR}" >&2
  exit 2
fi

mkdir -p "${GIGA_WORLD_MODEL_CACHE_DIR}" "${TIMELINE_LOG_DIR}"

cd "${REPO_DIR}"

exec "${PYTHON_BIN}" -m lerobot.async_inference.policy_server \
  --host="${HOST}" \
  --port="${PORT}" \
  --fps="${FPS}" \
  --inference_latency="${INFERENCE_LATENCY}" \
  --obs_queue_timeout="${OBS_QUEUE_TIMEOUT}" \
  --record_timeline="${RECORD_TIMELINE}" \
  --timeline_log_dir="${TIMELINE_LOG_DIR}"
