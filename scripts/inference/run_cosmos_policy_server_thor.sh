#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/hrx/Projects/lerobot}"
PYTHON_BIN="${PYTHON_BIN:-/home/hrx/Projects/cosmos-policy/.venv/bin/python}"
COSMOS_REPO="${COSMOS_REPO:-/home/hrx/Projects/cosmos-policy}"
LEROBOT_SRC="${LEROBOT_SRC:-${REPO_DIR}/src}"
GPU="${GPU:-0}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
FPS="${FPS:-30}"
INFERENCE_LATENCY="${INFERENCE_LATENCY:-0.033}"
OBS_QUEUE_TIMEOUT="${OBS_QUEUE_TIMEOUT:-2}"
RECORD_TIMELINE="${RECORD_TIMELINE:-true}"
TIMELINE_LOG_DIR="${TIMELINE_LOG_DIR:-/home/hrx/Projects/lerobot/logs/async_timeline}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${LEROBOT_SRC}:${COSMOS_REPO}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/home/hrx/.cache/huggingface}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

cd "${REPO_DIR}"

exec "${PYTHON_BIN}" -m lerobot.async_inference.policy_server \
  --host="${HOST}" \
  --port="${PORT}" \
  --fps="${FPS}" \
  --inference_latency="${INFERENCE_LATENCY}" \
  --obs_queue_timeout="${OBS_QUEUE_TIMEOUT}" \
  --record_timeline="${RECORD_TIMELINE}" \
  --timeline_log_dir="${TIMELINE_LOG_DIR}"
