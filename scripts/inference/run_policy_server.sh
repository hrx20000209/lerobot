#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PORT="${PORT:-8080}"
FPS="${FPS:-30}"
INFERENCE_LATENCY="${INFERENCE_LATENCY:-0.033}"
OBS_QUEUE_TIMEOUT="${OBS_QUEUE_TIMEOUT:-2}"
RECORD_TIMELINE="${RECORD_TIMELINE:-true}"
TIMELINE_LOG_DIR="${TIMELINE_LOG_DIR:-/home/hrx/Projects/lerobot/logs/async_timeline}"
RECORD_SYSTEM_RESOURCES="${RECORD_SYSTEM_RESOURCES:-false}"
SYSTEM_RESOURCE_INTERVAL_S="${SYSTEM_RESOURCE_INTERVAL_S:-1.0}"
SYSTEM_RESOURCE_SAMPLE_NVIDIA_SMI="${SYSTEM_RESOURCE_SAMPLE_NVIDIA_SMI:-true}"

"${PYTHON_BIN}" -m lerobot.async_inference.policy_server \
  --host="${HOST}" \
  --port="${PORT}" \
  --fps="${FPS}" \
  --inference_latency="${INFERENCE_LATENCY}" \
  --obs_queue_timeout="${OBS_QUEUE_TIMEOUT}" \
  --record_timeline="${RECORD_TIMELINE}" \
  --timeline_log_dir="${TIMELINE_LOG_DIR}" \
  --record_system_resources="${RECORD_SYSTEM_RESOURCES}" \
  --system_resource_interval_s="${SYSTEM_RESOURCE_INTERVAL_S}" \
  --system_resource_sample_nvidia_smi="${SYSTEM_RESOURCE_SAMPLE_NVIDIA_SMI}"
