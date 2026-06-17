#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
FPS="${FPS:-30}"
INFERENCE_LATENCY="${INFERENCE_LATENCY:-0.033}"
OBS_QUEUE_TIMEOUT="${OBS_QUEUE_TIMEOUT:-2}"
RECORD_TIMELINE="${RECORD_TIMELINE:-true}"
TIMELINE_LOG_DIR="${TIMELINE_LOG_DIR:-/home/rxhuang/Projects/lerobot/logs/async_timeline}"

cd /home/rxhuang/Projects/lerobot

python -m lerobot.async_inference.policy_server \
  --host="${HOST}" \
  --port="${PORT}" \
  --fps="${FPS}" \
  --inference_latency="${INFERENCE_LATENCY}" \
  --obs_queue_timeout="${OBS_QUEUE_TIMEOUT}" \
  --record_timeline="${RECORD_TIMELINE}" \
  --timeline_log_dir="${TIMELINE_LOG_DIR}"
