#!/usr/bin/env bash
set -euo pipefail

# PI0.5 continuous async policy server.
# The checkpoint is loaded after the robot client sends policy instructions.

REPO_DIR="${REPO_DIR:-/home/hrx/Projects/lerobot}"

export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-8080}"
export FPS="${FPS:-30}"
export OBS_QUEUE_TIMEOUT="${OBS_QUEUE_TIMEOUT:-0.001}"
export TIMELINE_LOG_PATH="${TIMELINE_LOG_PATH:-${REPO_DIR}/outputs/continuous_async/pi05_server_timeline.jsonl}"

exec "${REPO_DIR}/scripts/inference/run_continuous_async_policy_server.sh"
