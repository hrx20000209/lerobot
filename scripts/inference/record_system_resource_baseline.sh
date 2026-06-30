#!/usr/bin/env bash
set -euo pipefail

NAME="${NAME:-baseline}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DURATION_S="${DURATION_S:-30}"
SYSTEM_RESOURCE_INTERVAL_S="${SYSTEM_RESOURCE_INTERVAL_S:-1.0}"
SYSTEM_RESOURCE_SAMPLE_NVIDIA_SMI="${SYSTEM_RESOURCE_SAMPLE_NVIDIA_SMI:-false}"
LOG_DIR="${LOG_DIR:-/home/hrx/Projects/lerobot/logs/system_resources}"

EXTRA_ARGS=()
if [[ "${SYSTEM_RESOURCE_SAMPLE_NVIDIA_SMI}" == "true" ]]; then
  EXTRA_ARGS+=(--sample-nvidia-smi)
fi

"${PYTHON_BIN}" -m lerobot.async_inference.system_resources \
  --name="${NAME}" \
  --log-dir="${LOG_DIR}" \
  --duration-s="${DURATION_S}" \
  --interval-s="${SYSTEM_RESOURCE_INTERVAL_S}" \
  "${EXTRA_ARGS[@]}"
