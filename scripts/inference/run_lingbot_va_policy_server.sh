#!/usr/bin/env bash
set -euo pipefail

# Policy server for LingBot-VA async inference (three_cubes_1 LoRA adapter).
#
# The checkpoint is built by:
#   python scripts/inference/build_lingbot_va_base_lerobot.py           # once: local base
#   python scripts/inference/build_lingbot_va_three_cubes_checkpoint.py # adapter -> servable ckpt
#
# Loading is slow by design: ~28 s for the 5 B base + ~31 s to attach the 628-module LoRA,
# then ~14 s to bring up the frozen VAE / UMT5 / tokenizer on the first inference.

REPO_DIR="${REPO_DIR:-/home/hrx/Projects/lerobot}"
PYTHON_BIN="${PYTHON_BIN:-/home/hrx/miniconda3/envs/lerobot/bin/python}"
GPU="${GPU:-0}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8080}"
# Must match the client's --fps. See run_lingbot_va_three_cubes.sh for why this is not 30.
FPS="${FPS:-3}"
INFERENCE_LATENCY="${INFERENCE_LATENCY:-4.5}"
OBS_QUEUE_TIMEOUT="${OBS_QUEUE_TIMEOUT:-10}"
RECORD_TIMELINE="${RECORD_TIMELINE:-true}"
TIMELINE_LOG_DIR="${TIMELINE_LOG_DIR:-/home/hrx/Projects/lerobot/logs/async_timeline/lingbot_va}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HOME="${HF_HOME:-/home/hrx/.cache/huggingface}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

if [[ ! -d "${REPO_DIR}/src/lerobot" ]]; then
  echo "Missing LeRobot checkout at: ${REPO_DIR}" >&2
  exit 2
fi

# The frozen VAE / UMT5 / tokenizer are NOT in the policy checkpoint; the policy pulls them
# from config.wan_pretrained_path at first inference. Fail early rather than 60 s in.
WAN_BASE="${WAN_BASE:-/home/hrx/Projects/models/lingbot-va-base}"
for sub in vae text_encoder tokenizer; do
  if [[ ! -d "${WAN_BASE}/${sub}" ]]; then
    echo "Missing ${WAN_BASE}/${sub} (frozen LingBot-VA sub-models)." >&2
    echo "Set WAN_BASE, or re-point config.wan_pretrained_path in the checkpoint." >&2
    exit 2
  fi
done

mkdir -p "${TIMELINE_LOG_DIR}"

cd "${REPO_DIR}"

exec "${PYTHON_BIN}" -m lerobot.async_inference.policy_server \
  --host="${HOST}" \
  --port="${PORT}" \
  --fps="${FPS}" \
  --inference_latency="${INFERENCE_LATENCY}" \
  --obs_queue_timeout="${OBS_QUEUE_TIMEOUT}" \
  --record_timeline="${RECORD_TIMELINE}" \
  --timeline_log_dir="${TIMELINE_LOG_DIR}"
