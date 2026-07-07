#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/hrx/Projects/lerobot}"
PYTHON_BIN="${PYTHON_BIN:-/home/hrx/miniconda3/envs/lerobot/bin/python}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/outputs/continuous_async_debug}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

exec "${PYTHON_BIN}" -m lerobot.scripts.simulate_continuous_async_inference \
  --duration="${DURATION:-8}" \
  --observation_fps="${OBSERVATION_FPS:-30}" \
  --control_fps="${CONTROL_FPS:-30}" \
  --chunk_size="${CHUNK_SIZE:-50}" \
  --inference_latency_mean="${INFERENCE_LATENCY_MEAN:-0.6}" \
  --inference_latency_jitter="${INFERENCE_LATENCY_JITTER:-0.1}" \
  --aggregation_fn="${AGGREGATION_FN:-splice_by_timestamp}" \
  --stale_inference_max_age="${STALE_INFERENCE_MAX_AGE:-2.0}" \
  --blend_horizon="${BLEND_HORIZON:-5}" \
  --blend_alpha="${BLEND_ALPHA:-0.5}" \
  --output_dir="${OUTPUT_DIR}"
