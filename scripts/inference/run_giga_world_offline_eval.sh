#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/hrx/Projects/lerobot}"
PYTHON_BIN="${PYTHON_BIN:-/home/hrx/miniconda3/envs/lerobot/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/hrx/Projects/models/three_cubes_1/giga_world}"
DATASET_ROOT="${DATASET_ROOT:-/home/hrx/Datasets/three_cubes_1}"
EPISODE_INDEX="${EPISODE_INDEX:-0}"
OUT_DIR="${OUT_DIR:-/home/hrx/Projects/lerobot/outputs/giga_world_eval}"

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export GIGA_WORLD_POLICY_ROOT="${GIGA_WORLD_POLICY_ROOT:-/home/hrx/Projects/giga-world-policy}"
export GIGA_WORLD_MODEL_CACHE_DIR="${GIGA_WORLD_MODEL_CACHE_DIR:-/home/hrx/Projects/models/giga-world-policy}"
export HF_HOME="${HF_HOME:-/home/hrx/.cache/huggingface}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "${OUT_DIR}"
cd "${REPO_DIR}"

EXTRA_ARGS=()
if [[ -n "${MAX_FRAMES:-}" ]]; then
  EXTRA_ARGS+=(--max-frames "${MAX_FRAMES}")
fi
if [[ -n "${NUM_INFERENCE_STEPS:-}" ]]; then
  EXTRA_ARGS+=(--num-inference-steps "${NUM_INFERENCE_STEPS}")
fi

exec "${PYTHON_BIN}" scripts/inference/eval_vla_jepa_episode_curve.py \
  --dataset-root "${DATASET_ROOT}" \
  --model-path "${MODEL_PATH}" \
  --episode-index "${EPISODE_INDEX}" \
  --device "${POLICY_DEVICE:-cuda}" \
  --horizon "${HORIZON:-48}" \
  --second-camera-key observation.images.wrist \
  --second-camera-policy-key observation.images.wrist \
  --out "${OUT_DIR}/episode_${EPISODE_INDEX}_overview.png" \
  --npz-out "${OUT_DIR}/episode_${EPISODE_INDEX}_actions.npz" \
  "${EXTRA_ARGS[@]}"
