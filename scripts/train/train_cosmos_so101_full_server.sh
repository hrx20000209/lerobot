#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/rxhuang/Projects/lerobot}"
COSMOS_REPO="${COSMOS_REPO:-/home/rxhuang/Projects/cosmos-policy}"
LEROBOT_SRC="${LEROBOT_SRC:-${REPO_DIR}/src}"
PYTHON_BIN="${PYTHON_BIN:-/home/rxhuang/Projects/cosmos-policy/.venv/bin/python}"

GPU="${GPU:-0}"
STEPS="${STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SAVE_FREQ="${SAVE_FREQ:-500}"
LOG_FREQ="${LOG_FREQ:-10}"
NUM_WORKERS="${NUM_WORKERS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-/data/rxhuang/lerobot_cosmos_so101_full_debug}"
DATASET_ROOT="${DATASET_ROOT:-/data/rxhuang/three_cubes_1}"
DATASET_REPO_ID="${DATASET_REPO_ID:-hrx2000/Three_Cubes_1}"
DATASET_REVISION="${DATASET_REVISION:-v0.1.0}"
DATASET_STATS_PATH="${DATASET_STATS_PATH:-${DATASET_ROOT}/so101_dataset_statistics.json}"
T5_TEXT_EMBEDDINGS_PATH="${T5_TEXT_EMBEDDINGS_PATH:-${DATASET_ROOT}/so101_t5_embeddings.pkl}"

DEFAULT_CKPT="/home/rxhuang/Projects/models/cosmos"
FALLBACK_CKPT="/data/rxhuang/cosmos_action_focused_runs/cosmos_policy/so101_lerobot/action_focused_B_full_dit_2gpu_smoke_20260707/checkpoints/iter_000010000"
CKPT_PATH="${CKPT_PATH:-${DEFAULT_CKPT}}"
if [[ ! -e "${CKPT_PATH}" ]]; then
  echo "Requested Cosmos checkpoint does not exist: ${CKPT_PATH}" >&2
  echo "Available Cosmos checkpoint candidates:" >&2
  find /data/rxhuang/cosmos_action_focused_runs /data/rxhuang/cosmos_three_cubes_runs \
    -path '*/checkpoints/iter_*' -type d 2>/dev/null | sort | tail -40 >&2 || true
  CKPT_PATH="${FALLBACK_CKPT}"
  echo "Using fallback checkpoint: ${CKPT_PATH}" >&2
fi
if [[ ! -e "${CKPT_PATH}" ]]; then
  echo "No usable Cosmos checkpoint found." >&2
  exit 2
fi

INPUT_FEATURES='{
  "observation.state": {"type": "STATE", "shape": [6]},
  "observation.images.front": {"type": "VISUAL", "shape": [3, 480, 640]},
  "observation.images.right": {"type": "VISUAL", "shape": [3, 480, 640]},
  "observation.images.wrist": {"type": "VISUAL", "shape": [3, 480, 640]}
}'

export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTHONPATH="${LEROBOT_SRC}:${COSMOS_REPO}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/data/hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

cd "${REPO_DIR}"

exec "${PYTHON_BIN}" -m lerobot.scripts.lerobot_train \
  --policy.type=cosmos \
  --policy.input_features="${INPUT_FEATURES}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.revision="${DATASET_REVISION}" \
  --dataset.streaming=false \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=cosmos_so101_full_dit \
  --policy.device=cuda \
  --policy.cosmos_repo="${COSMOS_REPO}" \
  --policy.ckpt_path="${CKPT_PATH}" \
  --policy.dataset_root="${DATASET_ROOT}" \
  --policy.dataset_repo_id="${DATASET_REPO_ID}" \
  --policy.dataset_stats_path="${DATASET_STATS_PATH}" \
  --policy.t5_text_embeddings_path="${T5_TEXT_EMBEDDINGS_PATH}" \
  --policy.primary_camera_key=observation.images.front \
  --policy.wrist_left_camera_key=observation.images.right \
  --policy.wrist_camera_key=observation.images.wrist \
  --policy.train_mode=full_dit \
  --policy.action_loss_weight=1.0 \
  --policy.visual_loss_weight=0.0 \
  --policy.future_state_loss_weight=0.0 \
  --policy.chunk_size=50 \
  --policy.actions_per_chunk=50 \
  --policy.n_action_steps=16 \
  --policy.num_denoising_steps_action=10 \
  --policy.optimizer_lr=1e-5 \
  --policy.optimizer_weight_decay=1e-4 \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --log_freq="${LOG_FREQ}" \
  --save_checkpoint=true \
  --save_freq="${SAVE_FREQ}" \
  "$@"
