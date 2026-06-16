#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-2}"
STEPS="${STEPS:-40000}"
BATCH_SIZE="${BATCH_SIZE:-96}"
SAVE_FREQ="${SAVE_FREQ:-4000}"
LOG_FREQ="${LOG_FREQ:-200}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-output_lerobot_train/three_cubes/smolvla_front_wrist_b96}"
LEROBOT_TRAIN="${LEROBOT_TRAIN:-/home/rxhuang/anaconda3/envs/lerobot/bin/lerobot-train}"

INPUT_FEATURES='{
  "observation.state": {"type": "STATE", "shape": [6]},
  "observation.images.front": {"type": "VISUAL", "shape": [3, 480, 640]},
  "observation.images.wrist": {"type": "VISUAL", "shape": [3, 480, 640]}
}'

export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HOME="${HF_HOME:-/data/hf_cache}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

exec "${LEROBOT_TRAIN}" \
  --policy.type=smolvla \
  --policy.input_features="${INPUT_FEATURES}" \
  --dataset.repo_id=hrx2000/Three_Cubes_1 \
  --dataset.root=/data/rxhuang/three_cubes_1 \
  --dataset.revision=v0.1.0 \
  --dataset.streaming=false \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=three_cubes_smolvla_front_wrist \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --log_freq="${LOG_FREQ}" \
  --save_checkpoint=true \
  --save_freq="${SAVE_FREQ}"
