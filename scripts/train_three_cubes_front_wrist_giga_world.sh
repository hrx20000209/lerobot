#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-1}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
LOG_FREQ="${LOG_FREQ:-20}"
NUM_WORKERS="${NUM_WORKERS:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-/home/rxhuang/Projects/models/lerobot_train/three_cubes/giga_world_front_wrist_r64_w01_b1}"
LEROBOT_TRAIN="${LEROBOT_TRAIN:-/home/rxhuang/anaconda3/envs/lerobot/bin/lerobot-train}"

INPUT_FEATURES='{
  "observation.state": {"type": "STATE", "shape": [6]},
  "observation.images.front": {"type": "VISUAL", "shape": [3, 480, 640]},
  "observation.images.wrist": {"type": "VISUAL", "shape": [3, 480, 640]}
}'

export CUDA_VISIBLE_DEVICES="${GPU}"
export GIGA_WORLD_POLICY_ROOT="${GIGA_WORLD_POLICY_ROOT:-/home/rxhuang/Projects/giga-world-policy}"
export HF_HOME="${HF_HOME:-/data/hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

exec "${LEROBOT_TRAIN}" \
  --policy.type=giga_world \
  --policy.input_features="${INPUT_FEATURES}" \
  --dataset.repo_id=hrx2000/Three_Cubes_1 \
  --dataset.root=/data/rxhuang/three_cubes_1 \
  --dataset.revision=v0.1.0 \
  --dataset.streaming=false \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=three_cubes_giga_world_front_wrist \
  --policy.device=cuda \
  --policy.torch_dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.model_cache_dir=/home/rxhuang/Projects/models/giga-world-policy \
  --policy.giga_world_root=/home/rxhuang/Projects/giga-world-policy \
  --policy.use_transformer_lora=true \
  --policy.lora_rank=64 \
  --policy.lora_alpha=64 \
  --policy.lora_target_modules='["to_q","to_k","to_v"]' \
  --policy.freeze_transformer_backbone=true \
  --policy.train_action_heads=true \
  --policy.reinit_action_heads=true \
  --policy.action_loss_weight=1.0 \
  --policy.visual_loss_weight=0.1 \
  --policy.crop_mode=center \
  --policy.per_view_size='[256,192]' \
  --policy.chunk_size=48 \
  --policy.n_action_steps=16 \
  --policy.optimizer_lr=1e-4 \
  --policy.scheduler_decay_lr=1e-5 \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --log_freq="${LOG_FREQ}" \
  --save_checkpoint=true \
  --save_freq="${SAVE_FREQ}"
