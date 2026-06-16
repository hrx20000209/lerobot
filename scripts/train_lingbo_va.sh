#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-12}"
STEPS="${STEPS:-20000}"
LOG_FREQ="${LOG_FREQ:-20}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
TRAIN_VIDEO_HEAD="${TRAIN_VIDEO_HEAD:-true}"
USE_TRANSFORMER_LORA="${USE_TRANSFORMER_LORA:-false}"
LORA_RANK="${LORA_RANK:-8}"
LORA_ALPHA="${LORA_ALPHA:-16}"
LORA_LR="${LORA_LR:-1e-4}"
OUTPUT_DIR="${OUTPUT_DIR:-output_lerobot_train/three_cubes/lingbo_va}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

lerobot-train \
  --policy.type=lingbo_va \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.va_config_name=so101 \
  --policy.camera_keys='["observation.images.front", "observation.images.right"]' \
  --policy.server_camera_keys='["observation.images.front", "observation.images.right"]' \
  --policy.used_action_channel_ids='[0, 1, 2, 3, 4, 5]' \
  --policy.num_frames=25 \
  --policy.chunk_size=32 \
  --policy.frame_chunk_size=4 \
  --policy.action_per_frame=8 \
  --policy.video_frame_stride=2 \
  --policy.train_action_head_only=true \
  --policy.train_video_head="${TRAIN_VIDEO_HEAD}" \
  --policy.use_transformer_lora="${USE_TRANSFORMER_LORA}" \
  --policy.lora_rank="${LORA_RANK}" \
  --policy.lora_alpha="${LORA_ALPHA}" \
  --policy.lora_optimizer_lr="${LORA_LR}" \
  --policy.lora_target_modules='["to_q", "to_v"]' \
  --policy.freeze_vision_text_encoder=true \
  --policy.gradient_checkpointing=true \
  --policy.loss_log_freq="${LOG_FREQ}" \
  --policy.vae_device=cuda \
  --policy.vae_encode_batch_size=8 \
  --policy.text_encoder_device=cpu \
  --policy.transformer_device=cuda \
  --policy.offload_vae_after_encode=true \
  --policy.offload_text_encoder_after_encode=true \
  --policy.push_to_hub=false \
  --dataset.repo_id=hrx2000/Three_Cubes_1 \
  --dataset.root=/data/rxhuang/three_cubes_1 \
  --dataset.revision=v0.1.0 \
  --dataset.streaming=false \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=three_cubes_lingbo_va \
  --wandb.enable=false \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --num_workers=2 \
  --log_freq="${LOG_FREQ}" \
  --save_checkpoint="${SAVE_CHECKPOINT}" \
  --save_freq="${SAVE_FREQ}"
