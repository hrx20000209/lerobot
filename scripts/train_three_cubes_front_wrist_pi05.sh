#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-1}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-24}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
LOG_FREQ="${LOG_FREQ:-20}"
NUM_WORKERS="${NUM_WORKERS:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-output_lerobot_train/three_cubes/pi05_front_wrist_r64_full_expert_pad01_b24}"
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
  --policy.type=pi05 \
  --policy.pretrained_path=lerobot/pi05_base \
  --policy.input_features="${INPUT_FEATURES}" \
  --dataset.repo_id=hrx2000/Three_Cubes_1 \
  --dataset.root=/data/rxhuang/three_cubes_1 \
  --dataset.revision=v0.1.0 \
  --dataset.streaming=false \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=three_cubes_pi05_front_wrist \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.n_action_steps=10 \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.padded_action_loss_weight=0.1 \
  --policy.optimizer_lr=1e-4 \
  --policy.scheduler_decay_lr=1e-5 \
  --policy.push_to_hub=false \
  --wandb.enable=false \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --log_freq="${LOG_FREQ}" \
  --save_checkpoint=true \
  --save_freq="${SAVE_FREQ}" \
  --peft.method_type=LORA \
  --peft.r=64 \
  --peft.lora_alpha=64 \
  '--peft.target_modules=(model\.paligemma_with_expert\.paligemma\..*\.self_attn\.(q|v)_proj)' \
  '--peft.full_training_modules=["gemma_expert","action_in_proj","action_out_proj","time_mlp_in","time_mlp_out"]'
