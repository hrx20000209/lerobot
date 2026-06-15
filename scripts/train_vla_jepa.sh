#!/usr/bin/env bash

set -euo pipefail

GPU_ID="${GPU_ID:-0}"
HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LOG_FREQ="${LOG_FREQ:-200}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
REPEATED_DIFFUSION_STEPS="${REPEATED_DIFFUSION_STEPS:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-output_lerobot_train/three_cubes/vla_jepa}"

if [[ "${CONDA_DEFAULT_ENV:-}" == "lerobot" ]]; then
  TRAIN_CMD=(lerobot-train)
else
  TRAIN_CMD=(conda run --no-capture-output -n lerobot lerobot-train)
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET}" "${TRAIN_CMD[@]}" \
  --policy.path=lerobot/VLA-JEPA-Pretrain \
  --policy.device=cuda \
  --policy.torch_dtype=bfloat16 \
  --policy.freeze_qwen=true \
  --policy.enable_world_model=false \
  --policy.reinit_modules='["model.action_model.action_encoder", "model.action_model.action_decoder", "model.action_model.state_encoder"]' \
  --policy.gripper_dim=5 \
  --policy.repeated_diffusion_steps="${REPEATED_DIFFUSION_STEPS}" \
  --policy.push_to_hub=false \
  --dataset.repo_id=hrx2000/Three_Cubes_1 \
  --dataset.root=/data/rxhuang/three_cubes_1 \
  --dataset.revision=v0.1.0 \
  --dataset.streaming=false \
  --rename_map='{"observation.images.front": "observation.images.exterior_1_left", "observation.images.right": "observation.images.exterior_2_left"}' \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=three_cubes_vla_jepa \
  --wandb.enable=false \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --num_workers=4 \
  --log_freq="${LOG_FREQ}" \
  --save_checkpoint="${SAVE_CHECKPOINT}" \
  --save_freq="${SAVE_FREQ}"
