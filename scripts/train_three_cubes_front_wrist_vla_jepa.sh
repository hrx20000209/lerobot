#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-6}"
STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SAVE_FREQ="${SAVE_FREQ:-3000}"
LOG_FREQ="${LOG_FREQ:-100}"
NUM_WORKERS="${NUM_WORKERS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-output_lerobot_train/three_cubes/vla_jepa_front_wrist_qv_r64_rep4_w01}"
LEROBOT_TRAIN="${LEROBOT_TRAIN:-/home/rxhuang/anaconda3/envs/lerobot/bin/lerobot-train}"

INPUT_FEATURES='{
  "observation.state": {"type": "STATE", "shape": [6]},
  "observation.images.exterior_1_left": {"type": "VISUAL", "shape": [3, 480, 640]},
  "observation.images.exterior_2_left": {"type": "VISUAL", "shape": [3, 480, 640]}
}'

export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HOME="${HF_HOME:-/data/hf_cache}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

exec "${LEROBOT_TRAIN}" \
  --policy.path=lerobot/VLA-JEPA-Pretrain \
  --policy.input_features="${INPUT_FEATURES}" \
  --policy.device=cuda \
  --policy.torch_dtype=bfloat16 \
  --policy.freeze_qwen=false \
  --policy.enable_world_model=true \
  --policy.world_model_loss_weight=0.1 \
  --policy.reinit_modules='["model.action_model.action_encoder", "model.action_model.action_decoder", "model.action_model.state_encoder"]' \
  --policy.gripper_dim=5 \
  --policy.pre_snap_gripper_action=false \
  --policy.binarize_gripper_action=false \
  --policy.repeated_diffusion_steps=4 \
  --policy.push_to_hub=false \
  --peft.method_type=LORA \
  --peft.target_modules='["q_proj","v_proj"]' \
  --peft.full_training_modules='["action_model","video_predictor"]' \
  --peft.r=64 \
  --peft.lora_alpha=64 \
  --dataset.repo_id=hrx2000/Three_Cubes_1 \
  --dataset.root=/data/rxhuang/three_cubes_1 \
  --dataset.revision=v0.1.0 \
  --dataset.streaming=false \
  --rename_map='{"observation.images.front": "observation.images.exterior_1_left", "observation.images.wrist": "observation.images.exterior_2_left"}' \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=three_cubes_vla_jepa_front_wrist_qv_r64 \
  --seed=1000 \
  --wandb.enable=false \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --log_freq="${LOG_FREQ}" \
  --save_checkpoint=true \
  --save_freq="${SAVE_FREQ}"
