#!/usr/bin/env bash
set -euo pipefail

# Modes:
#   expert_only: pretrained PI0.5 VLM is frozen; full action expert is trained.
#   vlm_lora_qv: LoRA adapts q/v attention in vision, language, and action expert.
#   vlm_lora_all: LoRA adapts all attention and MLP projections.
#   vlm_lora_full_expert: VLM q/v LoRA plus full action-expert fine-tuning.
MODE="${MODE:-vlm_lora}"
GPU="${GPU:-1}"
STEPS="${STEPS:-20000}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LEROBOT_TRAIN="${LEROBOT_TRAIN:-lerobot-train}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-${LORA_R}}"
PADDED_ACTION_LOSS_WEIGHT="${PADDED_ACTION_LOSS_WEIGHT:-1.0}"
OPTIMIZER_LR="${OPTIMIZER_LR:-1e-4}"
SCHEDULER_DECAY_LR="${SCHEDULER_DECAY_LR:-1e-5}"

export CUDA_VISIBLE_DEVICES="${GPU}"
export HF_HOME="${HF_HOME:-/data/hf_cache}"

common_args=(
  --policy.type=pi05
  --policy.pretrained_path=lerobot/pi05_base
  --dataset.repo_id=hrx2000/Three_Cubes_1
  --dataset.root=/data/rxhuang/three_cubes_1
  --dataset.revision=v0.1.0
  --dataset.streaming=false
  --policy.device=cuda
  --policy.dtype=bfloat16
  --policy.gradient_checkpointing=true
  --policy.n_action_steps=10
  --policy.push_to_hub=false
  --wandb.enable=false
  --steps="${STEPS}"
  --save_freq="${SAVE_FREQ}"
  --save_checkpoint="${SAVE_CHECKPOINT}"
  --num_workers="${NUM_WORKERS}"
  --log_freq=20
)

case "${MODE}" in
  expert_only)
    batch_size="${BATCH_SIZE:-16}"
    output_dir="${OUTPUT_DIR:-output_lerobot_train/three_cubes/pi05_pretrained_expert_only}"
    mode_args=(
      --policy.freeze_vision_encoder=true
      --policy.train_expert_only=true
    )
    ;;
  vlm_lora|vlm_lora_qv)
    batch_size="${BATCH_SIZE:-2}"
    output_dir="${OUTPUT_DIR:-output_lerobot_train/three_cubes/pi05_pretrained_vlm_lora_qv_r${LORA_R}_b${batch_size}}"
    mode_args=(
      --policy.freeze_vision_encoder=false
      --policy.train_expert_only=false
      --policy.padded_action_loss_weight="${PADDED_ACTION_LOSS_WEIGHT}"
      --policy.optimizer_lr="${OPTIMIZER_LR}"
      --policy.scheduler_decay_lr="${SCHEDULER_DECAY_LR}"
      --peft.method_type=LORA
      --peft.r="${LORA_R}"
      --peft.lora_alpha="${LORA_ALPHA}"
      '--peft.target_modules=(model\.paligemma_with_expert\.(paligemma|gemma_expert)\..*\.self_attn\.(q|v)_proj)'
      '--peft.full_training_modules=["action_in_proj","action_out_proj","time_mlp_in","time_mlp_out"]'
    )
    ;;
  vlm_lora_all)
    batch_size="${BATCH_SIZE:-4}"
    output_dir="${OUTPUT_DIR:-output_lerobot_train/three_cubes/pi05_pretrained_vlm_lora_all_r${LORA_R}_b${batch_size}}"
    mode_args=(
      --policy.freeze_vision_encoder=false
      --policy.train_expert_only=false
      --policy.padded_action_loss_weight="${PADDED_ACTION_LOSS_WEIGHT}"
      --policy.optimizer_lr="${OPTIMIZER_LR}"
      --policy.scheduler_decay_lr="${SCHEDULER_DECAY_LR}"
      --peft.method_type=LORA
      --peft.r="${LORA_R}"
      --peft.lora_alpha="${LORA_ALPHA}"
      '--peft.target_modules=(model\.paligemma_with_expert\.(paligemma|gemma_expert)\..*\.(self_attn\.(q_proj|k_proj|v_proj|o_proj|out_proj)|mlp\.(down_proj|gate_proj|up_proj|fc1|fc2)))'
      '--peft.full_training_modules=["action_in_proj","action_out_proj","time_mlp_in","time_mlp_out"]'
    )
    ;;
  vlm_lora_full_expert)
    batch_size="${BATCH_SIZE:-4}"
    output_dir="${OUTPUT_DIR:-output_lerobot_train/three_cubes/pi05_pretrained_vlm_lora_full_expert_r${LORA_R}_b${batch_size}}"
    mode_args=(
      --policy.freeze_vision_encoder=false
      --policy.train_expert_only=false
      --policy.padded_action_loss_weight="${PADDED_ACTION_LOSS_WEIGHT}"
      --policy.optimizer_lr="${OPTIMIZER_LR}"
      --policy.scheduler_decay_lr="${SCHEDULER_DECAY_LR}"
      --peft.method_type=LORA
      --peft.r="${LORA_R}"
      --peft.lora_alpha="${LORA_ALPHA}"
      '--peft.target_modules=(model\.paligemma_with_expert\.paligemma\..*\.self_attn\.(q|v)_proj)'
      '--peft.full_training_modules=["gemma_expert","action_in_proj","action_out_proj","time_mlp_in","time_mlp_out"]'
    )
    ;;
  *)
    echo "Unknown MODE=${MODE}" >&2
    exit 2
    ;;
esac

exec "${LEROBOT_TRAIN}" \
  "${common_args[@]}" \
  --output_dir="${output_dir}" \
  --job_name="three_cubes_pi05_${MODE}" \
  --batch_size="${batch_size}" \
  "${mode_args[@]}"
