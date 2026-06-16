#!/usr/bin/env bash

set -euo pipefail

GPU_ID="${GPU_ID:-0}"
STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-2}"
LOG_FREQ="${LOG_FREQ:-100}"
SAVE_FREQ="${SAVE_FREQ:-3000}"
SEED="${SEED:-1000}"
WORLD_MODEL_LOSS_WEIGHT="${WORLD_MODEL_LOSS_WEIGHT:-0.1}"
REPEATED_DIFFUSION_STEPS="${REPEATED_DIFFUSION_STEPS:-4}"
LORA_R="${LORA_R:-32}"
LORA_ALPHA="${LORA_ALPHA:-${LORA_R}}"
TARGET_MODE="${TARGET_MODE:-all}"
OUTPUT_DIR="${OUTPUT_DIR:-output_lerobot_train/three_cubes/vla_jepa_lora_${TARGET_MODE}_r${LORA_R}_w01}"

case "${TARGET_MODE}" in
  qv)
    TARGET_MODULES='["q_proj","v_proj"]'
    ;;
  all)
    TARGET_MODULES='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]'
    ;;
  *)
    echo "Unknown TARGET_MODE=${TARGET_MODE}; expected qv or all" >&2
    exit 2
    ;;
esac

if [[ "${CONDA_DEFAULT_ENV:-}" == "lerobot" ]]; then
  TRAIN_CMD=(lerobot-train)
else
  TRAIN_CMD=(conda run --no-capture-output -n lerobot lerobot-train)
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" \
HF_HUB_DISABLE_XET=1 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
"${TRAIN_CMD[@]}" \
  --policy.path=lerobot/VLA-JEPA-Pretrain \
  --policy.device=cuda \
  --policy.torch_dtype=bfloat16 \
  --policy.freeze_qwen=false \
  --policy.enable_world_model=true \
  --policy.world_model_loss_weight="${WORLD_MODEL_LOSS_WEIGHT}" \
  --policy.reinit_modules='["model.action_model.action_encoder", "model.action_model.action_decoder", "model.action_model.state_encoder"]' \
  --policy.gripper_dim=5 \
  --policy.repeated_diffusion_steps="${REPEATED_DIFFUSION_STEPS}" \
  --policy.push_to_hub=false \
  --peft.method_type=LORA \
  --peft.target_modules="${TARGET_MODULES}" \
  --peft.full_training_modules='["action_model","video_predictor"]' \
  --peft.r="${LORA_R}" \
  --peft.lora_alpha="${LORA_ALPHA}" \
  --dataset.repo_id=hrx2000/Three_Cubes_1 \
  --dataset.root=/data/rxhuang/three_cubes_1 \
  --dataset.revision=v0.1.0 \
  --dataset.streaming=false \
  --rename_map='{"observation.images.front": "observation.images.exterior_1_left", "observation.images.wrist": "observation.images.exterior_2_left"}' \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="three_cubes_vla_jepa_${TARGET_MODE}_r${LORA_R}_wm${WORLD_MODEL_LOSS_WEIGHT}" \
  --seed="${SEED}" \
  --wandb.enable=false \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --num_workers=4 \
  --log_freq="${LOG_FREQ}" \
  --save_checkpoint=true \
  --save_freq="${SAVE_FREQ}"
