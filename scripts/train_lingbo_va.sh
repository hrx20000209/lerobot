#!/usr/bin/env bash
# Train LingBoVA on the SO101 three_cubes dataset.
#
# Usage:
#   bash scripts/train_lingbo_va.sh
#   TRAIN_MODE=full GPU_ID=1 bash scripts/train_lingbo_va.sh
#   bash scripts/train_lingbo_va.sh --help
#
# TRAIN_MODE selects the freeze strategy (env var, default "lora"):
#   lora            (recommended) LoRA on the transformer (rank=64, alpha=128, targets
#                   to_q/to_k/to_v/to_out.0/ffn.net.0.proj/ffn.net.2 in every block) + the
#                   action/video-head modules fully trainable at a separate (lower) lr.
#                   ~1-5% of transformer parameters trainable; fits a single 24GB GPU.
#   full            Full fine-tuning of the entire transformer (upstream's `train_mode="full"`
#                   default, lr=1e-5). UNTESTED on a single 24GB GPU: this repo's own prior
#                   direct-upstream experiment needed 2x 24GB GPUs just for a last-2-blocks
#                   partial fine-tune (see ~/Projects/models/lingbot-va-three-cubes/action_last2/
#                   training_config.json, world_size=2). Expect to need multi-GPU or a bigger
#                   card; gradient checkpointing is enabled but 8-bit AdamW is NOT wired in
#                   (plain AdamW is used) since that would need a new OptimizerConfig subclass.
#   action_head_only  Legacy/debug mode reproducing the ORIGINAL (buggy) default: only the
#                   action/video-head I/O modules train, none of the 30 transformer blocks ever
#                   unfreeze. Kept only for A/B comparison against `lora`/`full` — this is the
#                   configuration that produced the reported mean-collapse bug and should not be
#                   used for a real training run.
#
# Prerequisite: compute train-only-scoped action normalization stats BEFORE training (episodes
# 95-99 must never leak into normalization, matching the train/val split below):
#   python scripts/compute_lingbo_va_train_stats.py \
#     --repo-id hrx2000/Three_Cubes_1 --root /data/rxhuang/three_cubes_1 --revision v0.1.0 \
#     --train-episodes 0-94 \
#     --output /data/rxhuang/three_cubes_1/lingbo_va_train_action_stats_ep0-94.json
#
# Notes on attn_mode: training always requires train_attn_mode=flex (enforced by an assertion in
# LingBoVAPolicy.forward); torch/flashattn silently disable the model's causal training mask.
# Inference (serve_lingbo_va.py / deploy_so101_lingbo_va.py / eval_lingbo_va_offline.py) must use
# inference_attn_mode=torch or flashattn — flex is inference-incompatible (see docs in
# scripts/serve_lingbo_va.py). Every checkpoint saved by this script bakes attn_mode=flex into
# transformer/config.json; inference code force-overrides it at load time, so this is handled
# automatically and requires no manual config.json editing.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'
  exit 0
fi

GPU_ID="${GPU_ID:-0}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-8}"   # effective batch = BATCH_SIZE * this
STEPS="${STEPS:-20000}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
LOG_FREQ="${LOG_FREQ:-20}"
SAVE_FREQ="${SAVE_FREQ:-2000}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-lingbo_va_three_cubes}"

TRAIN_MODE="${TRAIN_MODE:-lora}"   # lora | full | action_head_only
TRAIN_VIDEO_HEAD="${TRAIN_VIDEO_HEAD:-true}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_LR="${LORA_LR:-1e-4}"
LORA_TARGET_MODULES='["to_q", "to_k", "to_v", "to_out.0", "ffn.net.0.proj", "ffn.net.2"]'
HEAD_LR="${HEAD_LR:-1e-5}"

CFG_PROB="${CFG_PROB:-0.1}"
VIDEO_LOSS_WEIGHT="${VIDEO_LOSS_WEIGHT:-0.1}"
ACTION_LOSS_WEIGHT="${ACTION_LOSS_WEIGHT:-1.0}"

# Memory levers if you hit CUDA OOM: lower BATCH_SIZE first (raise GRAD_ACCUMULATION_STEPS to
# compensate, effective batch = BATCH_SIZE * GRAD_ACCUMULATION_STEPS), then VAE_ENCODE_BATCH_SIZE
# (how many of the batch_size*num_cameras video clips get VAE-encoded per chunk; lower = less
# peak VAE memory, slower encode).
VAE_ENCODE_BATCH_SIZE="${VAE_ENCODE_BATCH_SIZE:-8}"

DATASET_REPO_ID="${DATASET_REPO_ID:-hrx2000/Three_Cubes_1}"
DATASET_ROOT="${DATASET_ROOT:-/data/rxhuang/three_cubes_1}"
DATASET_REVISION="${DATASET_REVISION:-v0.1.0}"
ACTION_STATS_PATH="${ACTION_STATS_PATH:-${DATASET_ROOT}/lingbo_va_train_action_stats_ep0-94.json}"

OUTPUT_DIR="${OUTPUT_DIR:-output_lerobot_train/three_cubes/lingbo_va}"

if [[ ! -f "${ACTION_STATS_PATH}" ]]; then
  echo "ERROR: train-only action stats file not found: ${ACTION_STATS_PATH}" >&2
  echo "Run scripts/compute_lingbo_va_train_stats.py first (see header of this script)." >&2
  exit 1
fi

# Episodes 0-94 train; 95-99 are held out for the training-time validation callback (see
# LingBoVAConfig.val_episodes / eval_utils.py) and must never appear in --dataset.episodes.
TRAIN_EPISODES="$(python3 -c 'print(list(range(95)))')"
VAL_EPISODES="[95, 96, 97, 98, 99]"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export PYTORCH_ALLOC_CONF="expandable_segments:True"

FREEZE_FLAGS=()
case "${TRAIN_MODE}" in
  lora)
    FREEZE_FLAGS+=(
      "--policy.train_action_head_only=false"
      "--policy.use_transformer_lora=true"
      "--policy.lora_rank=${LORA_RANK}"
      "--policy.lora_alpha=${LORA_ALPHA}"
      "--policy.lora_optimizer_lr=${LORA_LR}"
      "--policy.lora_target_modules=${LORA_TARGET_MODULES}"
      "--policy.optimizer_lr=${HEAD_LR}"
      "--policy.gradient_checkpointing=true"
    )
    ;;
  full)
    echo "WARNING: TRAIN_MODE=full is untested on a single 24GB GPU (see script header)." >&2
    FREEZE_FLAGS+=(
      "--policy.train_action_head_only=false"
      "--policy.use_transformer_lora=false"
      "--policy.optimizer_lr=${HEAD_LR}"
      "--policy.gradient_checkpointing=true"
    )
    ;;
  action_head_only)
    echo "WARNING: TRAIN_MODE=action_head_only never unfreezes any transformer block (legacy/debug only)." >&2
    FREEZE_FLAGS+=(
      "--policy.train_action_head_only=true"
      "--policy.use_transformer_lora=false"
      "--policy.optimizer_lr=${HEAD_LR}"
    )
    ;;
  *)
    echo "Unknown TRAIN_MODE=${TRAIN_MODE} (expected lora|full|action_head_only)" >&2
    exit 1
    ;;
esac

lerobot-train \
  --policy.type=lingbo_va \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.va_config_name=so101 \
  --policy.camera_keys='["observation.images.front", "observation.images.right", "observation.images.wrist"]' \
  --policy.server_camera_keys='["observation.images.front", "observation.images.right", "observation.images.wrist"]' \
  --policy.used_action_channel_ids='[0, 1, 2, 3, 4, 28]' \
  --policy.num_frames=25 \
  --policy.chunk_size=32 \
  --policy.frame_chunk_size=4 \
  --policy.action_per_frame=8 \
  --policy.video_frame_stride=2 \
  --policy.train_video_head="${TRAIN_VIDEO_HEAD}" \
  "${FREEZE_FLAGS[@]}" \
  --policy.train_attn_mode=flex \
  --policy.inference_attn_mode=torch \
  --policy.cfg_prob="${CFG_PROB}" \
  --policy.video_loss_weight="${VIDEO_LOSS_WEIGHT}" \
  --policy.action_loss_weight="${ACTION_LOSS_WEIGHT}" \
  --policy.scheduler_name=cosine \
  --policy.scheduler_warmup_steps="${WARMUP_STEPS}" \
  --policy.action_stats_path="${ACTION_STATS_PATH}" \
  --policy.val_episodes="${VAL_EPISODES}" \
  --policy.val_num_samples=8 \
  --policy.freeze_vision_text_encoder=true \
  --policy.loss_log_freq="${LOG_FREQ}" \
  --policy.vae_device=cuda \
  --policy.vae_encode_batch_size="${VAE_ENCODE_BATCH_SIZE}" \
  --policy.text_encoder_device=cpu \
  --policy.transformer_device=cuda \
  --policy.offload_vae_after_encode=true \
  --policy.offload_text_encoder_after_encode=true \
  --policy.push_to_hub=false \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.revision="${DATASET_REVISION}" \
  --dataset.streaming=false \
  --dataset.episodes="${TRAIN_EPISODES}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name=three_cubes_lingbo_va \
  --wandb.enable="${WANDB_ENABLE}" \
  --wandb.project="${WANDB_PROJECT}" \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --grad_accumulation_steps="${GRAD_ACCUMULATION_STEPS}" \
  --num_workers=2 \
  --log_freq="${LOG_FREQ}" \
  --save_checkpoint="${SAVE_CHECKPOINT}" \
  --save_freq="${SAVE_FREQ}"
