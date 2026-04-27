CUDA_VISIBLE_DEVICES=1 lerobot-train \
  --policy.type=act \
  --dataset.repo_id=hrx2000/lerobot_grab_blue_cube \
  --dataset.root=/data/hf_cache/hub/datasets--hrx2000--lerobot_grab_blue_cube/snapshots/2bec0c4eddf2c913509ba414f76c68fa0c918ab4 \
  --dataset.revision=v0.1.0 \
  --dataset.streaming=false \
  --output_dir=output_lerobot_train/grab_blue_block/act \
  --job_name=grab_blue_cube \
  --policy.device=cuda \
  --wandb.enable=false \
  --wandb.project=Lerobot_Blue_Cube \
  --policy.push_to_hub=false \
  --steps=10000 \
  --batch_size=16


# train pi05
lerobot-train \
  --policy.type=pi05 \
  --dataset.repo_id=hrx2000/lerobot_grab_blue_cube \
  --dataset.root=/data/hf_cache/hub/datasets--hrx2000--lerobot_grab_blue_cube/snapshots/2bec0c4eddf2c913509ba414f76c68fa0c918ab4 \
  --dataset.revision=v0.1.0 \
  --dataset.streaming=false \
  --output_dir=output_lerobot_train/grab_blue_block/pi05 \
  --job_name=grab_blue_cube \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --wandb.enable=false \
  --wandb.project=Lerobot_Blue_Cube \
  --policy.push_to_hub=false \
  --steps=30000 \
  --batch_size=1

  python -m lerobot.async_inference.policy_server \
     --host=0.0.0.0 \
     --port=8080