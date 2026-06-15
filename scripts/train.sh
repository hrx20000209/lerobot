



CUDA_VISIBLE_DEVICES=0 lerobot-train \
  --policy.type=act \
  --dataset.repo_id=hrx2000/Three_Cubes_1 \
  --dataset.root=/data/rxhuang/three_cubes_1/ \
  --dataset.revision=v0.1.0 \
  --dataset.streaming=false \
  --output_dir=output_lerobot_train/three_cubes/act \
  --job_name=cubes \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --steps=40000 \
  --batch_size=32 \
  --save_freq=4000 



# train pi05
CUDA_VISIBLE_DEVICES="0" lerobot-train \
  --policy.type=pi05 \
  --dataset.repo_id=hrx2000/Three_Cubes_1 \
  --dataset.root=/data/rxhuang/three_cubes_1 \
  --dataset.revision=v0.1.0 \
  --dataset.streaming=false \
  --output_dir=output_lerobot_train/three_cubes/pi05 \
  --job_name=three_cubes \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --wandb.enable=false \
  --policy.push_to_hub=false \
  --steps=40000 \
  --batch_size=16 \
  --save_freq=4000 


lerobot-train \
  --policy.type=fast_wam \
  --dataset.repo_id=hrx2000/lerobot_grab_blue_cube_2 \
  --dataset.root=/data/rxhuang/lerobot_grab_blue_cube \
  --dataset.revision=v0.1.0 \
  --dataset.streaming=false \
  --output_dir=output_lerobot_train/grab_blue_block/fast_wam \
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
  --save_freq=5000 \
  --batch_size=1
  
