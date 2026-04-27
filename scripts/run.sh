lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_arm \
  --robot.cameras="{ base_0_rgb: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
  --dataset.repo_id=hrx2000/eval_pi0 \
  --dataset.single_task="Grab the blue cube" \
  --policy.type=hrx2000/Lerobot_SmolVLA_Blue_Block \
  --teleop.type=so101_leader \
  --dataset.num_episodes=1 \
  --dataset.fps=30 \
  --dataset.episode_time_s=600 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --teleop.port=/dev/ttyACM1 \
  --teleop.id=leader_arm 
  # --rename_map='{"observation.images.front": "observation.images.camera1"}'


# remote
python -m lerobot.async_inference.robot_client \
  --server_address=100.127.53.101:8080 \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_so100 \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
  --task="Grab the blue block" \
  --policy_type=fast_wam \
  --pretrained_name_or_path=/home/rxhuang/Projects/lerobot/output_lerobot_train/grab_blue_block/fast_wam/checkpoints/last/pretrained_model \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average \
  --debug_visualize_queue_size=True