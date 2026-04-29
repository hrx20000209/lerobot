lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, right: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
  --dataset.repo_id=hrx2000/eval_pi0 \
  --dataset.single_task="Grab the blue cube" \
  --policy.path=/home/hrx/Projects/models/smolvla/ \
  --policy.device=cuda \
  --teleop.type=so101_leader \
  --dataset.num_episodes=1 \
  --dataset.episode_time_s=600 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --teleop.port=/dev/ttyACM0 \
  --teleop.id=leader_arm 
  # --dataset.fps=4 
  # --dataset.rename_map='{"observation.images.wrist": "observation.images.right"}'


# remote
python -m lerobot.async_inference.robot_client \
  --server_address=100.127.53.101:8080 \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_so100 \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
  --task="Grab the blue block" \
  --policy_type=act \
  --pretrained_name_or_path=/home/hrx/Projects/models/pi05/ \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average \
  --debug_visualize_queue_size=True


# remote localhost
python -m lerobot.async_inference.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, right: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
  --task="Grab the blue block" \
  --policy_type=smolvla \
  --pretrained_name_or_path=/home/hrx/Projects/models/smolvla/ \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.5 \
  --aggregate_fn_name=weighted_average \
  --debug_visualize_queue_size=True 

python -m lerobot.async_inference.policy_server \
    --host=0.0.0.0 \
    --port=8080
    