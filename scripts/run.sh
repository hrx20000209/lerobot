lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1\
  --robot.id=follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, right: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
  --dataset.repo_id=hrx2000/eval_pi0 \
  --dataset.single_task="go to red cube. take the red cube. go to box. put the red cube in box." \
  --policy.path=/home/hrx/Projects/models/three_cubes_1/smolvla/ \
  --policy.device=cuda \
  --dataset.episode_time_s=600 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --dataset.fps=10 
  # --dataset.rename_map='{"observation.images.wrist": "observation.images.right"}'

lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=follower_arm \
  --robot.cameras='{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
  --dataset.repo_id=hrx2000/eval_pi0 \
  --dataset.single_task="go to red cube. take the red cube. go to box. put the red cube in box." \
  --policy.path=/home/hrx/Projects/models/three_cubes_1/vla_jepa_lora/ \
  --rename_map='{"observation.images.front": "observation.images.exterior_1_left", "observation.images.wrist": "observation.images.exterior_2_left"}' \
  --policy.device=cuda \
  --inference.type=rtc \
  --inference.rtc.enabled=true \
  --inference.queue_threshold=5 \
  --fps=15 \
  --interpolation_multiplier=2 \
  --dataset.episode_time_s=600 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --dataset.fps=30


# remote
python -m lerobot.async_inference.robot_client \
  --server_address=143.89.191.15:8080 \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=follower_arm \
  --robot.cameras='{ front: {type: opencv, index_or_path: /dev/v4l/by-path/platform-a80aa10000.usb-usb-0:4.2.4:1.0-video-index0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, wrist: {type:
    opencv, index_or_path: /dev/v4l/by-path/platform-a80aa10000.usb-usb-0:4.2.2:1.0-video-index0, width: 640, height: 480, fps: 30}, right: {type: opencv, index_or_path: /dev/v4l/by-path/platform-a80aa10000.usb-usb-0:4.2.1:1.0-video-index0, width: 640, height: 480,
    fps: 30, fourcc: "MJPG"}}' \
  --task="go to red cube. take the red cube. go to box. put the red cube in box." \
  --policy_type=pi05 \
  --pretrained_name_or_path=/home/hrx/Projects/models/three_cubes_1/pi05/ \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --chunk_size_threshold=1.0 \
  --aggregate_fn_name=conservative \
  --debug_visualize_queue_size=True 



# remote localhost
# front + right + wrist
python -m lerobot.async_inference.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, right: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
  --task="go to red cube. take the red cube. go to box. put the red cube in box." \
  --policy_type=smolvla \
  --pretrained_name_or_path=/home/hrx/Projects/models/three_cubes_1/smolvla/ \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --chunk_size_threshold=1.0 \
  --aggregate_fn_name=conservative \
  --debug_visualize_queue_size=True 


# front + right
python -m lerobot.async_inference.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, right: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
  --task="Pick up the blue cube into the box." \
  --policy_type=smolvla \
  --pretrained_name_or_path=/home/hrx/Projects/models/smolvla/ \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.2 \
  --aggregate_fn_name=weighted_average \
  --debug_visualize_queue_size=True 
  
# front + wrist
python -m lerobot.async_inference.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM1 \
  --robot.id=follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
  --task="go to red cube. take the red cube. go to box. put the red cube in box." \
  --policy_type=pi05 \
  --pretrained_name_or_path=/home/hrx/Projects/models/three_cubes_1/pi05_front_wrist/ \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.8 \
  --aggregate_fn_name=latest_only \
  --debug_visualize_queue_size=True 


python -m lerobot.async_inference.robot_client \
  --server_address=127.0.0.1:8080 \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
  --task="Pick up the blue cube into the box." \
  --policy_type=smolvla \
  --pretrained_name_or_path=/home/hrx/Projects/models/smolvla/ \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.8 \
  --aggregate_fn_name=weighted_average \
  --debug_visualize_queue_size=True 

python -m lerobot.async_inference.policy_server \
  --host=0.0.0.0 \
  --port=8080 \
  --fps=30 \
  --inference_latency=0.033 \
  --obs_queue_timeout=1
