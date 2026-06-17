lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, right: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
  --dataset.repo_id=hrx2000/eval_pi0 \
  --dataset.single_task="Go to the cube. Take the cube. Go to the box. Put the cube in the box." \
  --policy.path=/home/hrx/Projects/models/blue_cube/act/ \
  --policy.device=cuda \
  --dataset.episode_time_s=600 \
  --dataset.push_to_hub=false \
  --display_data=true \
  --dataset.fps=10 
  # --dataset.rename_map='{"observation.images.wrist": "observation.images.right"}'


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
  --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, right: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
  --task="go to red cube. take the red cube. go to box. put the red cube in box." \
  --policy_type=pi05 \
  --pretrained_name_or_path=/home/hrx/Projects/models/three_cubes_1/pi05_lora_expert/ \
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
  --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, right: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
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
  --robot.port=/dev/ttyACM0 \
  --robot.id=follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
  --task="Pick up the blue cube into the box." \
  --policy_type=smolvla \
  --pretrained_name_or_path=/home/hrx/Projects/models/smolvla/ \
  --policy_device=cuda \
  --actions_per_chunk=50 \
  --chunk_size_threshold=0.2 \
  --aggregate_fn_name=weighted_average \
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
