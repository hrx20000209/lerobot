# data collection
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, right: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader_arm \
    --display_data=true \
    --dataset.repo_id=hrx2000/Move_Block \
    --dataset.num_episodes=50 \
    --dataset.single_task="Move the cube inside the box." \
    --dataset.push_to_hub=true \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=3


lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, right: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
    --display_data=false \
    --dataset.repo_id=hrx2000/eval_pi0 \
    --dataset.num_episodes=50 \
    --dataset.single_task="Move the cube inside the box." \
    --dataset.push_to_hub=true \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=3 \
    --policy.type=pi05 \
    --policy.pretrained_path=output_lerobot_train/move_block/pi05/checkpoints/016000/pretrained_model
