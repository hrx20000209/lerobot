# data collection
# 3 cube
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, right: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader_arm \
    --display_data=true \
    --dataset.repo_id=hrx2000/Three_Boxes_1 \
    --dataset.num_episodes=60 \
    --dataset.single_task="go to left dark box. take the red cube. go to middle light-colored box. put the red cube in box." \
    --dataset.push_to_hub=true \
    --dataset.episode_time_s=17 \
    --dataset.reset_time_s=2 \
    --dataset.root="/home/hrx/.cache/huggingface/lerobot/hrx2000/Three_Boxes_1" \
    --resume=true

#previous_task --task="go to red cube. take the red cube. go to box. put the red cube in box."

# Red Rectangle
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, right: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=leader_arm \
    --display_data=true \
    --dataset.repo_id=hrx2000/Move_Cube \
    --dataset.num_episodes=50 \
    --dataset.single_task="Go to cube in the right box. Take the cube. Go to the left box. Put the cube in the left box." \
    --dataset.push_to_hub=true \
    --dataset.episode_time_s=15 \
    --dataset.reset_time_s=3 



lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=follower_arm \
    --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, right: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}" \
    --display_data=false \
    --dataset.repo_id=hrx2000/eval_pi0 \
    --dataset.num_episodes=50 \
    --dataset.single_task="Pick up the blue cube into the box." \
    --dataset.push_to_hub=true \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=3 \
    --policy.type=pi05 \
    --policy.pretrained_path=output_lerobot_train/move_block/pi05/checkpoints/016000/pretrained_model
