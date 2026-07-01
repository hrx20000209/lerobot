#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/rxhuang/anaconda3/envs/lerobot/bin/python}"
DATASET_ROOT="${DATASET_ROOT:-/data/rxhuang/three_cubes_1}"
DATASET_REPO_ID="${DATASET_REPO_ID:-hrx2000/Three_Cubes_1}"
NUM_EPISODES="${NUM_EPISODES:-100}"

exec "${PYTHON_BIN}" -m lerobot.scripts.lerobot_record \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT:-/dev/ttyACM0}" \
  --robot.id="${ROBOT_ID:-follower_arm}" \
  --robot.cameras="{ front: {type: opencv, index_or_path: ${FRONT_CAMERA_INDEX:-4}, width: 640, height: 480, fps: 30, fourcc: \"MJPG\"}, right: {type: opencv, index_or_path: ${RIGHT_CAMERA_INDEX:-0}, width: 640, height: 480, fps: 30, fourcc: \"MJPG\"}, wrist: {type: opencv, index_or_path: ${WRIST_CAMERA_INDEX:-2}, width: 640, height: 480, fps: 30, fourcc: \"MJPG\"}}" \
  --teleop.type=so101_leader \
  --teleop.port="${TELEOP_PORT:-/dev/ttyACM1}" \
  --teleop.id="${TELEOP_ID:-leader_arm}" \
  --display_data=true \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.single_task="${TASK:-go to red cube. take the red cube. go to box. put the red cube in box.}" \
  --dataset.episode_time_s="${EPISODE_TIME_S:-17}" \
  --dataset.reset_time_s="${RESET_TIME_S:-2}" \
  --dataset.push_to_hub="${PUSH_TO_HUB:-false}" \
  --resume="${RESUME:-true}"
