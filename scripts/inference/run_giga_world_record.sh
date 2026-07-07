#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/hrx/Projects/lerobot}"
PYTHON_BIN="${PYTHON_BIN:-/home/hrx/miniconda3/envs/lerobot/bin/python}"
PRETRAINED_NAME_OR_PATH="${PRETRAINED_NAME_OR_PATH:-/home/hrx/Projects/models/three_cubes_1/giga_world_2}"
DATASET_ROOT="${DATASET_ROOT:-/home/hrx/Datasets/eval_giga_world}"
DATASET_REPO_ID="${DATASET_REPO_ID:-hrx2000/eval_giga_world_thor}"
NUM_EPISODES="${NUM_EPISODES:-1}"
FPS="${FPS:-30}"

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
export GIGA_WORLD_POLICY_ROOT="${GIGA_WORLD_POLICY_ROOT:-/home/hrx/Projects/giga-world-policy}"
export GIGA_WORLD_MODEL_CACHE_DIR="${GIGA_WORLD_MODEL_CACHE_DIR:-/home/hrx/Projects/models/giga-world-policy}"
export HF_HOME="${HF_HOME:-/home/hrx/.cache/huggingface}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -x "${PYTHON_BIN}" || ! -f "${PRETRAINED_NAME_OR_PATH}/config.json" ]]; then
  echo "Missing Python environment or Giga World checkpoint." >&2
  exit 2
fi

mkdir -p "$(dirname "${DATASET_ROOT}")"
cd "${REPO_DIR}"

INFERENCE_TYPE="${INFERENCE_TYPE:-rtc}"
INFERENCE_ARGS=(--inference.type="${INFERENCE_TYPE}")
if [[ "${INFERENCE_TYPE}" == "rtc" ]]; then
  INFERENCE_ARGS+=(
    --inference.rtc.enabled="${RTC_ENABLED:-true}"
    --inference.rtc.execution_horizon="${EXECUTION_HORIZON:-16}"
    --inference.queue_threshold="${QUEUE_THRESHOLD:-30}"
  )
fi

ROBOT_SAFETY_ARGS=()
if [[ -n "${MAX_RELATIVE_TARGET:-}" ]]; then
  ROBOT_SAFETY_ARGS+=(--robot.max_relative_target="${MAX_RELATIVE_TARGET}")
fi

# Policy-driven episodic rollout. This replaces lerobot-record, which is now
# teleoperation-only and rejects policy deployment.
exec "${PYTHON_BIN}" -m lerobot.scripts.lerobot_rollout \
  --strategy.type=episodic \
  "${INFERENCE_ARGS[@]}" \
  --policy.path="${PRETRAINED_NAME_OR_PATH}" \
  --policy.n_action_steps="${N_ACTION_STEPS:-48}" \
  --policy.num_inference_steps="${NUM_INFERENCE_STEPS:-5}" \
  --device=cuda \
  --robot.type=so101_follower \
  --robot.port="${ROBOT_PORT:-/dev/ttyACM1}" \
  --robot.id="${ROBOT_ID:-follower_arm}" \
  "${ROBOT_SAFETY_ARGS[@]}" \
  --robot.cameras="{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: ${FPS}, fourcc: \"MJPG\"}, wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: ${FPS}, fourcc: \"YUYV\"}}" \
  --task="${TASK:-go to red cube. take the red cube. go to box. put the red cube in box.}" \
  --fps="${FPS}" \
  --display_data="${DISPLAY_DATA:-true}" \
  --play_sounds="${PLAY_SOUNDS:-false}" \
  --return_to_initial_position="${RETURN_TO_INITIAL_POSITION:-true}" \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --dataset.num_episodes="${NUM_EPISODES}" \
  --dataset.single_task="${TASK:-go to red cube. take the red cube. go to box. put the red cube in box.}" \
  --dataset.episode_time_s="${EPISODE_TIME_S:-600}" \
  --dataset.reset_time_s="${RESET_TIME_S:-2}" \
  --dataset.push_to_hub="${PUSH_TO_HUB:-false}" \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads="${ENCODER_THREADS:-2}" \
  --resume="${RESUME:-false}"
