#!/usr/bin/env bash
set -euo pipefail

# Fully asynchronous continuous inference robot client.
#
# Robot/policy arguments intentionally mirror the original LeRobot async client
# scripts: --robot.type, --robot.port, --robot.id, --robot.cameras,
# --policy_type, --pretrained_name_or_path, --actions_per_chunk, and --fps.
# Motor commands are disabled by default via shadow mode.

REPO_DIR="${REPO_DIR:-/home/hrx/Projects/lerobot}"
PYTHON_BIN="${PYTHON_BIN:-/home/hrx/miniconda3/envs/lerobot/bin/python}"

SERVER_ADDRESS="${SERVER_ADDRESS:-127.0.0.1:8080}"
ROBOT_TYPE="${ROBOT_TYPE:-so101_follower}"
ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM1}"
ROBOT_ID="${ROBOT_ID:-follower_arm}"

CAMERA_WIDTH="${CAMERA_WIDTH:-640}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-480}"
FPS="${FPS:-30}"
CONTINUOUS_OBS_FPS="${CONTINUOUS_OBS_FPS:-${FPS}}"

# Defaults match the current Three Cubes VLA-JEPA async script. Override these
# env vars for GigaWorld or another checkpoint.
EXTERIOR_1_LEFT_CAMERA="${EXTERIOR_1_LEFT_CAMERA:-4}"
EXTERIOR_2_LEFT_CAMERA="${EXTERIOR_2_LEFT_CAMERA:-2}"
FOURCC_1="${FOURCC_1:-MJPG}"
FOURCC_2="${FOURCC_2:-YUYV}"
if [[ -z "${ROBOT_CAMERAS:-}" ]]; then
  ROBOT_CAMERAS="{ exterior_1_left: {type: opencv, index_or_path: ${EXTERIOR_1_LEFT_CAMERA}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${FPS}, fourcc: \"${FOURCC_1}\"}, exterior_2_left: {type: opencv, index_or_path: ${EXTERIOR_2_LEFT_CAMERA}, width: ${CAMERA_WIDTH}, height: ${CAMERA_HEIGHT}, fps: ${FPS}, fourcc: \"${FOURCC_2}\"} }"
fi

POLICY_TYPE="${POLICY_TYPE:-vla_jepa}"
PRETRAINED_NAME_OR_PATH="${PRETRAINED_NAME_OR_PATH:-/home/hrx/Projects/models/three_cubes_1/vla_jepa_lora/}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
CLIENT_DEVICE="${CLIENT_DEVICE:-cpu}"
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-}"

TASK="${TASK:-go to red cube. take the red cube. go to box. put the red cube in box.}"
DISPLAY_DATA="${DISPLAY_DATA:-true}"
RECORD_TIMELINE="${RECORD_TIMELINE:-true}"
TIMELINE_SAVE_IMAGES="${TIMELINE_SAVE_IMAGES:-key}"

AGGREGATION_FN="${AGGREGATION_FN:-latency_aligned_blend}"
MAX_PENDING_OBSERVATIONS="${MAX_PENDING_OBSERVATIONS:-1}"
STALE_INFERENCE_MAX_AGE="${STALE_INFERENCE_MAX_AGE:-2.0}"
MIN_USABLE_ACTIONS="${MIN_USABLE_ACTIONS:-5}"
BLEND_HORIZON="${BLEND_HORIZON:-5}"
BLEND_ALPHA="${BLEND_ALPHA:-0.5}"
MAX_JOINT_DELTA="${MAX_JOINT_DELTA:-}"
MAX_GRIPPER_DELTA="${MAX_GRIPPER_DELTA:-}"
MAX_JOINT_DELTA_PER_STEP="${MAX_JOINT_DELTA_PER_STEP:-}"
MAX_JOINT_ABS_RANGE="${MAX_JOINT_ABS_RANGE:-}"
MAX_GRIPPER_DELTA_PER_STEP="${MAX_GRIPPER_DELTA_PER_STEP:-}"
MAX_CONTROL_STEPS="${MAX_CONTROL_STEPS:-}"
EMERGENCY_STOP="${EMERGENCY_STOP:-false}"

SHADOW_MODE="${SHADOW_MODE:-true}"
ENABLE_ROBOT_EXECUTION="${ENABLE_ROBOT_EXECUTION:-false}"

TIMELINE_LOG_PATH="${TIMELINE_LOG_PATH:-${REPO_DIR}/outputs/continuous_async/client_timeline.jsonl}"
TIMELINE_PLOT_PATH="${TIMELINE_PLOT_PATH:-${REPO_DIR}/outputs/continuous_async/timeline.png}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found or not executable: ${PYTHON_BIN}" >&2
  exit 2
fi

if [[ ! -f "${PRETRAINED_NAME_OR_PATH}/config.json" ]]; then
  echo "Missing policy checkpoint config: ${PRETRAINED_NAME_OR_PATH}/config.json" >&2
  exit 2
fi

if [[ -z "${ACTIONS_PER_CHUNK}" ]]; then
  ACTIONS_PER_CHUNK="$("${PYTHON_BIN}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("chunk_size", 50))' "${PRETRAINED_NAME_OR_PATH}/config.json")"
fi

mkdir -p "$(dirname "${TIMELINE_LOG_PATH}")" "$(dirname "${TIMELINE_PLOT_PATH}")"
cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}/src:${PYTHONPATH:-}"

ROBOT_SAFETY_ARGS=()
if [[ -n "${MAX_RELATIVE_TARGET:-}" ]]; then
  ROBOT_SAFETY_ARGS+=(--robot.max_relative_target="${MAX_RELATIVE_TARGET}")
fi

CONTINUOUS_SAFETY_ARGS=()
if [[ -n "${MAX_JOINT_DELTA}" ]]; then
  CONTINUOUS_SAFETY_ARGS+=(--max_joint_delta="${MAX_JOINT_DELTA}")
fi
if [[ -n "${MAX_GRIPPER_DELTA}" ]]; then
  CONTINUOUS_SAFETY_ARGS+=(--max_gripper_delta="${MAX_GRIPPER_DELTA}")
fi
if [[ -n "${MAX_JOINT_DELTA_PER_STEP}" ]]; then
  CONTINUOUS_SAFETY_ARGS+=(--max_joint_delta_per_step="${MAX_JOINT_DELTA_PER_STEP}")
fi
if [[ -n "${MAX_JOINT_ABS_RANGE}" ]]; then
  CONTINUOUS_SAFETY_ARGS+=(--max_joint_abs_range="${MAX_JOINT_ABS_RANGE}")
fi
if [[ -n "${MAX_GRIPPER_DELTA_PER_STEP}" ]]; then
  CONTINUOUS_SAFETY_ARGS+=(--max_gripper_delta_per_step="${MAX_GRIPPER_DELTA_PER_STEP}")
fi
if [[ -n "${MAX_CONTROL_STEPS}" ]]; then
  CONTINUOUS_SAFETY_ARGS+=(--max_control_steps="${MAX_CONTROL_STEPS}")
fi

exec "${PYTHON_BIN}" -m lerobot.async_inference.continuous_robot_client \
  --async_mode=continuous \
  --server_address="${SERVER_ADDRESS}" \
  --robot.type="${ROBOT_TYPE}" \
  --robot.port="${ROBOT_PORT}" \
  --robot.id="${ROBOT_ID}" \
  "${ROBOT_SAFETY_ARGS[@]}" \
  --robot.cameras="${ROBOT_CAMERAS}" \
  --task="${TASK}" \
  --policy_type="${POLICY_TYPE}" \
  --pretrained_name_or_path="${PRETRAINED_NAME_OR_PATH}" \
  --policy_device="${POLICY_DEVICE}" \
  --client_device="${CLIENT_DEVICE}" \
  --actions_per_chunk="${ACTIONS_PER_CHUNK}" \
  --fps="${FPS}" \
  --continuous_obs_fps="${CONTINUOUS_OBS_FPS}" \
  --aggregation_fn="${AGGREGATION_FN}" \
  --max_pending_observations="${MAX_PENDING_OBSERVATIONS}" \
  --stale_inference_max_age="${STALE_INFERENCE_MAX_AGE}" \
  --min_usable_actions="${MIN_USABLE_ACTIONS}" \
  --blend_horizon="${BLEND_HORIZON}" \
  --blend_alpha="${BLEND_ALPHA}" \
  "${CONTINUOUS_SAFETY_ARGS[@]}" \
  --emergency_stop="${EMERGENCY_STOP}" \
  --shadow_mode="${SHADOW_MODE}" \
  --enable_robot_execution="${ENABLE_ROBOT_EXECUTION}" \
  --record_timeline="${RECORD_TIMELINE}" \
  --timeline_log_path="${TIMELINE_LOG_PATH}" \
  --timeline_plot_path="${TIMELINE_PLOT_PATH}" \
  --timeline_save_images="${TIMELINE_SAVE_IMAGES}" \
  --display_data="${DISPLAY_DATA}"
