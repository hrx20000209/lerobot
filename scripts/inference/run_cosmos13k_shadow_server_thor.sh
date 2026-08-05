#!/usr/bin/env bash
set -euo pipefail

# Cosmos Policy step-13000 SO101 policy server on Jetson Thor.
#
# Denoising steps default to 1 (the setting validated offline in
# models/three_cubes_1/cosmos_policy_step13000/evaluation/ensembled_episode006_stride5_denoise1).
#
# DRY_RUN=true  -> server returns the measured proprio (identity); the model
#                  action never crosses the gRPC boundary.  Use for the plumbing
#                  smoke test.
# DRY_RUN=false -> server returns the real (safety-clamped) model action.  Only
#                  ever pair this with the no-write shadow client, which cannot
#                  write to the motor bus.
#
# Either way the raw model candidate is logged as "SHADOW candidate" lines.

COSMOS_REPO="${COSMOS_REPO:-/home/hrx/Projects/cosmos-policy}"
LEROBOT_SRC="${LEROBOT_SRC:-/home/hrx/Projects/lerobot/src}"
PYTHON_BIN="${PYTHON_BIN:-/home/hrx/miniconda3/envs/cosmos/bin/python}"

# The Cosmos experiment package: holds configs/, processed_data/ (stats + T5).
TASK_ROOT="${TASK_ROOT:-/home/hrx/Projects/models/three_cubes_1/cosmos_policy}"
CKPT_PATH="${CKPT_PATH:-/home/hrx/Projects/models/three_cubes_1/cosmos_policy_step13000/model/model}"
DATASET_ROOT="${DATASET_ROOT:-/home/hrx/Datasets/three_cubes_1}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8082}"
FPS="${FPS:-8}"
NUM_DENOISING_STEPS="${NUM_DENOISING_STEPS:-1}"
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-16}"
DRY_RUN="${DRY_RUN:-true}"

# Safety clamps, in physical units (degrees; gripper 0-100).  0 disables.
#
# Disabled by default at the user's request so the raw model action is executed
# as-is.  The repo's INFERENCE_DEPLOY_THOR.md advises keeping these on; the
# concrete exposure is chunk-boundary jumps (step-13000 @ denoise=1 has
# max_normalized_action_jump = 0.1225, i.e. up to ~20 deg on shoulder_lift in a
# single step).  Restore the conservative values with:
#   MAX_DELTA_FROM_OBS=8 MAX_GRIPPER_DELTA_FROM_OBS=8 MAX_STEP_DELTA=4 MAX_GRIPPER_STEP_DELTA=5
PROFILE_STAGES="${PROFILE_STAGES:-false}"
TRACE_PATH="${TRACE_PATH:-}"

MAX_DELTA_FROM_OBS="${MAX_DELTA_FROM_OBS:-0}"
MAX_GRIPPER_DELTA_FROM_OBS="${MAX_GRIPPER_DELTA_FROM_OBS:-0}"
MAX_STEP_DELTA="${MAX_STEP_DELTA:-0}"
MAX_GRIPPER_STEP_DELTA="${MAX_GRIPPER_STEP_DELTA:-0}"
LOG_DIR="${LOG_DIR:-/home/hrx/Projects/models/three_cubes_1/cosmos_policy_step13000/shadow_logs}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

for p in "${CKPT_PATH}" "${TASK_ROOT}/processed_data/dataset_statistics.json" \
         "${TASK_ROOT}/processed_data/t5_text_embeddings.pkl" \
         "${TASK_ROOT}/configs/eval_config.py" "${TASK_ROOT}/processed_data/split.json"; do
  [[ -e "${p}" ]] || { echo "Missing required path: ${p}" >&2; exit 2; }
done

mkdir -p "${LOG_DIR}"

# The command below runs under a pipe to tee, so this script's PID is the shell,
# not python.  Killing the shell leaves the server orphaned and still bound --
# and because two processes can end up listening on the same port, a stale
# orphan will silently serve the client while the fresh server logs nothing.
# Refuse to start rather than produce a run whose traces belong to another
# process.
if ss -ltn 2>/dev/null | grep -q ":${PORT}\b"; then
  echo "ERROR: port ${PORT} is already in use. A previous server is still running." >&2
  ss -ltnp 2>/dev/null | grep ":${PORT}\b" >&2 || true
  echo "Stop it with:  pkill -9 -f so101_async_deploy_three_cubes_k16" >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/home/hrx/.cache/huggingface}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
# three_cubes_full_ft.py resolves split/stats/T5 from these; defaults point at
# the original training machine.
export THREE_CUBES_OUTPUT_ROOT="${TASK_ROOT}"
export THREE_CUBES_DATASET_ROOT="${DATASET_ROOT}"
export THREE_CUBES_CHUNK_SIZE="${THREE_CUBES_CHUNK_SIZE:-16}"
# `config_file=configs/eval_config.py` is relative and does `from configs...`,
# so the task package must be both the cwd and importable.
export PYTHONPATH="${TASK_ROOT}:${COSMOS_REPO}:${LEROBOT_SRC}:${PYTHONPATH:-}"

SERVER_LOG="${LOG_DIR}/server_13k_denoise${NUM_DENOISING_STEPS}_dry${DRY_RUN}_${RUN_TAG}.log"
echo "Cosmos 13K server -> ${SERVER_LOG}"
echo "  ckpt=${CKPT_PATH}"
echo "  denoise_steps=${NUM_DENOISING_STEPS} actions_per_chunk=${ACTIONS_PER_CHUNK} dry_run=${DRY_RUN}"
echo "  clamps: from_obs=${MAX_DELTA_FROM_OBS} grip_from_obs=${MAX_GRIPPER_DELTA_FROM_OBS} step=${MAX_STEP_DELTA} grip_step=${MAX_GRIPPER_STEP_DELTA}  (0 = disabled)"
if [[ "${MAX_DELTA_FROM_OBS}" == "0" && "${MAX_STEP_DELTA}" == "0" ]]; then
  echo "  >> SERVER SAFETY CLAMPS DISABLED: raw model actions are returned as-is."
fi

cd "${TASK_ROOT}"

exec "${PYTHON_BIN}" -u -m cosmos_policy.experiments.robot.so101_async_deploy_three_cubes_k16 \
  --host="${HOST}" \
  --port="${PORT}" \
  --fps="${FPS}" \
  --ckpt_path="${CKPT_PATH}" \
  --dataset_stats_path="${TASK_ROOT}/processed_data/dataset_statistics.json" \
  --t5_text_embeddings_path="${TASK_ROOT}/processed_data/t5_text_embeddings.pkl" \
  --num_denoising_steps_action="${NUM_DENOISING_STEPS}" \
  --actions_per_chunk="${ACTIONS_PER_CHUNK}" \
  --max_delta_from_observation="${MAX_DELTA_FROM_OBS}" \
  --max_gripper_delta_from_observation="${MAX_GRIPPER_DELTA_FROM_OBS}" \
  --max_step_delta="${MAX_STEP_DELTA}" \
  --max_gripper_step_delta="${MAX_GRIPPER_STEP_DELTA}" \
  --profile_stages="${PROFILE_STAGES}" \
  --trace_path="${TRACE_PATH:-${LOG_DIR}/server_stage_trace_${RUN_TAG}.jsonl}" \
  --use_wrist_image="${USE_WRIST_IMAGE:-true}" \
  --num_wrist_images="${NUM_WRIST_IMAGES:-2}" \
  --left_wrist_camera_key="${LEFT_WRIST_CAMERA_KEY:-right}" \
  --dry_run_zero_actions="${DRY_RUN}" \
  "$@" 2>&1 | tee "${SERVER_LOG}"
