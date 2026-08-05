#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Cosmos Policy step-20000 SO101 policy server on Jetson Thor.
#
# Pair with run_cosmos20k_client_thor.sh, which drives the real arm.
# Start this first and wait for the port to open (model load takes 2-3 min).
# ============================================================================
#
# Defaults chosen from the 2026-08-05 profiling runs:
#
#   NUM_DENOISING_STEPS=1   1597 ms at 10 steps vs 845 ms at 1, with no offline
#                           accuracy penalty on this checkpoint.
#   TRUNCATE_VAE_ENCODE     Encode only the 17 frames the model conditions on
#                           instead of all 41. Bit-identical conditioning
#                           latents (causal tokenizer), max action delta
#                           0.19 deg, and it removes 328 ms per chunk.
#   FPS=8                   Inference is ~520 ms for a 16-action chunk, so a
#                           chunk spans 2 s at 8 fps and the queue never starves.
#                           30 fps starved it half the time. MUST match the client.
#   clamps OFF              The generated action reaches the motors unmodified.
#                           See the MAX_* block below for what that trades away.

COSMOS_REPO="${COSMOS_REPO:-/home/hrx/Projects/cosmos-policy}"
LEROBOT_SRC="${LEROBOT_SRC:-/home/hrx/Projects/lerobot/src}"
PYTHON_BIN="${PYTHON_BIN:-/home/hrx/miniconda3/envs/cosmos/bin/python}"

# Experiment package: configs/ + processed_data/ (stats + T5), shared by all
# checkpoints of this training run.
TASK_ROOT="${TASK_ROOT:-/home/hrx/Projects/models/three_cubes_1/cosmos_policy}"
DATASET_ROOT="${DATASET_ROOT:-/home/hrx/Datasets/three_cubes_1}"
CKPT_ROOT="${CKPT_ROOT:-/home/hrx/Projects/models/three_cubes_1/cosmos_policy_step20000}"

# Export layouts differ: step20000 puts the DCP shards in model/, step13000
# nests them in model/model/. Locate them rather than hardcoding either.
if [[ -z "${CKPT_PATH:-}" ]]; then
  if compgen -G "${CKPT_ROOT}/model/model/*.distcp" > /dev/null; then
    CKPT_PATH="${CKPT_ROOT}/model/model"
  elif compgen -G "${CKPT_ROOT}/model/*.distcp" > /dev/null; then
    CKPT_PATH="${CKPT_ROOT}/model"
  else
    echo "ERROR: no .distcp shards under ${CKPT_ROOT}/model[/model]" >&2
    exit 2
  fi
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8082}"
FPS="${FPS:-8}"
NUM_DENOISING_STEPS="${NUM_DENOISING_STEPS:-1}"
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-16}"
TRUNCATE_VAE_ENCODE="${TRUNCATE_VAE_ENCODE:-true}"
PROFILE_STAGES="${PROFILE_STAGES:-true}"

# dry_run_zero_actions=true makes the server return the measured pose instead of
# the model action, so nothing can move. Only for plumbing smoke tests.
DRY_RUN="${DRY_RUN:-false}"

# Safety clamps in physical units (degrees; gripper 0-100). 0 disables.
#
# DISABLED BY DEFAULT: the generated action is executed as-is.
#
# What that costs, measured 2026-08-05, so the trade is on the record:
#   - With clamps on, the gripper locked closed on the cube and never released;
#     with them off, it kept cycling open/closed and did release. The clamps do
#     not block the release directly (the policy never commands one while
#     locked) -- they change the whole closed-loop trajectory, and the policy
#     ends up in a different basin.
#   - Clamps on also cut shoulder_lift tracking error ~46%.
#   - Each configuration overloaded a different servo: clamps off drove
#     shoulder_lift to 60 C, clamps on had the gripper stall on the cube until
#     its current protection latched.
#
# Restore the conservative values with:
#   MAX_DELTA_FROM_OBS=8 MAX_GRIPPER_DELTA_FROM_OBS=8 MAX_STEP_DELTA=4 MAX_GRIPPER_STEP_DELTA=5
MAX_DELTA_FROM_OBS="${MAX_DELTA_FROM_OBS:-0}"
MAX_GRIPPER_DELTA_FROM_OBS="${MAX_GRIPPER_DELTA_FROM_OBS:-0}"
MAX_STEP_DELTA="${MAX_STEP_DELTA:-0}"
MAX_GRIPPER_STEP_DELTA="${MAX_GRIPPER_STEP_DELTA:-0}"

# Camera -> Cosmos slot mapping. Must match the client's camera names.
PRIMARY_CAMERA_KEY="${PRIMARY_CAMERA_KEY:-front}"
LEFT_WRIST_CAMERA_KEY="${LEFT_WRIST_CAMERA_KEY:-right}"
RIGHT_WRIST_CAMERA_KEY="${RIGHT_WRIST_CAMERA_KEY:-wrist}"
USE_WRIST_IMAGE="${USE_WRIST_IMAGE:-true}"
NUM_WRIST_IMAGES="${NUM_WRIST_IMAGES:-2}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${LOG_DIR:-${CKPT_ROOT}/profiling}"
TRACE_PATH="${TRACE_PATH:-${LOG_DIR}/${RUN_TAG}_server_stage_trace.jsonl}"
SERVER_LOG="${SERVER_LOG:-${LOG_DIR}/${RUN_TAG}_server.log}"

for p in "${CKPT_PATH}" "${TASK_ROOT}/processed_data/dataset_statistics.json" \
         "${TASK_ROOT}/processed_data/t5_text_embeddings.pkl" \
         "${TASK_ROOT}/configs/eval_config.py" "${TASK_ROOT}/processed_data/split.json"; do
  [[ -e "${p}" ]] || { echo "Missing required path: ${p}" >&2; exit 2; }
done

mkdir -p "${LOG_DIR}"

# This script runs the server under a pipe to tee, so its own PID is the shell,
# not python. Killing the shell leaves the server orphaned and still bound --
# and two processes can end up listening on the same port, in which case a stale
# orphan silently serves the client while the fresh server logs nothing. Refuse
# to start rather than produce a run whose traces belong to another process.
if ss -ltn 2>/dev/null | grep -q ":${PORT}\b"; then
  echo "ERROR: port ${PORT} is already in use. A previous server is still running." >&2
  ss -ltnp 2>/dev/null | grep ":${PORT}\b" >&2 || true
  echo "Stop it with:  pkill -9 -f so101_async_deploy_three_cubes_k16" >&2
  exit 3
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/home/hrx/.cache/huggingface}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
# three_cubes_full_ft.py resolves split/stats/T5 from these; its defaults point
# at the original training machine.
export THREE_CUBES_OUTPUT_ROOT="${TASK_ROOT}"
export THREE_CUBES_DATASET_ROOT="${DATASET_ROOT}"
export THREE_CUBES_CHUNK_SIZE="${THREE_CUBES_CHUNK_SIZE:-16}"
# config_file=configs/eval_config.py is relative and does `from configs...`, so
# the task package must be both the cwd and importable.
export PYTHONPATH="${TASK_ROOT}:${COSMOS_REPO}:${LEROBOT_SRC}:${PYTHONPATH:-}"

echo "=========================================================="
echo " Cosmos step-20000 policy server"
echo "   ckpt        : ${CKPT_PATH}"
echo "   denoise     : ${NUM_DENOISING_STEPS}   chunk: ${ACTIONS_PER_CHUNK}   fps: ${FPS}"
echo "   truncate VAE: ${TRUNCATE_VAE_ENCODE}"
echo "   clamps      : obs=${MAX_DELTA_FROM_OBS} step=${MAX_STEP_DELTA} grip=${MAX_GRIPPER_DELTA_FROM_OBS}/${MAX_GRIPPER_STEP_DELTA} (0 = disabled)"
if [[ "${MAX_DELTA_FROM_OBS}" == "0" && "${MAX_STEP_DELTA}" == "0" \
   && "${MAX_GRIPPER_DELTA_FROM_OBS}" == "0" && "${MAX_GRIPPER_STEP_DELTA}" == "0" ]]; then
  echo "   >> UNBOUNDED: the generated action is executed as-is, no clamping anywhere."
fi
[[ "${DRY_RUN}" == "true" ]] && echo "   >> DRY RUN: returning measured pose, not model actions."
echo "   log         : ${SERVER_LOG}"
echo "   stage trace : ${TRACE_PATH}"
echo "   listening   : ${HOST}:${PORT}   (model load takes 2-3 min)"
echo "=========================================================="

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
  --action_gain="${ACTION_GAIN:-1.0}" \
  --action_stride="${ACTION_STRIDE:-1}" \
  --truncate_vae_encode="${TRUNCATE_VAE_ENCODE}" \
  --generate_future_state="${GENERATE_FUTURE_STATE:-false}" \
  --future_state_dir="${FUTURE_STATE_DIR:-${LOG_DIR}/${RUN_TAG}_future_state}" \
  --max_delta_from_observation="${MAX_DELTA_FROM_OBS}" \
  --max_gripper_delta_from_observation="${MAX_GRIPPER_DELTA_FROM_OBS}" \
  --max_step_delta="${MAX_STEP_DELTA}" \
  --max_gripper_step_delta="${MAX_GRIPPER_STEP_DELTA}" \
  --primary_camera_key="${PRIMARY_CAMERA_KEY}" \
  --left_wrist_camera_key="${LEFT_WRIST_CAMERA_KEY}" \
  --right_wrist_camera_key="${RIGHT_WRIST_CAMERA_KEY}" \
  --use_wrist_image="${USE_WRIST_IMAGE}" \
  --num_wrist_images="${NUM_WRIST_IMAGES}" \
  --profile_stages="${PROFILE_STAGES}" \
  --trace_path="${TRACE_PATH}" \
  --dry_run_zero_actions="${DRY_RUN}" \
  "$@" 2>&1 | tee "${SERVER_LOG}"
