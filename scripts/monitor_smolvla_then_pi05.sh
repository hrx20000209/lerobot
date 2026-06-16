#!/usr/bin/env bash
set -euo pipefail

TARGET_GPU="${TARGET_GPU:-2}"
LOSS_THRESHOLD="${LOSS_THRESHOLD:-0.070}"
SMOL_SAVE_FREQ="${SMOL_SAVE_FREQ:-4000}"
CHECK_INTERVAL_S="${CHECK_INTERVAL_S:-20}"
MIN_FREE_MIB_FOR_PI05="${MIN_FREE_MIB_FOR_PI05:-21000}"
LARGE_MEM_MIB="${LARGE_MEM_MIB:-2048}"

SMOL_LOG="${SMOL_LOG:-output_lerobot_train/logs/smolvla_front_wrist_b96.log}"
SMOL_OUTPUT_DIR="${SMOL_OUTPUT_DIR:-output_lerobot_train/three_cubes/smolvla_front_wrist_b96}"
SMOL_CMD_PATTERN="${SMOL_CMD_PATTERN:-smolvla_front_wrist_b96}"

PI05_OUTPUT_DIR="${PI05_OUTPUT_DIR:-output_lerobot_train/three_cubes/pi05_front_wrist_r64_full_expert_pad01_b24_after_smolvla}"
PI05_LOG="${PI05_LOG:-output_lerobot_train/logs/pi05_front_wrist_r64_full_expert_pad01_b24_after_smolvla.log}"
PI05_BATCH_SIZE="${PI05_BATCH_SIZE:-24}"
PI05_SAVE_FREQ="${PI05_SAVE_FREQ:-2000}"
PI05_LOG_FREQ="${PI05_LOG_FREQ:-20}"
PI05_NUM_WORKERS="${PI05_NUM_WORKERS:-2}"
OLD_PI05_WAIT_SESSION="${OLD_PI05_WAIT_SESSION:-pi05_front_wrist_b24}"

cd /home/rxhuang/Projects/lerobot
mkdir -p output_lerobot_train/logs

log() {
  echo "[$(date --iso-8601=seconds)] $*"
}

parse_latest_smol_metric() {
  python - "${SMOL_LOG}" <<'PY'
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    text = path.read_text(errors="ignore")
except FileNotFoundError:
    print("0 nan")
    raise SystemExit

latest = None
for line in text.replace("\r", "\n").splitlines():
    if "loss:" not in line or "ot_train.py:548" not in line:
        continue

    loss_match = re.search(r"loss:([0-9]+(?:\.[0-9]+)?)", line)
    if not loss_match:
        continue

    step = None
    progress_match = re.search(r"\|\s*(\d+)/\d+\s*\[", line)
    if progress_match:
        step = int(progress_match.group(1))
    else:
        step_match = re.search(r"step:(\d+)(K?)", line)
        if step_match:
            step = int(step_match.group(1)) * (1000 if step_match.group(2) else 1)

    if step is not None:
        latest = (step, float(loss_match.group(1)))

if latest is None:
    print("0 nan")
else:
    print(f"{latest[0]} {latest[1]:.6f}")
PY
}

loss_is_below_threshold() {
  local loss="$1"
  python - "${loss}" "${LOSS_THRESHOLD}" <<'PY'
import math
import sys

try:
    loss = float(sys.argv[1])
    threshold = float(sys.argv[2])
except ValueError:
    raise SystemExit(1)

if math.isfinite(loss) and loss < threshold:
    raise SystemExit(0)
raise SystemExit(1)
PY
}

checkpoint_dir_for_step() {
  printf "%s/checkpoints/%06d" "${SMOL_OUTPUT_DIR}" "$1"
}

latest_smol_pids() {
  pgrep -f "${SMOL_CMD_PATTERN}" || true
}

stop_smolvla() {
  local pids
  pids="$(latest_smol_pids)"
  if [ -z "${pids}" ]; then
    log "SmolVLA process pattern '${SMOL_CMD_PATTERN}' is already gone"
    return
  fi

  log "Stopping SmolVLA PIDs: ${pids//$'\n'/ }"
  kill ${pids} 2>/dev/null || true

  for _ in $(seq 1 30); do
    sleep 1
    pids="$(latest_smol_pids)"
    [ -z "${pids}" ] && return
  done

  pids="$(latest_smol_pids)"
  if [ -n "${pids}" ]; then
    log "SmolVLA did not exit after SIGTERM; sending SIGKILL to: ${pids//$'\n'/ }"
    kill -9 ${pids} 2>/dev/null || true
  fi
}

gpu_free_mib() {
  nvidia-smi --id="${TARGET_GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '
}

gpu_uuid() {
  nvidia-smi --id="${TARGET_GPU}" --query-gpu=uuid --format=csv,noheader | tr -d ' '
}

kill_large_non_pi05_processes_on_gpu() {
  local uuid
  uuid="$(gpu_uuid)"

  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits |
    while IFS=',' read -r app_uuid raw_pid raw_name raw_mem; do
      app_uuid="$(echo "${app_uuid}" | tr -d ' ')"
      local pid name mem cmd
      pid="$(echo "${raw_pid:-}" | tr -d ' ')"
      name="$(echo "${raw_name:-}" | sed 's/^ *//;s/ *$//')"
      mem="$(echo "${raw_mem:-0}" | tr -d ' ')"

      [ "${app_uuid}" = "${uuid}" ] || continue
      [ -n "${pid}" ] || continue
      [ "${pid}" != "$$" ] || continue
      [ "${mem:-0}" -ge "${LARGE_MEM_MIB}" ] || continue

      cmd="$(ps -p "${pid}" -o cmd= 2>/dev/null || true)"
      case "${cmd}" in
        *pi05_front_wrist_r64_full_expert_pad01_b24_after_smolvla*|*three_cubes_pi05_front_wrist*)
          log "Keeping PI05-related process on GPU${TARGET_GPU}: pid=${pid}, mem=${mem} MiB"
          continue
          ;;
      esac

      log "Killing non-PI05 large GPU${TARGET_GPU} process: pid=${pid}, mem=${mem} MiB, name=${name}, cmd=${cmd}"
      kill "${pid}" 2>/dev/null || true
    done

  sleep 5

  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits |
    while IFS=',' read -r app_uuid raw_pid raw_name raw_mem; do
      app_uuid="$(echo "${app_uuid}" | tr -d ' ')"
      local pid mem cmd
      pid="$(echo "${raw_pid:-}" | tr -d ' ')"
      mem="$(echo "${raw_mem:-0}" | tr -d ' ')"

      [ "${app_uuid}" = "$(gpu_uuid)" ] || continue
      [ -n "${pid}" ] || continue
      [ "${pid}" != "$$" ] || continue
      [ "${mem:-0}" -ge "${LARGE_MEM_MIB}" ] || continue

      cmd="$(ps -p "${pid}" -o cmd= 2>/dev/null || true)"
      case "${cmd}" in
        *pi05_front_wrist_r64_full_expert_pad01_b24_after_smolvla*|*three_cubes_pi05_front_wrist*)
          continue
          ;;
      esac

      log "SIGKILL lingering non-PI05 GPU${TARGET_GPU} process: pid=${pid}, mem=${mem} MiB, cmd=${cmd}"
      kill -9 "${pid}" 2>/dev/null || true
    done
}

wait_for_clean_gpu() {
  local free_mib
  while true; do
    free_mib="$(gpu_free_mib)"
    log "GPU${TARGET_GPU} free=${free_mib} MiB before PI05"
    if [ "${free_mib}" -ge "${MIN_FREE_MIB_FOR_PI05}" ]; then
      return
    fi

    kill_large_non_pi05_processes_on_gpu
    sleep 10
  done
}

if tmux has-session -t "${OLD_PI05_WAIT_SESSION}" 2>/dev/null; then
  log "Killing old PI05 wait tmux session '${OLD_PI05_WAIT_SESSION}' to avoid duplicate PI05 launch"
  tmux kill-session -t "${OLD_PI05_WAIT_SESSION}" || true
fi

log "Monitoring SmolVLA: threshold=${LOSS_THRESHOLD}, save_freq=${SMOL_SAVE_FREQ}, target_gpu=${TARGET_GPU}"

threshold_seen=0
target_checkpoint_step=0

while true; do
  read -r latest_step latest_loss < <(parse_latest_smol_metric)
  log "Latest SmolVLA metric: step=${latest_step}, loss=${latest_loss}"

  if [ "${latest_step}" -gt 0 ] && loss_is_below_threshold "${latest_loss}"; then
    if [ "${threshold_seen}" -eq 0 ]; then
      threshold_seen=1
      target_checkpoint_step=$(( ((latest_step + SMOL_SAVE_FREQ - 1) / SMOL_SAVE_FREQ) * SMOL_SAVE_FREQ ))
      if [ "${target_checkpoint_step}" -lt "${SMOL_SAVE_FREQ}" ]; then
        target_checkpoint_step="${SMOL_SAVE_FREQ}"
      fi
      log "SmolVLA loss ${latest_loss} < ${LOSS_THRESHOLD}; waiting for checkpoint ${target_checkpoint_step}"
    fi
  fi

  if [ "${threshold_seen}" -eq 1 ]; then
    ckpt_dir="$(checkpoint_dir_for_step "${target_checkpoint_step}")"
    if [ -d "${ckpt_dir}/pretrained_model" ] && [ -d "${ckpt_dir}/training_state" ]; then
      log "Checkpoint is ready after threshold: ${ckpt_dir}"
      stop_smolvla
      wait_for_clean_gpu
      log "Starting PI05 on GPU${TARGET_GPU}; batch=${PI05_BATCH_SIZE}; output=${PI05_OUTPUT_DIR}; log=${PI05_LOG}"
      exec env \
        GPU="${TARGET_GPU}" \
        BATCH_SIZE="${PI05_BATCH_SIZE}" \
        SAVE_FREQ="${PI05_SAVE_FREQ}" \
        LOG_FREQ="${PI05_LOG_FREQ}" \
        NUM_WORKERS="${PI05_NUM_WORKERS}" \
        OUTPUT_DIR="${PI05_OUTPUT_DIR}" \
        scripts/train_three_cubes_front_wrist_pi05.sh >> "${PI05_LOG}" 2>&1
    fi

    log "Waiting for checkpoint ${target_checkpoint_step}: ${ckpt_dir}"
  fi

  sleep "${CHECK_INTERVAL_S}"
done
