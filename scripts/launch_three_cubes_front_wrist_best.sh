#!/usr/bin/env bash
set -euo pipefail

mkdir -p output_lerobot_train/logs

GPU_CANDIDATES="${GPU_CANDIDATES:-0 1 2 3 4 5 6 7}"
POLL_SECONDS="${POLL_SECONDS:-60}"
MIN_FREE_MB="${MIN_FREE_MB:-22000}"
PI05_MIN_FREE_MB="${PI05_MIN_FREE_MB:-${MIN_FREE_MB}}"
SMOLVLA_MIN_FREE_MB="${SMOLVLA_MIN_FREE_MB:-${MIN_FREE_MB}}"
VLA_JEPA_MIN_FREE_MB="${VLA_JEPA_MIN_FREE_MB:-${MIN_FREE_MB}}"

selected_gpus=""

is_selected() {
  local gpu="$1"
  [[ " ${selected_gpus} " == *" ${gpu} "* ]]
}

gpu_free_mb() {
  local gpu="$1"
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F, -v target="${gpu}" '$1 + 0 == target {gsub(/ /, "", $2); print $2 + 0}'
}

choose_gpu() {
  local min_free_mb="$1"
  local line gpu free_mb
  while IFS=, read -r gpu free_mb; do
    gpu="${gpu// /}"
    free_mb="${free_mb// /}"
    if is_selected "${gpu}"; then
      continue
    fi
    for candidate in ${GPU_CANDIDATES}; do
      if [[ "${gpu}" == "${candidate}" && "${free_mb}" -ge "${min_free_mb}" ]]; then
        echo "${gpu}"
        return 0
      fi
    done
  done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
  return 1
}

wait_for_gpu() {
  local name="$1"
  local requested_gpu="$2"
  local min_free_mb="$3"
  local gpu free_mb
  while true; do
    if [[ -n "${requested_gpu}" ]]; then
      free_mb="$(gpu_free_mb "${requested_gpu}")"
      if [[ -n "${free_mb}" && "${free_mb}" -ge "${min_free_mb}" ]] && ! is_selected "${requested_gpu}"; then
        echo "${requested_gpu}"
        return 0
      fi
      echo "[$(date '+%F %T')] ${name}: waiting for GPU ${requested_gpu} to have >= ${min_free_mb} MiB free (now ${free_mb:-unknown})" >&2
    elif gpu="$(choose_gpu "${min_free_mb}")"; then
      echo "${gpu}"
      return 0
    else
      echo "[$(date '+%F %T')] ${name}: waiting for any GPU in [${GPU_CANDIDATES}] with >= ${min_free_mb} MiB free" >&2
    fi
    sleep "${POLL_SECONDS}"
  done
}

launch_job() {
  local name="$1"
  local requested_gpu="$2"
  local min_free_mb="$3"
  local script="$4"
  local log_path="$5"
  local gpu
  gpu="$(wait_for_gpu "${name}" "${requested_gpu}" "${min_free_mb}")"
  selected_gpus="${selected_gpus} ${gpu}"
  setsid bash -c "GPU=${gpu} ${script}" > "${log_path}" 2>&1 &
  echo "[$(date '+%F %T')] ${name}: launched on GPU ${gpu}, pid=$!, log=${log_path}"
  sleep 20
}

launch_job \
  "pi05_front_wrist" \
  "${PI05_GPU:-}" \
  "${PI05_MIN_FREE_MB}" \
  "scripts/train_three_cubes_front_wrist_pi05.sh" \
  "output_lerobot_train/logs/pi05_front_wrist_r64_full_expert_pad01_b16.log"

launch_job \
  "smolvla_front_wrist" \
  "${SMOLVLA_GPU:-}" \
  "${SMOLVLA_MIN_FREE_MB}" \
  "scripts/train_three_cubes_front_wrist_smolvla.sh" \
  "output_lerobot_train/logs/smolvla_front_wrist_b96.log"

launch_job \
  "vla_jepa_front_wrist" \
  "${VLA_JEPA_GPU:-}" \
  "${VLA_JEPA_MIN_FREE_MB}" \
  "scripts/train_three_cubes_front_wrist_vla_jepa.sh" \
  "output_lerobot_train/logs/vla_jepa_front_wrist_qv_r64_rep4_w01.log"
