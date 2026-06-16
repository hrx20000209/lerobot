#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-1}"
MIN_FREE_MIB="${MIN_FREE_MIB:-21000}"

cd /home/rxhuang/Projects/lerobot

echo "[$(date --iso-8601=seconds)] waiting for GPU${GPU} free memory >= ${MIN_FREE_MIB} MiB before starting PI05 b24"

while true; do
  free_mib="$(nvidia-smi --id="${GPU}" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
  echo "[$(date --iso-8601=seconds)] GPU${GPU} free=${free_mib} MiB"
  if [ "${free_mib}" -ge "${MIN_FREE_MIB}" ]; then
    break
  fi
  sleep 60
done

echo "[$(date --iso-8601=seconds)] starting PI05 front+wrist b24 on GPU${GPU}"
exec env \
  GPU="${GPU}" \
  BATCH_SIZE=24 \
  NUM_WORKERS=2 \
  LOG_FREQ=20 \
  OUTPUT_DIR=output_lerobot_train/three_cubes/pi05_front_wrist_r64_full_expert_pad01_b24 \
  scripts/train_three_cubes_front_wrist_pi05.sh
