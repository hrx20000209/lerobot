#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="${REPO_DIR:-/home/hrx/Projects/lerobot}"
PYTHON_BIN="${PYTHON_BIN:-/home/hrx/miniconda3/envs/lerobot/bin/python}"

# Classic async logs are written by robot_client.py / policy_server.py as
# robot_client_latency_*.jsonl and policy_server_latency_*.jsonl.
TIMELINE_LOG_DIR="${TIMELINE_LOG_DIR:-${REPO_DIR}/logs/async_timeline}"

# Continuous runs write a single structured timeline.jsonl in RUN_DIR.
RUN_DIR="${RUN_DIR:-}"
LOG_PATH="${LOG_PATH:-}"

OUTPUT_DIR="${OUTPUT_DIR:-}"
OUTPUT_PATH="${OUTPUT_PATH:-}"
SUMMARY_PATH="${SUMMARY_PATH:-}"
WINDOW_START="${WINDOW_START:-0}"
# 0 means plot until the end of the selected log(s).
WINDOW_DURATION="${WINDOW_DURATION:-0}"

SECONDS_PER_INCH="${SECONDS_PER_INCH:-1.4}"
MIN_FIG_WIDTH="${MIN_FIG_WIDTH:-24}"
FIG_HEIGHT="${FIG_HEIGHT:-9}"
DPI="${DPI:-220}"
OBS_LABEL_STRIDE="${OBS_LABEL_STRIDE:-1}"
INFERENCE_LABEL_STRIDE="${INFERENCE_LABEL_STRIDE:-1}"
ACTION_LABEL_STRIDE="${ACTION_LABEL_STRIDE:-5}"
QUEUE_LABEL_STRIDE="${QUEUE_LABEL_STRIDE:-3}"

# Optional detailed pages. The overview can cover the whole run; pages make
# every 30 Hz action label readable without creating a single giant PNG.
GENERATE_PAGES="${GENERATE_PAGES:-false}"
PAGE_SECONDS="${PAGE_SECONDS:-10}"
PAGE_SECONDS_PER_INCH="${PAGE_SECONDS_PER_INCH:-3.5}"
PAGE_ACTION_LABEL_STRIDE="${PAGE_ACTION_LABEL_STRIDE:-1}"

cd "${REPO_DIR}"

latest_file() {
  local pattern="$1"
  local root="$2"
  find "${root}" -type f -name "${pattern}" -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

LOG_ARGS=()
if [[ -n "${LOG_PATH}" ]]; then
  # Space-separated list is intentional for shell convenience.
  read -r -a LOG_ARGS <<<"${LOG_PATH}"
elif [[ -n "${RUN_DIR}" && -f "${RUN_DIR}/timeline.jsonl" ]]; then
  LOG_ARGS=("${RUN_DIR}/timeline.jsonl")
  OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}}"
else
  client_log="$(latest_file 'robot_client_latency_*.jsonl' "${TIMELINE_LOG_DIR}")"
  server_log="$(latest_file 'policy_server_latency_*.jsonl' "${TIMELINE_LOG_DIR}")"
  if [[ -z "${client_log}" && -z "${server_log}" ]]; then
    echo "No async timeline logs found." >&2
    echo "Set LOG_PATH, RUN_DIR, or TIMELINE_LOG_DIR. Current TIMELINE_LOG_DIR=${TIMELINE_LOG_DIR}" >&2
    exit 2
  fi
  if [[ -n "${client_log}" ]]; then
    LOG_ARGS+=("${client_log}")
  fi
  if [[ -n "${server_log}" ]]; then
    LOG_ARGS+=("${server_log}")
  fi
  OUTPUT_DIR="${OUTPUT_DIR:-${TIMELINE_LOG_DIR}}"
fi

mkdir -p "${OUTPUT_DIR}"
if [[ -z "${OUTPUT_PATH}" ]]; then
  OUTPUT_PATH="${OUTPUT_DIR}/async_timeline_readable.png"
fi
if [[ -z "${SUMMARY_PATH}" ]]; then
  SUMMARY_PATH="${OUTPUT_DIR}/async_timeline_readable_summary.json"
fi

"${PYTHON_BIN}" -m lerobot.scripts.plot_async_timeline \
  --log_path "${LOG_ARGS[@]}" \
  --output_path "${OUTPUT_PATH}" \
  --window_start "${WINDOW_START}" \
  --window_duration "${WINDOW_DURATION}" \
  --summary_path "${SUMMARY_PATH}" \
  --seconds_per_inch "${SECONDS_PER_INCH}" \
  --min_fig_width "${MIN_FIG_WIDTH}" \
  --fig_height "${FIG_HEIGHT}" \
  --dpi "${DPI}" \
  --obs_label_stride "${OBS_LABEL_STRIDE}" \
  --inference_label_stride "${INFERENCE_LABEL_STRIDE}" \
  --action_label_stride "${ACTION_LABEL_STRIDE}" \
  --queue_label_stride "${QUEUE_LABEL_STRIDE}"

echo "Logs:"
printf '  %s\n' "${LOG_ARGS[@]}"
echo "Timeline PNG: ${OUTPUT_PATH}"
echo "Timeline PDF: ${OUTPUT_PATH%.*}.pdf"
echo "Summary JSON: ${SUMMARY_PATH}"

if [[ "${GENERATE_PAGES}" =~ ^([Tt]rue|1|yes|YES)$ ]]; then
  pages_dir="${OUTPUT_DIR}/async_timeline_pages"
  mkdir -p "${pages_dir}"
  total_seconds="$("${PYTHON_BIN}" - "${LOG_ARGS[@]}" <<'PY'
import sys
from lerobot.scripts.plot_async_timeline import _load_events

events = _load_events(sys.argv[1:])
timestamps = [
    float(event["monotonic_ts"])
    for event in events
    if isinstance(event.get("monotonic_ts"), (int, float)) and not isinstance(event.get("monotonic_ts"), bool)
]
print(max(timestamps) - min(timestamps) if timestamps else 0)
PY
)"
  page_count="$("${PYTHON_BIN}" - <<PY
import math
print(max(1, int(math.ceil(float("${total_seconds}") / float("${PAGE_SECONDS}")))))
PY
)"
  for page_idx in $(seq 0 $((page_count - 1))); do
    page_start="$("${PYTHON_BIN}" - <<PY
print(float("${page_idx}") * float("${PAGE_SECONDS}"))
PY
)"
    page_label="$(printf '%03d' "${page_idx}")"
    "${PYTHON_BIN}" -m lerobot.scripts.plot_async_timeline \
      --log_path "${LOG_ARGS[@]}" \
      --output_path "${pages_dir}/timeline_page_${page_label}.png" \
      --window_start "${page_start}" \
      --window_duration "${PAGE_SECONDS}" \
      --summary_path "${pages_dir}/timeline_page_${page_label}_summary.json" \
      --seconds_per_inch "${PAGE_SECONDS_PER_INCH}" \
      --min_fig_width "${MIN_FIG_WIDTH}" \
      --fig_height "${FIG_HEIGHT}" \
      --dpi "${DPI}" \
      --obs_label_stride "${OBS_LABEL_STRIDE}" \
      --inference_label_stride "${INFERENCE_LABEL_STRIDE}" \
      --action_label_stride "${PAGE_ACTION_LABEL_STRIDE}" \
      --queue_label_stride "${QUEUE_LABEL_STRIDE}" \
      >"${pages_dir}/timeline_page_${page_label}_stdout.log"
  done
  echo "Detailed pages: ${pages_dir}"
fi
