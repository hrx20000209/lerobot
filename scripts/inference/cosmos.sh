#!/usr/bin/env bash
# Single entry point for the Cosmos SO101 experiments.
#
#   cosmos.sh status                 what is running, where the arm is
#   cosmos.sh home                   ramp the arm back to the rest pose
#   cosmos.sh server [--fg]          start the policy server (background by default)
#   cosmos.sh stop                   stop the policy server
#   cosmos.sh shadow  [SECONDS]      run with the motor bus read-only (arm will NOT move)
#   cosmos.sh real    [SECONDS]      run for real (THE ARM MOVES)
#   cosmos.sh go      [SECONDS]      server (if needed) -> wait -> real run -> leave server up
#   cosmos.sh analyze [RUN_TAG]      stage budget + async timeline + dwell phases
#
# Settings are env vars; the ones the server and client must agree on are set in
# one place here so they cannot drift apart:
#
#   CKPT_ROOT   checkpoint dir            (default: step20000)
#   FPS         control rate              (default: 8)   server and client must match
#   TRUNCATE    truncated VAE encode      (default: true)
#   CLAMPS      safety clamps on/off      (default: OFF -- action executed as-is)
#   RUN_TAG     names the run's artifacts (default: timestamp)
#
# Examples
#   cosmos.sh go 180
#   CKPT_ROOT=~/Projects/models/three_cubes_1/cosmos_policy_step13000 cosmos.sh go 60
#   TRUNCATE=false RUN_TAG=baseline cosmos.sh go 100
#   CLAMPS=on  cosmos.sh real 30          # restore the conservative 8/4 deg bounds

set -euo pipefail

REPO_DIR="${REPO_DIR:-/home/hrx/Projects/lerobot}"
SCRIPTS="${REPO_DIR}/scripts/inference"
PORT="${PORT:-8082}"
SERVER_PATTERN="so101_async_deploy_three_cubes_k16"

export CKPT_ROOT="${CKPT_ROOT:-/home/hrx/Projects/models/three_cubes_1/cosmos_policy_step20000}"
export FPS="${FPS:-8}"
export ROBOT_PORT="${ROBOT_PORT:-/dev/ttyACM0}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
TRUNCATE="${TRUNCATE:-true}"
# Off by default: the generated action is executed as-is. CLAMPS=on restores
# the conservative 8/4 deg bounds (see run_cosmos20k_server_thor.sh for what
# each setting was measured to cost).
CLAMPS="${CLAMPS:-off}"

LOG_ROOT="${LOG_ROOT:-${CKPT_ROOT}/profiling}"
SERVER_OUT="${LOG_ROOT}/${RUN_TAG}_server.log"

if [[ "${CLAMPS}" == "on" ]]; then
  export MAX_DELTA_FROM_OBS=8 MAX_GRIPPER_DELTA_FROM_OBS=8 MAX_STEP_DELTA=4 MAX_GRIPPER_STEP_DELTA=5
else
  export MAX_DELTA_FROM_OBS=0 MAX_GRIPPER_DELTA_FROM_OBS=0 MAX_STEP_DELTA=0 MAX_GRIPPER_STEP_DELTA=0
fi

PY="${PY:-/home/hrx/miniconda3/envs/lerobot/bin/python}"

_server_pid() { pgrep -f "${SERVER_PATTERN}" | head -1; }
_listening()  { ss -ltn 2>/dev/null | grep -q ":${PORT}\b"; }

_banner() {
  echo "-----------------------------------------------------------"
  echo " ckpt      : ${CKPT_ROOT##*/}"
  echo " fps       : ${FPS}    truncate_vae: ${TRUNCATE}    clamps: ${CLAMPS}"
  echo " run_tag   : ${RUN_TAG}"
  [[ "${CLAMPS}" == "off" ]] && echo " !! UNBOUNDED: generated action executed as-is, no clamping."
  echo "-----------------------------------------------------------"
}

cmd_status() {
  local pid; pid="$(_server_pid || true)"
  if [[ -n "${pid}" ]]; then echo "server : running (pid ${pid})"; else echo "server : stopped"; fi
  if _listening; then echo "port   : ${PORT} listening"; else echo "port   : ${PORT} free"; fi
  echo "ckpt   : ${CKPT_ROOT}"
  # A servo that has latched its overload protection drops off the bus
  # entirely, and the stock connect() then dies with a wall of "expected motor
  # list" output. That has happened repeatedly to the gripper, so name the
  # condition instead of dumping the traceback.
  PYTHONPATH="${REPO_DIR}/src" timeout 60 "${PY}" - <<EOF
from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors import Motor, MotorNormMode

names = [(1,"shoulder_pan"),(2,"shoulder_lift"),(3,"elbow_flex"),
         (4,"wrist_flex"),(5,"wrist_roll"),(6,"gripper")]
motors = {n: Motor(i, "sts3215",
                   MotorNormMode.RANGE_0_100 if n == "gripper" else MotorNormMode.DEGREES)
          for i, n in names}
bus = FeetechMotorsBus(port="${ROBOT_PORT}", motors=motors)
try:
    bus.connect(handshake=False)
except Exception as exc:
    print("arm    : <cannot open ${ROBOT_PORT}>:", exc)
    raise SystemExit(0)

dead = []
for i, n in names:
    try:
        if bus.ping(i) is None:
            dead.append((i, n))
    except Exception:
        dead.append((i, n))

if dead:
    print("arm    : MOTORS NOT RESPONDING ->", ", ".join(f"id{i} {n}" for i, n in dead))
    print("         这通常是过载保护锁存，需要给舵机断电重启一次。")
    raise SystemExit(0)

# Position needs the calibration to denormalise; temperature and load are raw
# registers and read fine without it. Report whatever is available rather than
# failing the whole status on a missing calibration.
for label, reg, rnd in (("temp   ", "Present_Temperature", False),
                        ("load   ", "Present_Load", False)):
    try:
        print(label + ":", bus.sync_read(reg))
    except Exception as exc:
        print(label + ": <unavailable>", type(exc).__name__)
try:
    temp = bus.sync_read("Present_Temperature")
    hot = [k for k, v in temp.items() if v >= 55]
    if hot:
        print("         偏热:", ", ".join(hot), "(今天 shoulder_lift 在 60C 跳过闸)")
except Exception:
    pass
try:
    from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig
    bus.disconnect()
    r = SOFollower(SOFollowerRobotConfig(port="${ROBOT_PORT}", id="follower_arm", cameras={},
                                         use_degrees=True, disable_torque_on_disconnect=False))
    r.connect()
    print("arm    :", {k: round(v, 1) for k, v in r.bus.sync_read("Present_Position").items()})
    r.disconnect()
except Exception as exc:
    print("arm    : <position unavailable>", type(exc).__name__)
EOF
}

cmd_stop() {
  if [[ -n "$(_server_pid || true)" ]]; then
    # The launcher runs the server under a pipe to tee, so its own PID is a
    # shell; kill the python process by pattern or it is left orphaned and
    # still bound to the port.
    pkill -9 -f "${SERVER_PATTERN}" || true
    sleep 3
  fi
  _listening && echo "WARNING: port ${PORT} still bound" || echo "server stopped, port ${PORT} free"
}

cmd_home() {
  echo "Homing ${ROBOT_PORT} (dry run first)..."
  PYTHONPATH="${REPO_DIR}/src" "${PY}" "${SCRIPTS}/home_so101_follower.py" --port "${ROBOT_PORT}" --dry-run
  echo
  read -r -p "Proceed with the move? [y/N] " ans
  [[ "${ans}" == "y" || "${ans}" == "Y" ]] || { echo "aborted"; return 0; }
  PYTHONPATH="${REPO_DIR}/src" "${PY}" "${SCRIPTS}/home_so101_follower.py" \
    --port "${ROBOT_PORT}" --step-deg "${STEP_DEG:-0.8}"
}

cmd_server() {
  if _listening; then echo "Port ${PORT} already in use; run 'cosmos.sh stop' first." >&2; exit 3; fi
  mkdir -p "${LOG_ROOT}"
  _banner
  export DRY_RUN=false PROFILE_STAGES=true TRUNCATE_VAE_ENCODE="${TRUNCATE}" RUN_TAG
  export TRACE_PATH="${LOG_ROOT}/${RUN_TAG}_server_stage_trace.jsonl"
  if [[ "${1:-}" == "--fg" ]]; then
    exec "${SCRIPTS}/run_cosmos20k_server_thor.sh"
  fi
  nohup "${SCRIPTS}/run_cosmos20k_server_thor.sh" > "${SERVER_OUT}" 2>&1 &
  echo "server starting -> ${SERVER_OUT}"
}

cmd_wait() {
  echo -n "waiting for server"
  for _ in $(seq 1 60); do
    if _listening; then echo " ready"; return 0; fi
    if [[ -f "${SERVER_OUT}" ]] && grep -qE "Traceback|out of memory|ERROR: port" "${SERVER_OUT}"; then
      echo " FAILED"; tail -20 "${SERVER_OUT}" >&2; return 1
    fi
    echo -n "."; sleep 10
  done
  echo " timeout"; return 1
}

_client() {  # $1 = shadow|real, $2 = seconds
  _listening || { echo "No server on port ${PORT}. Run 'cosmos.sh server' first." >&2; exit 3; }
  _banner
  SHADOW=$([[ "$1" == "shadow" ]] && echo true || echo false) \
  RUN_SECONDS="${2:-60}" RUN_TAG="${RUN_TAG}" BASE_DIR="${LOG_ROOT}" \
    "${SCRIPTS}/run_cosmos20k_client_thor.sh"
}

cmd_analyze() {
  local tag="${1:-${RUN_TAG}}"
  "${PY}" "${SCRIPTS}/analyze_cosmos_profiling.py" \
    --run-dir "${LOG_ROOT}/${tag}" \
    --server-trace "${LOG_ROOT}/${tag}_server_stage_trace.jsonl"
}

case "${1:-}" in
  status)  cmd_status ;;
  stop)    cmd_stop ;;
  home)    cmd_home ;;
  server)  shift; cmd_server "$@" ;;
  shadow)  shift; _client shadow "${1:-60}" ;;
  real)    shift; _client real   "${1:-60}" ;;
  go)      shift
           if ! _listening; then cmd_server; cmd_wait || exit 1; fi
           _client real "${1:-60}"
           echo; echo "server left running; 'cosmos.sh analyze ${RUN_TAG}' then 'cosmos.sh stop'" ;;
  analyze) shift; cmd_analyze "$@" ;;
  *) awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 && !/^#/ {exit}' "$0" ;;
esac
