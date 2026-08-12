#!/usr/bin/env python
"""Instrumented Cosmos SO101 client: full trace + async screenshots.

Same control loop as the stock ``robot_client``, plus:

* a JSONL trace with one record per executed action -- commanded action, the
  joint positions measured at that instant, the delta, action age, queue depth,
  and the server-side stage breakdown (VAE encode / DiT denoise / ...) that the
  profiling server attaches to the action metadata;
* an observation trace (capture time, per-camera capture cost, queue depth);
* screenshots written from a **background thread** at a fixed cadence, so JPEG
  encoding never lands in the control loop and distorts the timings we are
  trying to measure.

``--shadow=true`` (default) makes it physically unable to write to the motor
bus: ``send_action`` is replaced by a raiser and ``bus.sync_write`` refuses every
``Goal_*`` register.  ``--shadow=false`` performs the run for real and the arm
moves.  Both paths share this file so the instrumented code that gets validated
in shadow is the same code that runs on hardware.
"""

import json
import logging
import sys
import queue
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import draccus

import numpy as np

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient
from lerobot.utils.utils import init_logging

sys.path.insert(0, str(Path(__file__).resolve().parent))
from action_feedback import ActionFeedback, FeedbackConfig  # noqa: E402
from apex import Apex, ApexConfig  # noqa: E402
from chunk_reanchor import ChunkReanchor, ReanchorConfig  # noqa: E402

SO101_ACTION_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]

BLOCKED_REGISTERS = {"Goal_Position", "Goal_Current", "Goal_Velocity", "Goal_PWM"}


class MotorWriteBlocked(RuntimeError):
    """Raised when something attempts to command the arm during a shadow run."""


@dataclass
class ProfiledClientConfig(RobotClientConfig):
    shadow: bool = field(
        default=True,
        metadata={"help": "True = motor bus is read-only (cannot move). False = REAL run, the arm moves."},
    )
    trace_dir: str = field(default="", metadata={"help": "Directory for traces + screenshots."})
    screenshot_hz: float = field(default=2.0, metadata={"help": "Background screenshot rate. 0 disables."})
    screenshot_quality: int = field(default=85, metadata={"help": "JPEG quality for screenshots."})
    actuator_sample_every: int = field(
        default=4,
        metadata={
            "help": "Read Present_Load/Temperature every Nth action (0 disables). "
            "Each read costs a serial round-trip, so keep it well below the command rate."
        },
    )
    # --- closed-loop guards (see action_feedback.py) ---
    feedback: bool = field(default=False, metadata={"help": "Enable the action feedback guards."})
    fb_max_step_deg: float = field(default=0.0, metadata={"help": "Slew limit per command, deg. 0 = off."})
    fb_max_lead_deg: float = field(default=0.0, metadata={"help": "Max |command - measured|, deg. 0 = off."})
    fb_stall_load: float = field(default=95.0, metadata={"help": "|load| counting as saturated."})
    fb_stall_secs: float = field(default=1.0, metadata={"help": "Saturated-and-stuck time before freezing a joint."})
    fb_autotune_secs: float = field(
        default=0.0,
        metadata={"help": "Observe (do not act) for this long, then raise the thresholds to fit. 0 = fixed."},
    )
    fb_gripper_lead_deg: float = field(
        default=0.0,
        metadata={"help": "Gripper-only lead limit, deg. Off by default: it clips healthy grasps too. 0 = use fb_max_lead_deg."},
    )
    fb_fixed_point_secs: float = field(
        default=0.0,
        metadata={"help": "Net-displacement window, s. Below the move threshold over it (while squeezing) = release. 0 = off."},
    )
    fb_fixed_point_move_deg: float = field(
        default=3.0, metadata={"help": "Net displacement over the window counting as going nowhere."}
    )
    fb_fixed_point_release_deg: float = field(
        default=30.0, metadata={"help": "How far to open the gripper to break a fixed point."}
    )
    # --- APEX adaptive execution layer (see apex.py, arXiv:2606.16504) ---
    apex: bool = field(default=False, metadata={"help": "Enable the APEX execution-gap correction."})
    apex_k1: float = field(default=12.0, metadata={"help": "Reconstruction filter gain K1, 1/s."})
    apex_k2: float = field(default=12.0, metadata={"help": "Reconstruction filter gain K2, 1/s."})
    apex_alpha: float = field(default=6.0, metadata={"help": "Tracking-error feedback gain alpha."})
    apex_gamma_w: float = field(default=2.0e-4, metadata={"help": "Learning rate for w."})
    apex_gamma_wd: float = field(default=2.0e-4, metadata={"help": "Learning rate for w_d."})
    apex_gamma_j: float = field(default=2.0e-5, metadata={"help": "Learning rate for J."})
    apex_warmup_secs: float = field(default=1.5, metadata={"help": "Pass the policy command through for this long."})
    apex_max_correction_deg: float = field(
        default=12.0, metadata={"help": "Hard bound on |corrected - policy|, deg."}
    )
    # --- chunk re-anchoring (see chunk_reanchor.py) ---
    reanchor: bool = field(default=False, metadata={"help": "Re-anchor each chunk to the measured pose."})
    reanchor_blend_steps: int = field(default=8, metadata={"help": "Actions over which the offset decays to zero."})


class ProfiledRobotClient(RobotClient):
    prefix = "cosmos_profiled_client"
    logger = logging.getLogger("cosmos_profiled_client")

    def __init__(self, config: ProfiledClientConfig):
        super().__init__(config)
        self.shadow = config.shadow

        features = list(self.robot.action_features)
        if features != SO101_ACTION_NAMES:
            raise RuntimeError(
                "Robot action_features order does not match the Cosmos joint order.\n"
                f"  robot.action_features = {features}\n"
                f"  cosmos joint_order    = {SO101_ACTION_NAMES}\n"
                "Actions are mapped positionally, so this would command the wrong joints."
            )
        self.logger.info("Verified action feature order: %s", features)

        if self.shadow:
            self._install_write_guards()
            self.logger.warning("SHADOW: motor bus is read-only; the arm will NOT move.")
        else:
            self.logger.warning("REAL RUN: goal positions WILL be written to %s.", config.robot.port)

        base = Path(config.trace_dir or self.latency_recorder.log_dir)
        base.mkdir(parents=True, exist_ok=True)
        self.trace_dir = base
        run_id = self.latency_recorder.run_id
        self._action_trace = (base / f"action_trace_{run_id}.jsonl").open("w", buffering=1)
        self._obs_trace = (base / f"observation_trace_{run_id}.jsonl").open("w", buffering=1)
        self._trace_lock = threading.Lock()
        self._n_actions = 0
        self._n_obs = 0

        self.shot_dir = base / f"screenshots_{run_id}"
        self._shot_q: queue.Queue = queue.Queue(maxsize=8)
        self._shot_stop = threading.Event()
        self._shot_period = 1.0 / config.screenshot_hz if config.screenshot_hz > 0 else None
        self._last_shot = 0.0
        self._n_shots = 0
        self._shot_thread = None
        if self._shot_period is not None:
            self.shot_dir.mkdir(parents=True, exist_ok=True)
            self._shot_thread = threading.Thread(target=self._screenshot_worker, daemon=True)
            self._shot_thread.start()
            self.logger.info("Screenshots -> %s at %.1f Hz (background thread)", self.shot_dir, config.screenshot_hz)

        self.logger.info("Traces -> %s", base)

        self.feedback = ActionFeedback(
            SO101_ACTION_NAMES,
            FeedbackConfig(
                enabled=config.feedback,
                max_step_deg=config.fb_max_step_deg,
                max_lead_deg=config.fb_max_lead_deg,
                stall_load=config.fb_stall_load,
                stall_secs=config.fb_stall_secs,
                autotune_secs=config.fb_autotune_secs,
                max_lead_deg_per_joint=(
                    {"gripper.pos": config.fb_gripper_lead_deg}
                    if config.fb_gripper_lead_deg > 0
                    else {}
                ),
                fixed_point_secs=config.fb_fixed_point_secs,
                fixed_point_move_deg=config.fb_fixed_point_move_deg,
                fixed_point_release_deg=config.fb_fixed_point_release_deg,
            ),
        )
        self.apex = Apex(
            SO101_ACTION_NAMES,
            ApexConfig(
                enabled=config.apex,
                k1=config.apex_k1,
                k2=config.apex_k2,
                alpha=config.apex_alpha,
                gamma_w=config.apex_gamma_w,
                gamma_wd=config.apex_gamma_wd,
                gamma_j=config.apex_gamma_j,
                warmup_secs=config.apex_warmup_secs,
                max_correction_deg=config.apex_max_correction_deg,
            ),
        )
        if config.apex:
            self.logger.warning(
                "APEX ON: K1=%.1f K2=%.1f alpha=%.1f  gamma=(%.1e,%.1e,%.1e)  "
                "warmup=%.1fs  |correction|<=%.1f deg  (gripper excluded)",
                config.apex_k1, config.apex_k2, config.apex_alpha,
                config.apex_gamma_w, config.apex_gamma_wd, config.apex_gamma_j,
                config.apex_warmup_secs, config.apex_max_correction_deg,
            )

        self.reanchor = ChunkReanchor(
            SO101_ACTION_NAMES,
            ReanchorConfig(enabled=config.reanchor, blend_steps=config.reanchor_blend_steps),
        )
        if config.reanchor:
            self.logger.warning(
                "REANCHOR ON: each chunk shifted onto the measured pose, decaying over %d actions",
                config.reanchor_blend_steps,
            )

        self._last_load: np.ndarray | None = None
        if config.feedback:
            self.logger.warning(
                "FEEDBACK ON: slew=%.1f deg  lead=%.1f deg (gripper %.1f)  "
                "stall=|load|>=%.0f for %.1fs  fixed-point=%.1fs/%.1fdeg -> open %.0f deg",
                config.fb_max_step_deg, config.fb_max_lead_deg, config.fb_gripper_lead_deg,
                config.fb_stall_load, config.fb_stall_secs,
                config.fb_fixed_point_secs, config.fb_fixed_point_move_deg,
                config.fb_fixed_point_release_deg,
            )

    # ---------------- write guards ----------------

    def _install_write_guards(self) -> None:
        robot = self.robot

        def _blocked_send_action(action, *_a, **_kw):
            raise MotorWriteBlocked(f"send_action() during a shadow run (action={action}); no write performed.")

        robot.send_action = _blocked_send_action  # type: ignore[method-assign]

        bus = robot.bus
        original_sync_write = bus.sync_write

        def _guarded_sync_write(data_name, *a, **kw):
            if data_name in BLOCKED_REGISTERS:
                raise MotorWriteBlocked(f"Refusing to sync_write {data_name!r}; the motor bus is read-only.")
            return original_sync_write(data_name, *a, **kw)

        bus.sync_write = _guarded_sync_write  # type: ignore[method-assign]

    # ---------------- screenshots ----------------

    def _screenshot_worker(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.logger.warning("PIL unavailable; screenshots disabled")
            return
        while not self._shot_stop.is_set():
            try:
                item = self._shot_q.get(timeout=0.25)
            except queue.Empty:
                continue
            if item is None:
                break
            timestep, wall, frames = item
            for name, arr in frames.items():
                try:
                    Image.fromarray(arr).save(
                        self.shot_dir / f"t{timestep:06d}_{wall:.3f}_{name}.jpg",
                        quality=self.config.screenshot_quality,
                    )
                except Exception as exc:  # noqa: BLE001 - a bad frame must not kill the run
                    self.logger.debug("screenshot failed for %s: %s", name, exc)
            self._n_shots += 1

    def _maybe_queue_screenshot(self, raw_observation: dict, timestep: int) -> None:
        if self._shot_period is None:
            return
        now = time.time()
        if now - self._last_shot < self._shot_period:
            return
        frames = {}
        for key, value in raw_observation.items():
            arr = getattr(value, "shape", None)
            if arr is not None and len(value.shape) == 3 and value.shape[-1] == 3:
                # Copy: the camera buffer is reused by the next capture.
                frames[str(key)] = value.copy()
        if not frames:
            return
        try:
            self._shot_q.put_nowait((timestep, now, frames))
            self._last_shot = now
        except queue.Full:
            # Saver is behind; skip this one rather than block the control loop.
            pass

    # ---------------- instrumented loops ----------------

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        get_start = time.perf_counter()
        with self.action_queue_lock:
            queue_size_before = self.action_queue.qsize()
            self.action_queue_size.append(queue_size_before)
            timed_action = self.action_queue.get_nowait()
            queue_size_after = self.action_queue.qsize()
        queue_pop_ms = (time.perf_counter() - get_start) * 1000

        commanded = self._action_tensor_to_action_dict(timed_action.get_action())
        policy_commanded = dict(commanded)  # keep the unmodified policy output for the trace

        exec_start = time.time()
        send_start = time.perf_counter()
        # send_action itself reads Present_Position when max_relative_target is
        # set; read it explicitly so the trace has it either way -- and because
        # the feedback guards need the live measurement before the write.
        present = self._read_present_position()
        if present and (self.reanchor.cfg.enabled or self.apex.cfg.enabled or self.feedback.cfg.enabled):
            cmd_vec = np.array([commanded[j] for j in SO101_ACTION_NAMES], dtype=np.float64)
            pres_vec = np.array([present[j.removesuffix(".pos")] for j in SO101_ACTION_NAMES], dtype=np.float64)
            # APEX first: it reconstructs the reference the servo loop is
            # missing and adds lead. The guards run after it, on its output,
            # because their job is to bound whatever finally reaches the bus --
            # including APEX's own correction if adaptation misbehaves.
            # Re-anchor before anything else: it removes the stale-observation
            # offset the chunk arrived with, so every guard downstream sees the
            # command the policy meant for *now* rather than for 415 ms ago.
            # The chunk is identified by the observation it was computed from,
            # not by the action's own timestep -- that increments every action,
            # which would make every step look like a chunk boundary and pin
            # the command onto the measurement forever.
            _m = timed_action.get_metadata()
            _src = _m.get("source_observation_timestep") if isinstance(_m, dict) else None
            cmd_vec = self.reanchor.apply(cmd_vec, pres_vec, _src)
            cmd_vec = self.apex.step(cmd_vec, pres_vec, exec_start)
            adjusted = self.feedback.apply(cmd_vec, pres_vec, self._last_load, exec_start)
            commanded = {j: float(v) for j, v in zip(SO101_ACTION_NAMES, adjusted)}
        performed = None if self.shadow else self.robot.send_action(commanded)
        send_ms = (time.perf_counter() - send_start) * 1000
        exec_end = time.time()

        deltas = None
        if present:
            deltas = {
                k: commanded[k] - present[k.removesuffix(".pos")]
                for k in commanded
                if k.removesuffix(".pos") in present
            }

        # Actuator state, sampled sparsely: tracking error only becomes
        # actionable once you can tell "the policy asked for little" from
        # "the servo could not deliver".
        actuator = None
        every = self.config.actuator_sample_every
        if every and self._n_actions % every == 0:
            actuator = self._read_actuator_state()
            if actuator and "load" in actuator:
                self._last_load = np.array(
                    [abs(actuator["load"].get(j.removesuffix(".pos"), 0)) for j in SO101_ACTION_NAMES],
                    dtype=np.float64,
                )

        meta = timed_action.get_metadata()
        server_meta = {}
        src_ts = src_step = None
        if isinstance(meta, dict):
            src_ts = meta.get("source_observation_timestamp")
            src_step = meta.get("source_observation_timestep")
            server_meta = {
                k: meta[k]
                for k in (
                    "stages_ms",
                    "proprio",
                    "raw_model_action_first",
                    "raw_abs_max_delta",
                    "num_denoising_steps_action",
                    "dry_run_zero_actions",
                    "server_timestamp",
                )
                if k in meta
            }

        record = {
            "timestep": timed_action.get_timestep(),
            "action_timestamp": timed_action.get_timestamp(),
            "exec_start": exec_start,
            "exec_end": exec_end,
            "action_age_ms": (exec_start - timed_action.get_timestamp()) * 1000,
            "source_observation_timestep": src_step,
            "source_observation_timestamp": src_ts,
            "observation_to_execution_ms": ((exec_start - src_ts) * 1000 if src_ts else None),
            "queue_pop_ms": queue_pop_ms,
            "robot_send_action_ms": send_ms,
            "queue_size_before": queue_size_before,
            "queue_size_after": queue_size_after,
            "commanded_action": commanded,
            "performed_action": performed,
            "present_position": ({f"{k}.pos": v for k, v in present.items()} if present else None),
            "delta_from_present": deltas,
            "max_abs_delta": (max(abs(v) for v in deltas.values()) if deltas else None),
            "written_to_motor_bus": not self.shadow,
            "actuator": actuator,
            # What the policy asked for before the guards, so their effect is
            # measurable rather than invisible.
            "policy_commanded_action": policy_commanded,
            "server": server_meta,
        }
        with self._trace_lock:
            self._action_trace.write(json.dumps(record) + "\n")
            self._n_actions += 1

        self.latency_recorder.record(
            "client_execute_action",
            timestep=timed_action.get_timestep(),
            queue_pop_ms=queue_pop_ms,
            robot_send_action_ms=send_ms,
            action_age_ms=(time.time() - timed_action.get_timestamp()) * 1000,
            total_ms=queue_pop_ms + send_ms,
        )
        self._record_timeline(
            "timeline_action",
            timestep=timed_action.get_timestep(),
            start_time=exec_start,
            end_time=exec_end,
            action_timestamp=timed_action.get_timestamp(),
            source_observation_timestamp=src_ts,
            source_observation_timestep=src_step,
            queue_size_before=queue_size_before,
            queue_size_after=queue_size_after,
            commanded_action=commanded,
            performed_action=performed,
        )
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()
            self.latest_action_wall_time = exec_end
        return performed or {}

    def _read_actuator_state(self):
        out = {}
        for reg, key in (("Present_Load", "load"), ("Present_Temperature", "temp_c")):
            try:
                out[key] = self.robot.bus.sync_read(reg)
            except Exception as exc:  # noqa: BLE001 - never abort a run over telemetry
                self.logger.debug("%s read failed: %s", reg, exc)
        return out or None

    def _read_present_position(self):
        try:
            return self.robot.bus.sync_read("Present_Position", num_retry=3)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Present_Position read failed: %s", exc)
            return None

    def control_loop_observation(self, task: str, verbose: bool = False):
        t0 = time.perf_counter()
        raw = super().control_loop_observation(task, verbose)
        total_ms = (time.perf_counter() - t0) * 1000
        if raw is not None:
            with self.action_queue_lock:
                qsize = self.action_queue.qsize()
            with self._trace_lock:
                self._obs_trace.write(
                    json.dumps(
                        {
                            "wall_time": time.time(),
                            "capture_plus_send_ms": total_ms,
                            "queue_size": qsize,
                            "n": self._n_obs,
                        }
                    )
                    + "\n"
                )
                self._n_obs += 1
            self._maybe_queue_screenshot(raw, self._n_obs)
        return raw

    def stop(self) -> None:
        super().stop()
        self._shot_stop.set()
        try:
            self._shot_q.put_nowait(None)
        except queue.Full:
            pass
        if self._shot_thread is not None:
            self._shot_thread.join(timeout=5)
        for f in (self._action_trace, self._obs_trace):
            try:
                f.close()
            except Exception:  # noqa: BLE001
                pass
        self.logger.warning(
            "Run complete: %d actions (%s), %d observations, %d screenshot sets -> %s",
            self._n_actions,
            "written to bus" if not self.shadow else "0 written to bus",
            self._n_obs,
            self._n_shots,
            self.trace_dir,
        )
        if self.reanchor.cfg.enabled:
            self.logger.warning("REANCHOR: %s", self.reanchor.stats.as_dict())
        if self.apex.cfg.enabled:
            self.apex.finalize()
            self.logger.warning("APEX: %s", self.apex.stats.as_dict())
        if self.feedback.cfg.enabled:
            s = self.feedback.stats
            n = max(s.n_commands, 1)
            self.logger.warning(
                "FEEDBACK: slew clipped %d/%d (%.1f%%, worst %.2f deg) | lead clipped %d (%.1f%%, worst %.2f deg) "
                "| stall freezes %d %s",
                s.n_slew_clipped, s.n_commands, 100 * s.n_slew_clipped / n, s.max_slew_clip_deg,
                s.n_lead_clipped, 100 * s.n_lead_clipped / n, s.max_lead_clip_deg,
                s.n_stall_frozen, s.stalled_joints or "",
            )
            self.logger.warning(
                "FEEDBACK: fixed points detected %d at t=%s (%d release commands issued)",
                s.n_fixed_point, s.fixed_point_at or "-", s.n_release_commands,
            )
            if self.feedback.autotuned:
                self.logger.warning("FEEDBACK autotuned -> %s", self.feedback.autotuned)
            try:
                payload = s.as_dict()
                payload["autotuned"] = self.feedback.autotuned
                (self.trace_dir / "feedback_stats.json").write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False)
                )
            except Exception:  # noqa: BLE001
                pass


@draccus.wrap()
def profiled_client(cfg: ProfiledClientConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    client = ProfiledRobotClient(cfg)
    if not client.start():
        client.stop()
        return

    client.logger.info("Starting action receiver thread...")
    receiver = threading.Thread(target=client.receive_actions, daemon=True)
    receiver.start()
    try:
        client.control_loop(task=cfg.task)
    finally:
        client.stop()
        receiver.join()
        client.logger.info("Client stopped")


if __name__ == "__main__":
    profiled_client()
