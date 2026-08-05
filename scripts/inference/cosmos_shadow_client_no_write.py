#!/usr/bin/env python
"""HIL shadow client for the Cosmos SO101 policy: reads hardware, never writes it.

This is the stock LeRobot ``robot_client`` control loop with exactly one
behavioural change: the action popped from the queue is *recorded and analysed*
instead of being handed to ``robot.send_action()``.  The serial port is opened
read-only in practice -- observations (joint positions + cameras) are streamed
to the policy server as usual, so the policy sees genuine hardware state, but no
goal position ever reaches the motor bus.

Three independent layers enforce that:

1. ``control_loop_action`` below never calls ``robot.send_action``.
2. ``robot.send_action`` is replaced by a raiser, so any code path that still
   tries to command the arm fails loudly instead of moving it.
3. ``bus.sync_write`` is wrapped with a tripwire that refuses ``Goal_Position``
   (and ``Goal_Current`` / ``Goal_Velocity``) writes outright.

Layer 3 is the one that actually touches the wire, so it is the guarantee; 1
and 2 exist so that a violation is reported as a bug rather than silently
clamped.

Every popped action is compared against the joint positions measured at that
instant -- the same ``sync_read("Present_Position")`` that ``send_action`` would
have performed for ``max_relative_target`` -- and written to a JSONL file for
offline drivability analysis.
"""

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import draccus

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient
from lerobot.utils.utils import init_logging

# Order the Cosmos policy server emits actions in.  The client maps the action
# tensor onto ``robot.action_features`` positionally, so a mismatch here would
# silently drive the wrong joints.
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
class ShadowClientConfig(RobotClientConfig):
    """Robot client config plus the shadow-run recording path."""

    shadow_jsonl_path: str = field(
        default="",
        metadata={"help": "Where to write per-action shadow records (JSONL). Empty = derive from timeline dir."},
    )


class NoWriteShadowRobotClient(RobotClient):
    prefix = "cosmos_shadow_client"
    logger = logging.getLogger("cosmos_shadow_client")

    def __init__(self, config: ShadowClientConfig):
        super().__init__(config)

        self._assert_action_feature_order()
        self._install_write_guards()

        path = config.shadow_jsonl_path or str(
            Path(self.latency_recorder.log_dir) / f"shadow_actions_{self.latency_recorder.run_id}.jsonl"
        )
        self.shadow_path = Path(path)
        self.shadow_path.parent.mkdir(parents=True, exist_ok=True)
        self._shadow_file = self.shadow_path.open("w", buffering=1)
        self._shadow_lock = threading.Lock()
        self._shadow_count = 0
        self.logger.warning("SHADOW MODE: no goal position will be written to %s", config.robot.port)
        self.logger.warning("SHADOW MODE: recording candidate actions to %s", self.shadow_path)

    def _assert_action_feature_order(self) -> None:
        """A positional mismatch would drive the wrong joint; fail before moving anything."""
        features = list(self.robot.action_features)
        if features != SO101_ACTION_NAMES:
            raise RuntimeError(
                "Robot action_features order does not match the Cosmos server joint order.\n"
                f"  robot.action_features = {features}\n"
                f"  cosmos joint_order    = {SO101_ACTION_NAMES}\n"
                "Actions are mapped positionally, so this would command the wrong joints."
            )
        self.logger.info("Verified action feature order matches Cosmos joint order: %s", features)

    def _install_write_guards(self) -> None:
        robot = self.robot

        def _blocked_send_action(action, *_args, **_kwargs):
            raise MotorWriteBlocked(
                f"send_action() called during a shadow run (action={action}). No write was performed."
            )

        robot.send_action = _blocked_send_action  # type: ignore[method-assign]

        bus = robot.bus
        original_sync_write = bus.sync_write

        def _guarded_sync_write(data_name, *args, **kwargs):
            if data_name in BLOCKED_REGISTERS:
                raise MotorWriteBlocked(
                    f"Refusing to sync_write {data_name!r} during a shadow run; the motor bus is read-only."
                )
            return original_sync_write(data_name, *args, **kwargs)

        bus.sync_write = _guarded_sync_write  # type: ignore[method-assign]
        self.logger.info("Motor-bus write guard installed; blocked registers: %s", sorted(BLOCKED_REGISTERS))

    def _read_present_position(self) -> dict[str, float] | None:
        """The same read ``send_action`` performs for max_relative_target -- reads are allowed."""
        try:
            return self.robot.bus.sync_read("Present_Position", num_retry=3)
        except Exception as exc:  # noqa: BLE001 - a failed read must not abort the shadow run
            self.logger.warning("Present_Position read failed: %s", exc)
            return None

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """Pop the action the policy wants to execute, analyse it, and discard it."""
        get_start = time.perf_counter()
        with self.action_queue_lock:
            queue_size_before_pop = self.action_queue.qsize()
            self.action_queue_size.append(queue_size_before_pop)
            timed_action = self.action_queue.get_nowait()
            queue_size_after_pop = self.action_queue.qsize()
        get_end = time.perf_counter() - get_start

        execution_start_time = time.time()
        commanded_action = self._action_tensor_to_action_dict(timed_action.get_action())
        present = self._read_present_position()
        execution_end_time = time.time()

        # Deltas the arm *would* have been asked to make, per joint, in physical
        # units (degrees for the five body joints, 0-100 range for the gripper).
        deltas = None
        if present is not None:
            deltas = {
                key: commanded_action[key] - present[key.removesuffix(".pos")]
                for key in commanded_action
                if key.removesuffix(".pos") in present
            }

        action_metadata = timed_action.get_metadata()
        source_observation_timestamp = None
        source_observation_timestep = None
        server_meta: dict[str, Any] = {}
        if isinstance(action_metadata, dict):
            source_observation_timestamp = action_metadata.get("source_observation_timestamp")
            source_observation_timestep = action_metadata.get("source_observation_timestep")
            server_meta = {
                key: action_metadata[key]
                for key in (
                    "joint_order",
                    "proprio",
                    "raw_model_action_first",
                    "raw_model_action_last",
                    "raw_abs_max_delta",
                    "dry_run_zero_actions",
                    "num_denoising_steps_action",
                )
                if key in action_metadata
            }

        record = {
            "timestep": timed_action.get_timestep(),
            "action_timestamp": timed_action.get_timestamp(),
            "wall_time": execution_start_time,
            "action_age_ms": (execution_start_time - timed_action.get_timestamp()) * 1000,
            "source_observation_timestep": source_observation_timestep,
            "source_observation_timestamp": source_observation_timestamp,
            "commanded_action": commanded_action,
            "present_position": {f"{k}.pos": v for k, v in present.items()} if present else None,
            "delta_from_present": deltas,
            "max_abs_delta": max(abs(v) for v in deltas.values()) if deltas else None,
            "queue_size_before": queue_size_before_pop,
            "written_to_motor_bus": False,
            "server": server_meta,
        }
        with self._shadow_lock:
            self._shadow_file.write(json.dumps(record) + "\n")
            self._shadow_count += 1

        self.latency_recorder.record(
            "client_execute_action",
            timestep=timed_action.get_timestep(),
            queue_pop_ms=get_end * 1000,
            robot_send_action_ms=0.0,
            action_age_ms=(time.time() - timed_action.get_timestamp()) * 1000,
            total_ms=(get_end + (execution_end_time - execution_start_time)) * 1000,
        )
        self._record_timeline(
            "timeline_action",
            timestep=timed_action.get_timestep(),
            start_time=execution_start_time,
            end_time=execution_end_time,
            action_timestamp=timed_action.get_timestamp(),
            source_observation_timestamp=source_observation_timestamp,
            source_observation_timestep=source_observation_timestep,
            queue_size_before=queue_size_before_pop,
            queue_size_after=queue_size_after_pop,
            commanded_action=commanded_action,
            performed_action=None,
        )

        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()
            self.latest_action_wall_time = execution_end_time

        if verbose and deltas:
            self.logger.info(
                "shadow action #%s | max|delta|=%.2f | %s",
                timed_action.get_timestep(),
                record["max_abs_delta"],
                {k: round(v, 2) for k, v in deltas.items()},
            )

        # Nothing was performed: the motor bus was never written.
        return {}

    def stop(self) -> None:
        super().stop()
        try:
            self._shadow_file.close()
        except Exception:  # noqa: BLE001
            pass
        self.logger.warning(
            "SHADOW MODE complete: %d candidate actions recorded, 0 written to the motor bus (%s)",
            self._shadow_count,
            self.shadow_path,
        )


@draccus.wrap()
def shadow_client(cfg: ShadowClientConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    client = NoWriteShadowRobotClient(cfg)

    if not client.start():
        client.stop()
        return

    client.logger.info("Starting action receiver thread...")
    action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
    action_receiver_thread.start()
    try:
        client.control_loop(task=cfg.task, verbose=True)
    finally:
        client.stop()
        action_receiver_thread.join()
        client.logger.info("Shadow client stopped")


if __name__ == "__main__":
    shadow_client()
