# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Building blocks for fully asynchronous inference.

This module is intentionally separate from the existing threshold-triggered
async client/server. It provides latest-only observation publishing, structured
event logging, and a thread-safe action queue that only mutates future actions.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

import torch

from .helpers import TimedAction, TimedObservation

ContinuousAggregationName = Literal[
    "latency_aligned_blend",
    "replace_remaining",
    "splice_by_timestamp",
    "smooth_blend_remaining",
    "conservative_update",
]


def monotonic_now() -> float:
    return time.monotonic()


def wall_now() -> float:
    return time.time()


def _json_default(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _action_norm(action: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(action.detach().float().cpu()))


def _action_max_abs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a.detach().float().cpu() - b.detach().float().cpu())))


@dataclass
class StructuredEventLogger:
    """Thread-safe JSONL event logger for continuous async inference."""

    path: str | Path | None
    enabled: bool = True
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _file: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not self.enabled or self.path is None:
            return
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def record(self, event_type: str, **fields: Any) -> dict[str, Any]:
        event = {
            "event_type": event_type,
            "monotonic_ts": monotonic_now(),
            "wall_time": wall_now(),
            "obs_id": None,
            "inference_id": None,
            "action_id": None,
            "source_obs_id": None,
            "source_inference_id": None,
            **fields,
        }
        if not self.enabled:
            return event
        with self._lock:
            if self._file is not None and not self._file.closed:
                self._file.write(json.dumps(event, default=_json_default) + "\n")
                self._file.flush()
        return event

    def close(self) -> None:
        with self._lock:
            if self._file is not None and not self._file.closed:
                self._file.close()


@dataclass
class ContinuousAsyncConfig:
    async_mode: str = "continuous"
    continuous_obs_fps: float = 30.0
    control_fps: float = 30.0
    continuous_inference_workers: int = 1
    aggregation_fn: ContinuousAggregationName = "latency_aligned_blend"
    max_pending_observations: int = 1
    stale_inference_max_age: float = 2.0
    min_usable_actions: int = 5
    blend_horizon: int = 5
    blend_alpha: float = 0.5
    max_joint_delta: float | None = None
    max_gripper_delta: float | None = None
    max_joint_delta_per_step: float | None = None
    max_joint_abs_range: float | None = None
    max_gripper_delta_per_step: float | None = None
    emergency_stop: bool = False
    shadow_mode: bool = True
    enable_robot_execution: bool = False
    timeline_log_path: str | None = None
    timeline_plot_path: str | None = None

    @property
    def control_dt(self) -> float:
        return 1.0 / self.control_fps


@dataclass
class InferenceResult:
    inference_id: int
    source_obs_id: int
    source_obs_capture_ts: float
    source_obs_capture_wall_time: float | None
    inference_start_ts: float
    inference_end_ts: float
    generated_action_chunk: list[torch.Tensor]
    action_chunk_size: int
    model_latency: float
    server_queue_delay: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionQueueItem:
    action_id: int
    action: torch.Tensor
    source_obs_id: int
    source_inference_id: int
    chunk_index: int
    predicted_for_step_index: int
    enqueue_ts: float
    planned_exec_ts: float | None = None
    actual_exec_start_ts: float | None = None
    actual_exec_end_ts: float | None = None
    status: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_event_fields(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "source_obs_id": self.source_obs_id,
            "source_inference_id": self.source_inference_id,
            "chunk_index": self.chunk_index,
            "predicted_for_step_index": self.predicted_for_step_index,
            "enqueue_ts": self.enqueue_ts,
            "planned_exec_ts": self.planned_exec_ts,
            "actual_exec_start_ts": self.actual_exec_start_ts,
            "actual_exec_end_ts": self.actual_exec_end_ts,
            "status": self.status,
            "action_values": self.action,
            **self.metadata,
        }


@dataclass
class QueueUpdateStats:
    applied: bool
    aggregation_fn: str
    old_queue_length: int
    new_queue_length: int
    num_actions_replaced: int = 0
    num_actions_blended: int = 0
    num_actions_kept: int = 0
    num_actions_dropped: int = 0
    first_pending_action_age: float | None = None
    source_obs_age: float | None = None
    update_reason: str = "inference_result"
    discard_reason: str | None = None
    mean_action_jump_before_update: float | None = None
    mean_action_jump_after_update: float | None = None


class AsyncActionQueueManager:
    """Thread-safe queue manager for continuous inference results.

    Only actions whose status is ``pending`` are replaced or blended. An action
    that has been popped for execution is marked ``executing`` before the lock
    is released, so queue updates can never overwrite it.
    """

    def __init__(self, config: ContinuousAsyncConfig, event_logger: StructuredEventLogger | None = None):
        self.config = config
        self.event_logger = event_logger or StructuredEventLogger(None, enabled=False)
        self._lock = threading.RLock()
        self._pending: list[ActionQueueItem] = []
        self._history: list[ActionQueueItem] = []
        self._next_action_id = 0
        self._control_step_index = 0
        self.rejected_update_count = 0

    @property
    def lock(self) -> threading.RLock:
        return self._lock

    def qsize(self) -> int:
        with self._lock:
            return len(self._pending)

    def snapshot(self) -> list[ActionQueueItem]:
        with self._lock:
            return list(self._pending)

    def _new_items(
        self,
        result: InferenceResult,
        start_chunk_index: int = 0,
        planned_start_ts: float | None = None,
    ) -> list[ActionQueueItem]:
        items = []
        now = monotonic_now()
        planned_start_ts = now if planned_start_ts is None else planned_start_ts
        chunk = result.generated_action_chunk[start_chunk_index:]
        for chunk_index, action in enumerate(chunk, start_chunk_index):
            action_tensor = (
                action.detach().clone().cpu() if isinstance(action, torch.Tensor) else torch.tensor(action)
            )
            item = ActionQueueItem(
                action_id=self._next_action_id,
                action=action_tensor,
                source_obs_id=result.source_obs_id,
                source_inference_id=result.inference_id,
                chunk_index=chunk_index,
                predicted_for_step_index=self._control_step_index + len(items) + 1,
                enqueue_ts=now,
                planned_exec_ts=planned_start_ts + len(items) * self.config.control_dt,
            )
            self._next_action_id += 1
            items.append(item)
        return items

    def _validate_result(self, result: InferenceResult) -> str | None:
        if self.config.emergency_stop:
            return "emergency_stop"
        if monotonic_now() - result.source_obs_capture_ts > self.config.stale_inference_max_age:
            return "stale_inference"
        for action in result.generated_action_chunk:
            tensor = (
                action.detach().float().cpu() if isinstance(action, torch.Tensor) else torch.tensor(action)
            )
            if torch.isnan(tensor).any():
                return "action_nan"
            if torch.isinf(tensor).any():
                return "action_inf"
            if self.config.max_joint_abs_range is not None and float(torch.max(torch.abs(tensor))) > float(
                self.config.max_joint_abs_range
            ):
                return "action_out_of_abs_range"
        return None

    def _discard(
        self,
        result: InferenceResult,
        reason: str,
        old_queue_length: int,
        source_obs_age: float | None,
    ) -> QueueUpdateStats:
        self.rejected_update_count += 1
        self.event_logger.record(
            "inference_discarded",
            inference_id=result.inference_id,
            source_obs_id=result.source_obs_id,
            discard_reason=reason,
            source_obs_age=source_obs_age,
        )
        return QueueUpdateStats(
            applied=False,
            aggregation_fn=self.config.aggregation_fn,
            old_queue_length=old_queue_length,
            new_queue_length=old_queue_length,
            source_obs_age=source_obs_age,
            discard_reason=reason,
        )

    def apply_inference_result(self, result: InferenceResult) -> QueueUpdateStats:
        queue_update_ts = monotonic_now()
        source_obs_age = queue_update_ts - result.source_obs_capture_ts
        self.event_logger.record(
            "queue_update_started",
            inference_id=result.inference_id,
            source_obs_id=result.source_obs_id,
            aggregation_fn=self.config.aggregation_fn,
            source_obs_age=source_obs_age,
        )

        with self._lock:
            old_pending = list(self._pending)
            old_queue_length = len(old_pending)
            first_pending_action_age = (
                queue_update_ts - old_pending[0].enqueue_ts if old_pending else None
            )
            discard_reason = self._validate_result(result)
            if discard_reason is not None:
                stats = self._discard(result, discard_reason, old_queue_length, source_obs_age)
                self._finish_update_event(result, stats, first_pending_action_age)
                return stats

            if self.config.aggregation_fn == "latency_aligned_blend":
                stats = self._latency_aligned_blend(
                    result, old_pending, first_pending_action_age, source_obs_age
                )
            elif self.config.aggregation_fn == "replace_remaining":
                stats = self._replace_remaining(result, old_pending, first_pending_action_age, source_obs_age)
            elif self.config.aggregation_fn == "splice_by_timestamp":
                stats = self._splice_by_timestamp(
                    result, old_pending, first_pending_action_age, source_obs_age
                )
            elif self.config.aggregation_fn == "smooth_blend_remaining":
                stats = self._smooth_blend_remaining(
                    result, old_pending, first_pending_action_age, source_obs_age
                )
            elif self.config.aggregation_fn == "conservative_update":
                stats = self._conservative_update(
                    result, old_pending, first_pending_action_age, source_obs_age
                )
            else:
                stats = self._discard(
                    result,
                    f"unknown_aggregation_fn:{self.config.aggregation_fn}",
                    old_queue_length,
                    source_obs_age,
                )

            self._finish_update_event(result, stats, first_pending_action_age)
            return stats

    def _mark_replaced(self, old_pending: list[ActionQueueItem], result: InferenceResult) -> None:
        for item in old_pending:
            item.status = "replaced"
            item.metadata["replaced_by_inference_id"] = result.inference_id
            self.event_logger.record(
                "action_replaced",
                **item.to_event_fields(),
                replacement_inference_id=result.inference_id,
            )
        self._history.extend(old_pending)

    def _replace_remaining(
        self,
        result: InferenceResult,
        old_pending: list[ActionQueueItem],
        first_pending_action_age: float | None,
        source_obs_age: float | None,
    ) -> QueueUpdateStats:
        self._mark_replaced(old_pending, result)
        self._pending = self._new_items(result)
        return QueueUpdateStats(
            applied=True,
            aggregation_fn="replace_remaining",
            old_queue_length=len(old_pending),
            new_queue_length=len(self._pending),
            num_actions_replaced=len(old_pending),
            num_actions_kept=0,
            first_pending_action_age=first_pending_action_age,
            source_obs_age=source_obs_age,
            update_reason="replace_all_pending",
        )

    def _splice_by_timestamp(
        self,
        result: InferenceResult,
        old_pending: list[ActionQueueItem],
        first_pending_action_age: float | None,
        source_obs_age: float | None,
    ) -> QueueUpdateStats:
        estimated_action_index = int(
            round((monotonic_now() - result.source_obs_capture_ts) * self.config.control_fps)
        )
        if estimated_action_index >= result.action_chunk_size:
            return self._discard(result, "splice_index_out_of_chunk", len(old_pending), source_obs_age)
        self._mark_replaced(old_pending, result)
        self._pending = self._new_items(result, start_chunk_index=max(0, estimated_action_index))
        return QueueUpdateStats(
            applied=True,
            aggregation_fn="splice_by_timestamp",
            old_queue_length=len(old_pending),
            new_queue_length=len(self._pending),
            num_actions_replaced=len(old_pending),
            num_actions_dropped=estimated_action_index,
            first_pending_action_age=first_pending_action_age,
            source_obs_age=source_obs_age,
            update_reason=f"splice_from_chunk_index:{estimated_action_index}",
        )

    def _smooth_blend_remaining(
        self,
        result: InferenceResult,
        old_pending: list[ActionQueueItem],
        first_pending_action_age: float | None,
        source_obs_age: float | None,
    ) -> QueueUpdateStats:
        new_items = self._new_items(result)
        blend_count = min(self.config.blend_horizon, len(old_pending), len(new_items))
        jumps_before = []
        jumps_after = []
        for i in range(blend_count):
            old_action = old_pending[i].action
            new_action = new_items[i].action
            blended = self.config.blend_alpha * old_action + (1.0 - self.config.blend_alpha) * new_action
            jumps_before.append(_action_max_abs_delta(old_action, new_action))
            jumps_after.append(_action_max_abs_delta(old_action, blended))
            new_items[i].action = blended
            new_items[i].metadata["blended"] = True
            new_items[i].metadata["blend_alpha"] = self.config.blend_alpha
            new_items[i].metadata["action_jump_before_update"] = jumps_before[-1]
            new_items[i].metadata["action_jump_after_update"] = jumps_after[-1]
        self._mark_replaced(old_pending, result)
        self._pending = new_items
        return QueueUpdateStats(
            applied=True,
            aggregation_fn="smooth_blend_remaining",
            old_queue_length=len(old_pending),
            new_queue_length=len(self._pending),
            num_actions_replaced=len(old_pending),
            num_actions_blended=blend_count,
            first_pending_action_age=first_pending_action_age,
            source_obs_age=source_obs_age,
            update_reason=f"blend_horizon:{blend_count}",
            mean_action_jump_before_update=sum(jumps_before) / len(jumps_before) if jumps_before else None,
            mean_action_jump_after_update=sum(jumps_after) / len(jumps_after) if jumps_after else None,
        )

    def _latency_aligned_blend(
        self,
        result: InferenceResult,
        old_pending: list[ActionQueueItem],
        first_pending_action_age: float | None,
        source_obs_age: float | None,
    ) -> QueueUpdateStats:
        # source_obs_capture_ts must have been rewritten by the client to its own
        # local monotonic clock before this method is called. Server monotonic
        # timestamps are not comparable across processes or machines.
        offset = int(round((monotonic_now() - result.source_obs_capture_ts) * self.config.control_fps))
        offset = max(0, offset)
        min_usable = max(0, int(self.config.min_usable_actions))
        if offset >= result.action_chunk_size - min_usable:
            return self._discard(
                result, f"chunk_too_stale:offset={offset}", len(old_pending), source_obs_age
            )

        new_items = self._new_items(result, start_chunk_index=offset)
        blend_count = min(self.config.blend_horizon, len(old_pending), len(new_items))
        jumps_before = []
        jumps_after = []
        for i in range(blend_count):
            w_old = self.config.blend_alpha * (1.0 - (i + 1) / (blend_count + 1))
            old_action = old_pending[i].action
            new_action = new_items[i].action
            blended = w_old * old_action + (1.0 - w_old) * new_action
            jumps_before.append(_action_max_abs_delta(old_action, new_action))
            jumps_after.append(_action_max_abs_delta(old_action, blended))
            new_items[i].action = blended
            new_items[i].metadata.update(
                {
                    "blended": True,
                    "w_old": w_old,
                    "latency_offset_steps": offset,
                    "action_jump_before_update": jumps_before[-1],
                    "action_jump_after_update": jumps_after[-1],
                }
            )

        self._mark_replaced(old_pending, result)
        self._pending = new_items
        return QueueUpdateStats(
            applied=True,
            aggregation_fn="latency_aligned_blend",
            old_queue_length=len(old_pending),
            new_queue_length=len(self._pending),
            num_actions_replaced=len(old_pending),
            num_actions_blended=blend_count,
            num_actions_dropped=offset,
            first_pending_action_age=first_pending_action_age,
            source_obs_age=source_obs_age,
            update_reason=f"offset={offset},blend={blend_count}",
            mean_action_jump_before_update=sum(jumps_before) / len(jumps_before) if jumps_before else None,
            mean_action_jump_after_update=sum(jumps_after) / len(jumps_after) if jumps_after else None,
        )

    def _conservative_update(
        self,
        result: InferenceResult,
        old_pending: list[ActionQueueItem],
        first_pending_action_age: float | None,
        source_obs_age: float | None,
    ) -> QueueUpdateStats:
        if old_pending and result.generated_action_chunk:
            new_first = result.generated_action_chunk[0].detach().float().cpu()
            old_first = old_pending[0].action.detach().float().cpu()
            max_delta = _action_max_abs_delta(old_first, new_first)
            if self.config.max_joint_delta is not None and max_delta > self.config.max_joint_delta:
                return self._discard(
                    result, f"max_joint_delta_exceeded:{max_delta:.6f}", len(old_pending), source_obs_age
                )
            if (
                self.config.max_gripper_delta is not None
                and old_first.numel() > 0
                and abs(float(old_first[-1] - new_first[-1])) > self.config.max_gripper_delta
            ):
                return self._discard(result, "max_gripper_delta_exceeded", len(old_pending), source_obs_age)
        return self._replace_remaining(result, old_pending, first_pending_action_age, source_obs_age)

    def _finish_update_event(
        self,
        result: InferenceResult,
        stats: QueueUpdateStats,
        first_pending_action_age: float | None,
    ) -> None:
        self.event_logger.record(
            "queue_update_finished",
            inference_id=result.inference_id,
            source_obs_id=result.source_obs_id,
            aggregation_fn=stats.aggregation_fn,
            old_queue_length=stats.old_queue_length,
            new_queue_length=stats.new_queue_length,
            num_actions_replaced=stats.num_actions_replaced,
            num_actions_blended=stats.num_actions_blended,
            num_actions_kept=stats.num_actions_kept,
            num_actions_dropped=stats.num_actions_dropped,
            first_pending_action_age=first_pending_action_age,
            source_obs_age=stats.source_obs_age,
            update_reason=stats.update_reason,
            discard_reason=stats.discard_reason,
            mean_action_jump_before_update=stats.mean_action_jump_before_update,
            mean_action_jump_after_update=stats.mean_action_jump_after_update,
        )

    def pop_next_action(self) -> ActionQueueItem | None:
        with self._lock:
            if not self._pending:
                return None
            item = self._pending.pop(0)
            item.status = "executing"
            item.actual_exec_start_ts = monotonic_now()
            return item

    def finish_action_execution(self, item: ActionQueueItem, performed_action: Any = None) -> None:
        with self._lock:
            item.status = "executed"
            item.actual_exec_end_ts = monotonic_now()
            self._control_step_index += 1
            item.metadata["control_step_index"] = self._control_step_index
            self._history.append(item)
        self.event_logger.record(
            "action_execution_finished",
            **item.to_event_fields(),
            performed_action=performed_action,
        )

    def mark_action_execution_started(self, item: ActionQueueItem) -> None:
        self.event_logger.record(
            "action_execution_started",
            **item.to_event_fields(),
            control_step_index=self._control_step_index + 1,
        )


class ContinuousObservationPublisher:
    """Capture observations at fixed FPS and send only the latest pending one."""

    def __init__(
        self,
        fps: float,
        capture_fn: Callable[[int], TimedObservation],
        send_fn: Callable[[TimedObservation], bool],
        event_logger: StructuredEventLogger,
        max_pending_observations: int = 1,
    ) -> None:
        if max_pending_observations != 1:
            raise ValueError("ContinuousObservationPublisher currently implements latest-only max_pending=1")
        self.fps = fps
        self.capture_fn = capture_fn
        self.send_fn = send_fn
        self.event_logger = event_logger
        self.max_pending_observations = max_pending_observations
        self.shutdown_event = threading.Event()
        self._condition = threading.Condition()
        self._latest: TimedObservation | None = None
        self._capture_thread: threading.Thread | None = None
        self._send_thread: threading.Thread | None = None
        self.dropped_observation_count = 0

    def start(self) -> None:
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._send_thread = threading.Thread(target=self._send_loop, daemon=True)
        self._capture_thread.start()
        self._send_thread.start()

    def stop(self) -> None:
        self.shutdown_event.set()
        with self._condition:
            self._condition.notify_all()
        for thread in (self._capture_thread, self._send_thread):
            if thread is not None:
                thread.join(timeout=2)

    def _capture_loop(self) -> None:
        obs_id = 0
        dt = 1.0 / self.fps
        next_capture = monotonic_now()
        while not self.shutdown_event.is_set():
            now = monotonic_now()
            if now < next_capture:
                time.sleep(min(next_capture - now, dt))
                continue
            capture_start_ts = monotonic_now()
            self.event_logger.record("observation_capture_start", obs_id=obs_id)
            obs = self.capture_fn(obs_id)
            capture_end_ts = monotonic_now()
            obs.metadata.update(
                {
                    "obs_id": obs_id,
                    "obs_capture_start_ts": capture_start_ts,
                    "obs_capture_end_ts": capture_end_ts,
                    "client_monotonic_ts": capture_end_ts,
                    "wall_time_ts": wall_now(),
                }
            )
            self.event_logger.record(
                "observation_capture_end",
                obs_id=obs_id,
                obs_capture_start_ts=capture_start_ts,
                obs_capture_end_ts=capture_end_ts,
            )
            with self._condition:
                if self._latest is not None:
                    dropped_id = self._latest.metadata.get("obs_id")
                    self.dropped_observation_count += 1
                    self.event_logger.record(
                        "observation_dropped",
                        obs_id=dropped_id,
                        dropped_observation_count=self.dropped_observation_count,
                        drop_reason="latest_only_overwrite_before_send",
                    )
                self._latest = obs
                self._condition.notify()
            obs_id += 1
            next_capture += dt

    def _send_loop(self) -> None:
        while not self.shutdown_event.is_set():
            with self._condition:
                while self._latest is None and not self.shutdown_event.is_set():
                    self._condition.wait(timeout=0.1)
                obs = self._latest
                self._latest = None
            if obs is None:
                continue
            obs_send_ts = monotonic_now()
            obs.metadata["obs_send_ts"] = obs_send_ts
            try:
                sent = self.send_fn(obs)
            except Exception as exc:  # noqa: BLE001
                self.event_logger.record(
                    "observation_send_failed",
                    obs_id=obs.metadata.get("obs_id"),
                    obs_send_ts=obs_send_ts,
                    error=f"{type(exc).__name__}: {exc}",
                    dropped_observation_count=self.dropped_observation_count,
                )
                if self.shutdown_event.is_set() or "Client not running" in str(exc):
                    break
                raise
            self.event_logger.record(
                "observation_sent",
                obs_id=obs.metadata.get("obs_id"),
                obs_send_ts=obs_send_ts,
                sent=int(sent),
                dropped_observation_count=self.dropped_observation_count,
            )


def timed_actions_to_inference_result(
    timed_actions: list[TimedAction],
    default_receive_ts: float | None = None,
) -> InferenceResult | None:
    if not timed_actions:
        return None
    first = timed_actions[0]
    metadata = first.get_metadata() if isinstance(first.get_metadata(), dict) else {}
    inference_id = int(metadata.get("inference_id", metadata.get("source_inference_id", 0)))
    source_obs_id = int(metadata.get("source_obs_id", metadata.get("source_observation_timestep", 0)))
    source_obs_capture_ts = float(
        metadata.get(
            "source_obs_capture_ts",
            metadata.get("source_observation_monotonic_ts", monotonic_now()),
        )
    )
    source_obs_capture_wall_time = metadata.get("source_obs_capture_wall_time")
    inference_start_ts = float(metadata.get("inference_start_ts", monotonic_now()))
    inference_end_ts = float(metadata.get("inference_end_ts", default_receive_ts or monotonic_now()))
    actions = [action.get_action().detach().clone().cpu() for action in timed_actions]
    return InferenceResult(
        inference_id=inference_id,
        source_obs_id=source_obs_id,
        source_obs_capture_ts=source_obs_capture_ts,
        source_obs_capture_wall_time=source_obs_capture_wall_time,
        inference_start_ts=inference_start_ts,
        inference_end_ts=inference_end_ts,
        generated_action_chunk=actions,
        action_chunk_size=len(actions),
        model_latency=float(metadata.get("model_latency", inference_end_ts - inference_start_ts)),
        server_queue_delay=float(metadata.get("server_queue_delay", 0.0)),
        metadata=metadata,
    )


def make_timed_actions_from_result(result: InferenceResult, control_dt: float) -> list[TimedAction]:
    actions = []
    now_wall = wall_now()
    for i, action in enumerate(result.generated_action_chunk):
        actions.append(
            TimedAction(
                timestamp=now_wall + i * control_dt,
                timestep=i,
                action=action,
                metadata={
                    "inference_id": result.inference_id,
                    "source_inference_id": result.inference_id,
                    "source_obs_id": result.source_obs_id,
                    "source_obs_capture_ts": result.source_obs_capture_ts,
                    "source_obs_capture_wall_time": result.source_obs_capture_wall_time,
                    "inference_start_ts": result.inference_start_ts,
                    "inference_end_ts": result.inference_end_ts,
                    "model_latency": result.model_latency,
                    "server_queue_delay": result.server_queue_delay,
                    "action_chunk_size": result.action_chunk_size,
                    "chunk_index": i,
                    **result.metadata,
                },
            )
        )
    return actions
