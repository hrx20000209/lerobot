# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Continuous async robot client.

This is a separate entrypoint from ``robot_client.py``. It continuously captures
and sends observations, polls completed action chunks, and updates only pending
future actions in an internal queue. Motor execution is disabled by default.
"""

import logging
import pickle  # nosec
import threading
import time
from collections import OrderedDict
from dataclasses import asdict
from pprint import pformat
from typing import Any

import draccus
import grpc
import torch

from lerobot.transport import services_pb2
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data, shutdown_rerun

from .continuous import (
    AsyncActionQueueManager,
    ContinuousAsyncConfig,
    ContinuousObservationPublisher,
    StructuredEventLogger,
    monotonic_now,
    timed_actions_to_inference_result,
)
from .continuous_configs import ContinuousRobotClientConfig
from .helpers import RawObservation, TimedObservation, save_observation_images
from .robot_client import RobotClient


class ContinuousRobotClient(RobotClient):
    prefix = "continuous_robot_client"

    def __init__(self, config: ContinuousRobotClientConfig):
        super().__init__(config)
        self.config = config
        self.event_logger = StructuredEventLogger(config.timeline_log_path, enabled=True)
        self.continuous_config = ContinuousAsyncConfig(
            continuous_obs_fps=config.continuous_obs_fps,
            control_fps=config.fps,
            aggregation_fn=config.aggregation_fn,  # type: ignore[arg-type]
            max_pending_observations=config.max_pending_observations,
            stale_inference_max_age=config.stale_inference_max_age,
            min_usable_actions=config.min_usable_actions,
            blend_horizon=config.blend_horizon,
            blend_alpha=config.blend_alpha,
            max_joint_delta=config.max_joint_delta,
            max_gripper_delta=config.max_gripper_delta,
            max_joint_delta_per_step=config.max_joint_delta_per_step,
            max_joint_abs_range=config.max_joint_abs_range,
            max_gripper_delta_per_step=config.max_gripper_delta_per_step,
            emergency_stop=config.emergency_stop,
            shadow_mode=config.shadow_mode,
            enable_robot_execution=config.enable_robot_execution,
            timeline_log_path=config.timeline_log_path,
            timeline_plot_path=config.timeline_plot_path,
        )
        self.queue_manager = AsyncActionQueueManager(self.continuous_config, self.event_logger)
        self.publisher = ContinuousObservationPublisher(
            fps=config.continuous_obs_fps,
            capture_fn=self._capture_continuous_observation,
            send_fn=self.send_observation,
            event_logger=self.event_logger,
            max_pending_observations=config.max_pending_observations,
        )
        self._action_receiver_thread: threading.Thread | None = None
        self._action_executor_thread: threading.Thread | None = None
        self._obs_local_capture_ts: OrderedDict[int, float] = OrderedDict()
        self._obs_local_capture_ts_lock = threading.RLock()
        self._obs_local_capture_ts_capacity = 256
        self._last_commanded_action: torch.Tensor | None = None
        self._last_sanitized_action: torch.Tensor | None = None
        self._executed_control_steps = 0

    def _remember_local_observation_time(self, obs_id: int, capture_ts: float) -> None:
        with self._obs_local_capture_ts_lock:
            self._obs_local_capture_ts[obs_id] = capture_ts
            self._obs_local_capture_ts.move_to_end(obs_id)
            while len(self._obs_local_capture_ts) > self._obs_local_capture_ts_capacity:
                self._obs_local_capture_ts.popitem(last=False)

    def _get_local_observation_time(self, obs_id: int) -> float | None:
        with self._obs_local_capture_ts_lock:
            value = self._obs_local_capture_ts.get(obs_id)
            if value is not None:
                self._obs_local_capture_ts.move_to_end(obs_id)
            return value

    def _capture_continuous_observation(self, obs_id: int) -> TimedObservation:
        raw_observation: RawObservation = self.robot.get_observation()
        if self._last_commanded_action is None:
            try:
                self._last_commanded_action = torch.tensor(
                    [raw_observation[key] for key in self.robot.action_features], dtype=torch.float32
                )
            except KeyError:
                self.logger.warning("Could not initialize action clamp state from observation keys")
        raw_observation["task"] = self.config.task
        if self.config.display_data:
            log_rerun_data(observation=raw_observation, compress_images=True)
        image_paths = save_observation_images(
            raw_observation,
            self.timeline_image_dir,
            obs_id,
            mode=self.timeline_image_mode if self.config.record_timeline else "",
            key_frame=True,
        )
        local_capture_ts = monotonic_now()
        self._remember_local_observation_time(obs_id, local_capture_ts)
        return TimedObservation(
            timestamp=time.time(),
            observation=raw_observation,
            timestep=obs_id,
            must_go=True,
            metadata={"obs_id": obs_id, "image_paths": image_paths, "client_local_capture_ts": local_capture_ts},
        )

    def receive_actions_continuous(self) -> None:
        self.logger.info("Continuous action receiver thread starting")
        while self.running:
            try:
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                receive_ts = monotonic_now()
                if len(actions_chunk.data) == 0:
                    continue
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                result = timed_actions_to_inference_result(timed_actions, default_receive_ts=receive_ts)
                if result is None:
                    continue
                local_capture_ts = self._get_local_observation_time(result.source_obs_id)
                if local_capture_ts is None:
                    self.event_logger.record(
                        "inference_discarded",
                        inference_id=result.inference_id,
                        source_obs_id=result.source_obs_id,
                        discard_reason="missing_client_capture_time",
                    )
                    continue
                # Server monotonic timestamps are only meaningful on the server.
                # Client-side queue alignment, stale checks, and latency logs must
                # use this process's local monotonic clock.
                result.metadata["server_source_obs_capture_ts"] = result.source_obs_capture_ts
                result.source_obs_capture_ts = local_capture_ts
                self.event_logger.record(
                    "action_chunk_received_by_client",
                    inference_id=result.inference_id,
                    source_obs_id=result.source_obs_id,
                    action_chunk_size=result.action_chunk_size,
                    obs_age_at_receive=receive_ts - result.source_obs_capture_ts,
                    inference_latency=result.model_latency,
                )
                self.queue_manager.apply_inference_result(result)
            except grpc.RpcError as exc:
                self.logger.error("Error receiving continuous actions: %s", exc)
                time.sleep(self.config.environment_dt)

    def _action_tensor_to_action_dict_safe(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = action_tensor.detach().float().cpu()
        if torch.isnan(action).any() or torch.isinf(action).any():
            raise ValueError("Refusing to execute action with NaN or Inf")
        if (
            self.config.max_joint_abs_range is not None
            and float(torch.max(torch.abs(action))) > self.config.max_joint_abs_range
        ):
            raise ValueError("Refusing to execute action outside max_joint_abs_range")
        if self._last_commanded_action is not None:
            previous = self._last_commanded_action
            if action.shape != previous.shape:
                raise ValueError(
                    f"Refusing to execute action with shape {tuple(action.shape)} after {tuple(previous.shape)}"
                )
            delta = torch.abs(action - previous)
            clipped = action.clone()
            clipped_any = False
            if self.config.max_joint_delta_per_step is not None and action.numel() > 1:
                limit = float(self.config.max_joint_delta_per_step)
                body_delta = torch.clamp(action[:-1] - previous[:-1], -limit, limit)
                clipped[:-1] = previous[:-1] + body_delta
                clipped_any = clipped_any or bool(torch.any(torch.abs(delta[:-1]) > limit))
            if self.config.max_gripper_delta_per_step is not None and action.numel() > 0:
                limit = float(self.config.max_gripper_delta_per_step)
                gripper_delta = torch.clamp(action[-1] - previous[-1], -limit, limit)
                clipped[-1] = previous[-1] + gripper_delta
                clipped_any = clipped_any or bool(delta[-1] > limit)
            if clipped_any:
                self.event_logger.record(
                    "action_step_clipped",
                    max_delta_before=float(torch.max(delta)),
                    max_delta_after=float(torch.max(torch.abs(clipped - previous))),
                    max_joint_delta_per_step=self.config.max_joint_delta_per_step,
                    max_gripper_delta_per_step=self.config.max_gripper_delta_per_step,
                )
                action = clipped
        self._last_sanitized_action = action.detach().clone()
        return {key: action[i].item() for i, key in enumerate(self.robot.action_features)}

    def execute_actions_continuous(self) -> None:
        self.logger.info(
            "Continuous action executor starting | shadow_mode=%s enable_robot_execution=%s",
            self.config.shadow_mode,
            self.config.enable_robot_execution,
        )
        while self.running:
            loop_start = monotonic_now()
            item = self.queue_manager.pop_next_action()
            if item is not None:
                self.queue_manager.mark_action_execution_started(item)
                performed_action: Any = None
                try:
                    commanded_action = self._action_tensor_to_action_dict_safe(item.action)
                    if self.config.emergency_stop:
                        raise RuntimeError("emergency_stop is set")
                    if self.config.shadow_mode or not self.config.enable_robot_execution:
                        performed_action = {"shadow_mode": True, "commanded_action": commanded_action}
                    else:
                        performed_action = self.robot.send_action(commanded_action)
                    self._last_commanded_action = (
                        self._last_sanitized_action.detach().float().cpu()
                        if self._last_sanitized_action is not None
                        else item.action.detach().float().cpu()
                    )
                    if self.config.display_data:
                        log_rerun_data(action=performed_action)
                except Exception as exc:  # noqa: BLE001
                    performed_action = {"execution_rejected": True, "reason": str(exc)}
                    self.event_logger.record(
                        "inference_discarded",
                        action_id=item.action_id,
                        source_obs_id=item.source_obs_id,
                        source_inference_id=item.source_inference_id,
                        discard_reason=f"action_execution_rejected:{type(exc).__name__}",
                    )
                self.queue_manager.finish_action_execution(item, performed_action=performed_action)
                self._executed_control_steps += 1
                if (
                    self.config.max_control_steps is not None
                    and self._executed_control_steps >= self.config.max_control_steps
                ):
                    self.event_logger.record(
                        "max_control_steps_reached",
                        executed_control_steps=self._executed_control_steps,
                        max_control_steps=self.config.max_control_steps,
                    )
                    self.shutdown_event.set()
            elapsed = monotonic_now() - loop_start
            time.sleep(max(0.0, self.config.environment_dt - elapsed))

    def run_continuous(self) -> None:
        self.publisher.start()
        self._action_receiver_thread = threading.Thread(target=self.receive_actions_continuous, daemon=True)
        self._action_executor_thread = threading.Thread(target=self.execute_actions_continuous, daemon=True)
        self._action_receiver_thread.start()
        self._action_executor_thread.start()
        while self.running:
            time.sleep(0.2)

    def stop(self):
        self.publisher.stop()
        super().stop()
        self.event_logger.close()
        for thread in (self._action_receiver_thread, self._action_executor_thread):
            if thread is not None:
                thread.join(timeout=2)


@draccus.wrap()
def continuous_async_client(cfg: ContinuousRobotClientConfig):
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="continuous_async_inference", ip=cfg.display_ip, port=cfg.display_port)
    client = ContinuousRobotClient(cfg)
    if client.start():
        try:
            client.run_continuous()
        finally:
            client.stop()
            if cfg.display_data:
                shutdown_rerun()
    else:
        client.stop()
        if cfg.display_data:
            shutdown_rerun()


if __name__ == "__main__":
    register_third_party_plugins()
    continuous_async_client()
