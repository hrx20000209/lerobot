# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Continuous async inference server.

Unlike ``policy_server.py``, inference is not triggered by ``GetActions``.
Incoming observations update a latest-only buffer; a background worker runs
policy inference whenever it is idle and a fresh observation exists.
"""

import logging
import pickle  # nosec
import queue
import threading
import time
import inspect
from concurrent import futures
from dataclasses import asdict
from pprint import pformat
from typing import Any

import draccus
import grpc

from lerobot.transport import services_pb2, services_pb2_grpc  # type: ignore
from lerobot.transport.utils import receive_bytes_in_chunks

from .continuous import InferenceResult, StructuredEventLogger, make_timed_actions_from_result, monotonic_now
from .continuous_configs import ContinuousPolicyServerConfig
from .helpers import TimedObservation
from .policy_server import PolicyServer


class ContinuousInferenceServer(PolicyServer):
    prefix = "continuous_policy_server"

    def __init__(self, config: ContinuousPolicyServerConfig):
        super().__init__(config)
        self.config = config
        self.event_logger = StructuredEventLogger(config.timeline_log_path, enabled=True)
        self._obs_condition = threading.Condition()
        self._latest_observation: TimedObservation | None = None
        self._latest_observation_recv_ts: float | None = None
        self._inference_thread: threading.Thread | None = None
        self._inference_id = 0
        self._result_queue: queue.Queue[list] = queue.Queue(maxsize=2)
        self._worker_started = False
        self._dropped_observation_count = 0
        self._continuous_policy_state: Any = None

    def _reset_server(self) -> None:
        super()._reset_server()
        with self._obs_condition:
            self._latest_observation = None
            self._latest_observation_recv_ts = None
            self._obs_condition.notify_all()
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=2)
        self._inference_thread = None
        self._worker_started = False
        self._inference_id = 0
        self._dropped_observation_count = 0
        self._result_queue = queue.Queue(maxsize=2)
        self._reset_continuous_policy_state()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        response = super().SendPolicyInstructions(request, context)
        self._reset_continuous_policy_state()
        self._start_inference_worker()
        return response

    def _start_inference_worker(self) -> None:
        if self._worker_started:
            return
        self._worker_started = True
        self._inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._inference_thread.start()

    def _supports_continuous_policy_state(self) -> bool:
        return bool(getattr(self.policy, "supports_continuous_state", False))

    def _reset_continuous_policy_state(self) -> None:
        self._continuous_policy_state = None
        if not self._supports_continuous_policy_state():
            return
        reset = getattr(self.policy, "reset_continuous_inference_state", None)
        if callable(reset):
            self._continuous_policy_state = reset()
            return
        initializer = getattr(self.policy, "init_continuous_inference_state", None)
        if callable(initializer):
            self._continuous_policy_state = initializer()

    def _get_action_chunk(self, observation):  # noqa: ANN001
        if not self._supports_continuous_policy_state():
            return super()._get_action_chunk(observation)

        method = getattr(self.policy, "predict_action_chunk_continuous", None)
        if not callable(method):
            return super()._get_action_chunk(observation)

        kwargs = {}
        try:
            parameters = inspect.signature(method).parameters
            if "state" in parameters:
                kwargs["state"] = self._continuous_policy_state
            elif "continuous_state" in parameters:
                kwargs["continuous_state"] = self._continuous_policy_state
        except (TypeError, ValueError):
            pass

        output = method(observation, **kwargs)
        if isinstance(output, tuple) and len(output) == 2:
            action_chunk, self._continuous_policy_state = output
        elif isinstance(output, dict):
            action_chunk = output.get("actions", output.get("action_chunk"))
            self._continuous_policy_state = output.get(
                "state", output.get("continuous_state", self._continuous_policy_state)
            )
        else:
            action_chunk = output

        if action_chunk.ndim != 3:
            action_chunk = action_chunk.unsqueeze(0)
        return action_chunk[:, : self.actions_per_chunk, :]

    def SendObservations(self, request_iterator, context):  # noqa: N802
        receive_start_ts = monotonic_now()
        received_bytes = receive_bytes_in_chunks(request_iterator, None, self.shutdown_event, self.logger)
        timed_observation = pickle.loads(received_bytes)  # nosec
        if not isinstance(timed_observation, TimedObservation):
            raise TypeError(f"Expected TimedObservation, got {type(timed_observation)}")

        metadata = timed_observation.get_metadata()
        obs_id = int(metadata.get("obs_id", timed_observation.get_timestep()))
        server_obs_recv_ts = monotonic_now()
        metadata["server_obs_recv_ts"] = server_obs_recv_ts

        with self._obs_condition:
            if self._latest_observation is not None:
                old_id = self._latest_observation.get_metadata().get("obs_id")
                self._dropped_observation_count += 1
                self.event_logger.record(
                    "observation_dropped",
                    obs_id=old_id,
                    drop_reason="server_latest_only_overwrite",
                    dropped_observation_count=self._dropped_observation_count,
                )
            self._latest_observation = timed_observation
            self._latest_observation_recv_ts = server_obs_recv_ts
            self._obs_condition.notify()

        self.event_logger.record(
            "observation_received_by_server",
            obs_id=obs_id,
            server_obs_recv_ts=server_obs_recv_ts,
            receive_rpc_ms=(server_obs_recv_ts - receive_start_ts) * 1000,
            obs_capture_start_ts=metadata.get("obs_capture_start_ts"),
            obs_capture_end_ts=metadata.get("obs_capture_end_ts"),
            obs_send_ts=metadata.get("obs_send_ts"),
        )
        return services_pb2.Empty()

    def _take_latest_observation(self) -> tuple[TimedObservation, float] | None:
        with self._obs_condition:
            while self._latest_observation is None and self.running:
                self._obs_condition.wait(timeout=0.1)
            if not self.running or self._latest_observation is None:
                return None
            obs = self._latest_observation
            recv_ts = self._latest_observation_recv_ts or monotonic_now()
            self._latest_observation = None
            self._latest_observation_recv_ts = None
            return obs, recv_ts

    def _inference_loop(self) -> None:
        while self.running:
            taken = self._take_latest_observation()
            if taken is None:
                continue
            obs, recv_ts = taken
            if self.policy is None:
                time.sleep(0.01)
                continue

            obs_metadata = obs.get_metadata() if isinstance(obs.get_metadata(), dict) else {}
            obs_id = int(obs_metadata.get("obs_id", obs.get_timestep()))
            source_capture_ts = float(obs_metadata.get("obs_capture_end_ts", recv_ts))
            inference_id = self._inference_id
            self._inference_id += 1

            inference_start_ts = monotonic_now()
            self.event_logger.record(
                "inference_started",
                inference_id=inference_id,
                source_obs_id=obs_id,
                obs_id=obs_id,
                inference_start_ts=inference_start_ts,
                server_queue_delay=inference_start_ts - recv_ts,
            )
            wall_start = time.time()
            perf_start = time.perf_counter()
            try:
                action_chunk, predict_timing = self._predict_action_chunk(obs)
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("Continuous inference failed for obs %s: %s", obs_id, exc)
                self.event_logger.record(
                    "inference_discarded",
                    inference_id=inference_id,
                    source_obs_id=obs_id,
                    discard_reason=f"server_exception:{type(exc).__name__}",
                )
                continue

            model_latency = time.perf_counter() - perf_start
            inference_end_ts = monotonic_now()
            self.event_logger.record(
                "inference_finished",
                inference_id=inference_id,
                source_obs_id=obs_id,
                inference_start_ts=inference_start_ts,
                inference_end_ts=inference_end_ts,
                action_chunk_size=len(action_chunk),
                model_latency=model_latency,
                server_queue_delay=inference_start_ts - recv_ts,
                **predict_timing,
            )

            result = InferenceResult(
                inference_id=inference_id,
                source_obs_id=obs_id,
                source_obs_capture_ts=source_capture_ts,
                source_obs_capture_wall_time=obs_metadata.get("wall_time_ts", obs.get_timestamp()),
                inference_start_ts=inference_start_ts,
                inference_end_ts=inference_end_ts,
                generated_action_chunk=[action.get_action() for action in action_chunk],
                action_chunk_size=len(action_chunk),
                model_latency=model_latency,
                server_queue_delay=inference_start_ts - recv_ts,
                metadata={
                    "server_wall_inference_start": wall_start,
                    "source_observation_timestep": obs.get_timestep(),
                    "source_observation_timestamp": obs.get_timestamp(),
                },
            )
            timed_actions = make_timed_actions_from_result(result, self.config.environment_dt)
            self._put_latest_result(timed_actions)
            action_chunk_send_ts = monotonic_now()
            self.event_logger.record(
                "action_chunk_sent",
                inference_id=inference_id,
                source_obs_id=obs_id,
                action_chunk_send_ts=action_chunk_send_ts,
                action_chunk_size=len(timed_actions),
            )

    def _put_latest_result(self, timed_actions: list) -> None:
        while self._result_queue.full():
            try:
                _ = self._result_queue.get_nowait()
            except queue.Empty:
                break
        self._result_queue.put(timed_actions)

    def GetActions(self, request, context):  # noqa: N802
        try:
            timed_actions = self._result_queue.get(timeout=self.config.obs_queue_timeout)
        except queue.Empty:
            return services_pb2.Empty()
        return services_pb2.Actions(data=pickle.dumps(timed_actions))  # nosec

    def stop(self):
        super().stop()
        self.event_logger.close()
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=2)


@draccus.wrap()
def serve(cfg: ContinuousPolicyServerConfig):
    logging.info(pformat(asdict(cfg)))
    policy_server = ContinuousInferenceServer(cfg)
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")
    policy_server.logger.info(f"ContinuousInferenceServer started on {cfg.host}:{cfg.port}")
    server.start()
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        policy_server.logger.info("KeyboardInterrupt received, stopping server...")
    finally:
        policy_server.stop()
        server.stop(grace=0)


if __name__ == "__main__":
    serve()
