# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Example:
```shell
python -m lerobot.async_inference.policy_server \
     --host=127.0.0.1 \
     --port=8080 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
```
"""

import logging
import pickle  # nosec
import threading
import time
from concurrent import futures
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.processor import PolicyProcessorPipeline
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import receive_bytes_in_chunks
from lerobot.types import PolicyAction

from .configs import PolicyServerConfig
from .constants import SUPPORTED_POLICIES, normalize_policy_type
from .helpers import (
    FPSTracker,
    LatencyRecorder,
    Observation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    observations_similar,
    raw_observation_to_observation,
)
from .system_resources import SystemResourceRecorder


class PolicyServer(services_pb2_grpc.AsyncInferenceServicer):
    prefix = "policy_server"
    logger = get_logger(prefix)

    def __init__(self, config: PolicyServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=config.fps)
        self.latency_recorder = LatencyRecorder("policy_server", log_dir=config.timeline_log_dir)
        self.system_resource_recorder = SystemResourceRecorder(
            "policy_server",
            log_dir=config.timeline_log_dir,
            interval_s=config.system_resource_interval_s,
            enabled=config.record_system_resources,
            sample_nvidia_smi=config.system_resource_sample_nvidia_smi,
        )
        self.system_resource_recorder.start()

        self.observation_queue = Queue(maxsize=1)

        self._predicted_timesteps_lock = threading.Lock()
        self._predicted_timesteps = set()

        self.last_processed_obs = None

        # Attributes will be set by SendPolicyInstructions
        self.device = None
        self.policy_type = None
        self.lerobot_features = None
        self.actions_per_chunk = None
        self.policy = None
        self.preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None
        self.postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None

    def _record_timeline(self, kind: str, **metrics: Any) -> dict[str, Any] | None:
        if not self.config.record_timeline:
            return None
        return self.latency_recorder.record(kind, **metrics)

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    @property
    def policy_image_features(self):
        return self.policy.config.image_features

    def _reset_server(self) -> None:
        """Flushes server state when new client connects."""
        # only running inference on the latest observation received by the server
        self.shutdown_event.set()
        self.observation_queue = Queue(maxsize=1)

        with self._predicted_timesteps_lock:
            self._predicted_timesteps = set()

    def Ready(self, request, context):  # noqa: N802
        client_id = context.peer()
        self.logger.info(f"Client {client_id} connected and ready")
        self._reset_server()
        self.shutdown_event.clear()

        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        """Receive policy instructions from the robot client"""

        if not self.running:
            self.logger.warning("Server is not running. Ignoring policy instructions.")
            return services_pb2.Empty()

        client_id = context.peer()

        policy_specs = pickle.loads(request.data)  # nosec

        if not isinstance(policy_specs, RemotePolicyConfig):
            raise TypeError(f"Policy specs must be a RemotePolicyConfig. Got {type(policy_specs)}")

        # Normalize CLI aliases before validating and loading. The policy registry uses
        # Python module names such as "vla_jepa", but users may pass "vla-jepa".
        policy_specs.policy_type = normalize_policy_type(policy_specs.policy_type)

        if policy_specs.policy_type not in SUPPORTED_POLICIES:
            raise ValueError(
                f"Policy type {policy_specs.policy_type} not supported. "
                f"Supported policies: {SUPPORTED_POLICIES}"
            )

        self.logger.info(
            f"Receiving policy instructions from {client_id} | "
            f"Policy type: {policy_specs.policy_type} | "
            f"Pretrained name or path: {policy_specs.pretrained_name_or_path} | "
            f"Actions per chunk: {policy_specs.actions_per_chunk} | "
            f"Device: {policy_specs.device}"
        )

        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type  # act, pi0, etc.
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk

        start = time.perf_counter()
        self.policy = self._load_policy(policy_specs)

        # Load preprocessor and postprocessor, overriding device to match requested device.
        # GigaWorld stores Identity-only processors; its policy moves tensors to the
        # policy device internally, so device/rename overrides do not apply.
        device_override = {"device": self.device}
        if policy_specs.policy_type == "giga_world":
            preprocessor_overrides = {}
            postprocessor_overrides = {}
        else:
            preprocessor_overrides = {
                "device_processor": device_override,
                "rename_observations_processor": {"rename_map": policy_specs.rename_map},
            }
            postprocessor_overrides = {"device_processor": device_override}
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_specs.pretrained_name_or_path,
            preprocessor_overrides=preprocessor_overrides,
            postprocessor_overrides=postprocessor_overrides,
        )

        end = time.perf_counter()
        self.latency_recorder.record(
            "server_policy_setup",
            total_ms=(end - start) * 1000,
        )

        self.logger.info(f"Time taken to put policy on {self.device}: {end - start:.4f} seconds")

        return services_pb2.Empty()

    def _load_policy(self, policy_specs: RemotePolicyConfig) -> torch.nn.Module:
        policy_class = get_policy_class(policy_specs.policy_type)
        config = PreTrainedConfig.from_pretrained(policy_specs.pretrained_name_or_path)
        config.device = policy_specs.device

        checkpoint_path = Path(policy_specs.pretrained_name_or_path)
        local_adapter_checkpoint = (
            checkpoint_path.is_dir() and (checkpoint_path / "adapter_config.json").exists()
        )
        local_model_file = checkpoint_path.is_dir() and (
            checkpoint_path / "model.safetensors"
        ).exists()
        is_peft_checkpoint = getattr(config, "use_peft", False) and (
            local_adapter_checkpoint or not local_model_file
        )
        if is_peft_checkpoint:
            from peft import PeftConfig, PeftModel

            peft_config = PeftConfig.from_pretrained(policy_specs.pretrained_name_or_path)
            base_model_name_or_path = peft_config.base_model_name_or_path
            if not base_model_name_or_path:
                raise ValueError(
                    "PEFT adapter config does not define base_model_name_or_path; cannot load async policy."
                )

            self.logger.info(
                "Loading PEFT policy adapter for async inference | "
                f"Base: {base_model_name_or_path} | Adapter: {policy_specs.pretrained_name_or_path}"
            )
            policy = policy_class.from_pretrained(base_model_name_or_path, config=config)
            policy = PeftModel.from_pretrained(
                policy,
                policy_specs.pretrained_name_or_path,
                config=peft_config,
                is_trainable=False,
            )
        else:
            policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path, config=config)

        policy.to(policy_specs.device)
        policy.eval()
        prepare_for_inference = getattr(policy, "prepare_for_inference", None)
        if callable(prepare_for_inference):
            self.logger.info("Preparing policy runtime before accepting observations")
            if policy_specs.policy_type == "giga_world":
                prepare_for_inference(task=policy_specs.task)
            else:
                prepare_for_inference()
        return policy

    def SendObservations(self, request_iterator, context):  # noqa: N802
        """Receive observations from the robot client"""
        client_id = context.peer()
        self.logger.debug(f"Receiving observations from {client_id}")

        receive_time = time.time()  # comparing timestamps so need time.time()
        start_deserialize = time.perf_counter()
        received_bytes = receive_bytes_in_chunks(
            request_iterator, None, self.shutdown_event, self.logger
        )  # blocking call while looping over request_iterator
        timed_observation = pickle.loads(received_bytes)  # nosec
        deserialize_time = time.perf_counter() - start_deserialize

        self.logger.debug(f"Received observation #{timed_observation.get_timestep()}")

        obs_timestep = timed_observation.get_timestep()
        obs_timestamp = timed_observation.get_timestamp()

        # Calculate FPS metrics
        fps_metrics = self.fps_tracker.calculate_fps_metrics(obs_timestamp)

        self.logger.debug(
            f"Received observation #{obs_timestep} | "
            f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "  # fps at which observations are received from client
            f"Target: {fps_metrics['target_fps']:.2f} | "
            f"One-way latency: {(receive_time - obs_timestamp) * 1000:.2f}ms"
        )

        self.logger.debug(
            f"Server timestamp: {receive_time:.6f} | "
            f"Client timestamp: {obs_timestamp:.6f} | "
            f"Deserialization time: {deserialize_time:.6f}s"
        )

        enqueued = self._enqueue_observation(
            timed_observation  # wrapping a RawObservation
        )
        self.latency_recorder.record(
            "server_receive_observation",
            timestep=obs_timestep,
            client_to_server_ms=(receive_time - obs_timestamp) * 1000,
            deserialize_ms=deserialize_time * 1000,
            enqueued=int(enqueued),
        )
        if not enqueued:
            self.logger.debug(f"Observation #{obs_timestep} has been filtered out")

        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        """Returns actions to the robot client. Actions are sent as a single
        chunk, containing multiple actions."""
        client_id = context.peer()
        self.logger.debug(f"Client {client_id} connected for action streaming")

        # Generate action based on the most recent observation and its timestep
        try:
            getactions_starts = time.perf_counter()
            obs = self.observation_queue.get(timeout=self.config.obs_queue_timeout)
            queue_wait_time = time.perf_counter() - getactions_starts
            self.logger.info(
                f"Running inference for observation #{obs.get_timestep()} (must_go: {obs.must_go})"
            )

            with self._predicted_timesteps_lock:
                self._predicted_timesteps.add(obs.get_timestep())

            inference_start_time = time.time()
            start_time = time.perf_counter()
            action_chunk, predict_timing = self._predict_action_chunk(obs)
            inference_end_time = time.time()
            inference_time = time.perf_counter() - start_time

            start_time = time.perf_counter()
            if action_chunk:
                obs_metadata = obs.get_metadata() if hasattr(obs, "get_metadata") else {}
                if not isinstance(obs_metadata, dict):
                    obs_metadata = {}
                queue_size_at_observation = obs_metadata.get("queue_size_at_capture")
                server_timeline = {
                    "observation_timestep": obs.get_timestep(),
                    "observation_timestamp": obs.get_timestamp(),
                    "inference_start_time": inference_start_time,
                    "inference_end_time": inference_end_time,
                    "first_timestep": action_chunk[0].get_timestep(),
                    "last_timestep": action_chunk[-1].get_timestep(),
                    "actions_count": len(action_chunk),
                    "queue_size_at_observation": queue_size_at_observation,
                    "must_go": int(obs.must_go),
                }
                for timed_action in action_chunk:
                    timed_action.metadata.update(
                        {
                            "source_observation_timestep": obs.get_timestep(),
                            "source_observation_timestamp": obs.get_timestamp(),
                            "inference_start_time": inference_start_time,
                            "inference_end_time": inference_end_time,
                            "server_timeline": server_timeline,
                        }
                    )
                action_chunk[0].metadata["server_timestamp"] = time.time()
            actions_bytes = pickle.dumps(action_chunk)  # nosec
            serialize_time = time.perf_counter() - start_time

            # Create and return the action chunk
            actions = services_pb2.Actions(data=actions_bytes)

            self.logger.info(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Total time: {(inference_time + serialize_time) * 1000:.2f}ms"
            )

            self.logger.debug(
                f"Action chunk #{obs.get_timestep()} generated | "
                f"Inference time: {inference_time:.2f}s |"
                f"Serialize time: {serialize_time:.2f}s |"
                f"Total time: {inference_time + serialize_time:.2f}s"
            )

            sleep_time = max(0, self.config.inference_latency - max(0, time.perf_counter() - getactions_starts))
            self.latency_recorder.record(
                "server_action_chunk",
                timestep=obs.get_timestep(),
                queue_wait_ms=queue_wait_time * 1000,
                inference_ms=inference_time * 1000,
                serialize_ms=serialize_time * 1000,
                sleep_ms=sleep_time * 1000,
                total_ms=(time.perf_counter() - getactions_starts) * 1000,
                **predict_timing,
            )
            self._record_timeline(
                "timeline_server_inference",
                timestep=obs.get_timestep(),
                observation_time=obs.get_timestamp(),
                start_time=inference_start_time,
                end_time=inference_end_time,
                first_timestep=action_chunk[0].get_timestep() if action_chunk else -1,
                last_timestep=action_chunk[-1].get_timestep() if action_chunk else -1,
                actions_count=len(action_chunk),
                queue_size_at_observation=(
                    obs.get_metadata().get("queue_size_at_capture")
                    if hasattr(obs, "get_metadata") and isinstance(obs.get_metadata(), dict)
                    else None
                ),
                must_go=int(obs.must_go),
            )

            time.sleep(sleep_time)  # sleep controls inference latency

            return actions

        except Empty:  # no observation added to queue in obs_queue_timeout
            return services_pb2.Empty()

        except Exception as e:
            self.logger.error(f"Error in StreamActions: {e}")

            return services_pb2.Empty()

    def _obs_sanity_checks(self, obs: TimedObservation, previous_obs: TimedObservation) -> bool:
        """Check if the observation is valid to be processed by the policy"""
        with self._predicted_timesteps_lock:
            predicted_timesteps = self._predicted_timesteps

        if obs.get_timestep() in predicted_timesteps:
            self.logger.debug(f"Skipping observation #{obs.get_timestep()} - Timestep predicted already!")
            return False

        elif observations_similar(obs, previous_obs, lerobot_features=self.lerobot_features):
            self.logger.debug(
                f"Skipping observation #{obs.get_timestep()} - Observation too similar to last obs predicted!"
            )
            return False

        else:
            return True

    def _enqueue_observation(self, obs: TimedObservation) -> bool:
        """Enqueue an observation if it must go through processing, otherwise skip it.
        Observations not in queue are never run through the policy network"""

        if (
            obs.must_go
            or self.last_processed_obs is None
            or self._obs_sanity_checks(obs, self.last_processed_obs)
        ):
            last_obs = self.last_processed_obs.get_timestep() if self.last_processed_obs else "None"
            self.logger.debug(
                f"Enqueuing observation. Must go: {obs.must_go} | Last processed obs: {last_obs}"
            )

            # If queue is full, get the old observation to make room
            if self.observation_queue.full():
                # pops from queue
                _ = self.observation_queue.get_nowait()
                self.logger.debug("Observation queue was full, removed oldest observation")

            # Now put the new observation (never blocks as queue is non-full here)
            self.observation_queue.put(obs)
            return True

        return False

    def _time_action_chunk(
        self,
        t_0: float,
        action_chunk: list[torch.Tensor],
        i_0: int,
        metadata: dict[str, Any] | None = None,
    ) -> list[TimedAction]:
        """Turn a chunk of actions into a list of TimedAction instances,
        with the first action corresponding to t_0 and the rest corresponding to
        t_0 + i*environment_dt for i in range(len(action_chunk))
        """
        return [
            TimedAction(
                timestamp=t_0 + i * self.config.environment_dt,
                timestep=i_0 + i,
                action=action,
                metadata=metadata if i == 0 and metadata is not None else {},
            )
            for i, action in enumerate(action_chunk)
        ]

    def _sync_policy_device(self) -> None:
        if self.device is not None and torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(torch.device(str(self.device)))

    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get an action chunk from the policy. The chunk contains only"""
        chunk = self.policy.predict_action_chunk(observation)
        if chunk.ndim != 3:
            chunk = chunk.unsqueeze(0)  # adding batch dimension, now shape is (B, chunk_size, action_dim)

        return chunk[:, : self.actions_per_chunk, :]

    def _extract_policy_profile(self) -> dict[str, float]:
        profile = {}
        candidates = [
            self.policy,
            getattr(self.policy, "model", None),
            getattr(self.policy, "diffusion", None),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            getter = getattr(candidate, "get_and_reset_latency_profile", None)
            data = getter() if callable(getter) else getattr(candidate, "last_latency_profile", None)
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, int | float) and not isinstance(value, bool):
                        profile[f"model_{key}"] = float(value)
        return profile

    def _predict_action_chunk(self, observation_t: TimedObservation) -> tuple[list[TimedAction], dict[str, float]]:
        """Predict an action chunk based on an observation.

        Pipeline:
        1. Convert raw observation to LeRobot format
        2. Apply preprocessor (tokenization, normalization, batching, device placement)
        3. Run policy inference to get action chunk
        4. Apply postprocessor (unnormalization, device movement)
        5. Convert to TimedAction list
        """
        """1. Prepare observation"""
        start_prepare = time.perf_counter()
        observation: Observation = raw_observation_to_observation(
            observation_t.get_observation(),
            self.lerobot_features,
            self.policy_image_features,
        )
        prepare_time = time.perf_counter() - start_prepare

        """2. Apply preprocessor"""
        self._sync_policy_device()
        start_preprocess = time.perf_counter()
        observation = self.preprocessor(observation)
        self._sync_policy_device()
        self.last_processed_obs: TimedObservation = observation_t
        preprocessing_time = time.perf_counter() - start_preprocess

        """3. Get action chunk"""
        self._sync_policy_device()
        start_inference = time.perf_counter()
        action_tensor = self._get_action_chunk(observation)
        policy_output_tensor = action_tensor.detach().float().cpu()
        self._sync_policy_device()
        inference_time = time.perf_counter() - start_inference
        self.logger.info(
            f"Preprocessing and inference took {inference_time:.4f}s, action shape: {action_tensor.shape}"
        )

        """4. Apply postprocessor"""
        # Apply postprocessor (handles unnormalization and device movement)
        # Postprocessor expects (B, action_dim) per action, but we have (B, chunk_size, action_dim)
        # So we process each action in the chunk individually
        self._sync_policy_device()
        start_postprocess = time.perf_counter()
        _, chunk_size, _ = action_tensor.shape

        # Process each action in the chunk
        processed_actions = []
        for i in range(chunk_size):
            # Extract action at timestep i: (B, action_dim)
            single_action = action_tensor[:, i, :]
            processed_action = self.postprocessor(single_action)
            processed_actions.append(processed_action)

        # Stack back to (B, chunk_size, action_dim), then remove batch dim
        action_tensor = torch.stack(processed_actions, dim=1).squeeze(0)
        self.logger.debug(f"Postprocessed action shape: {action_tensor.shape}")

        action_tensor = action_tensor.detach().cpu()
        self._sync_policy_device()

        self._record_timeline(
            "policy_action_chunk",
            source_observation_timestep=observation_t.get_timestep(),
            policy_output=policy_output_tensor.squeeze(0).tolist(),
            postprocessed_action=action_tensor.tolist(),
        )

        """5. Convert to TimedAction list"""
        postprocess_stops = time.perf_counter()
        postprocessing_time = postprocess_stops - start_postprocess
        timing = {
            "prepare_ms": prepare_time * 1000,
            "preprocess_ms": preprocessing_time * 1000,
            "policy_predict_ms": inference_time * 1000,
            "postprocess_ms": postprocessing_time * 1000,
            "predict_total_ms": (postprocess_stops - start_prepare) * 1000,
        }
        timing.update(self._extract_policy_profile())
        action_chunk = self._time_action_chunk(
            observation_t.get_timestamp(),
            list(action_tensor),
            observation_t.get_timestep(),
            metadata={"server_latency": timing},
        )

        self.logger.info(
            f"Observation {observation_t.get_timestep()} | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        self.logger.debug(
            f"Observation {observation_t.get_timestep()} | "
            f"Prepare time: {1000 * prepare_time:.2f}ms | "
            f"Preprocessing time: {1000 * preprocessing_time:.2f}ms | "
            f"Inference time: {1000 * inference_time:.2f}ms | "
            f"Postprocessing time: {1000 * postprocessing_time:.2f}ms | "
            f"Total time: {1000 * (postprocess_stops - start_prepare):.2f}ms"
        )

        return action_chunk, timing

    def stop(self):
        """Stop the server"""
        self._reset_server()
        resource_summary = self.system_resource_recorder.stop()
        if resource_summary is not None:
            self.logger.info(
                "System resource summary written to %s",
                self.system_resource_recorder.summary_path,
            )
        self.latency_recorder.log_summary(self.logger)
        self.latency_recorder.close()
        self.logger.info("Server stopping...")


@draccus.wrap()
def serve(cfg: PolicyServerConfig):
    """Start the PolicyServer with the given configuration.

    Args:
        config: PolicyServerConfig instance. If None, uses default configuration.
    """
    logging.info(pformat(asdict(cfg)))

    # Create the server instance first
    policy_server = PolicyServer(cfg)

    # Setup and start gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
    server.add_insecure_port(f"{cfg.host}:{cfg.port}")

    policy_server.logger.info(f"PolicyServer started on {cfg.host}:{cfg.port}")
    server.start()

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        policy_server.logger.info("KeyboardInterrupt received, stopping server...")
    finally:
        policy_server.stop()
        server.stop(grace=0)
        policy_server.logger.info("Server terminated")


if __name__ == "__main__":
    serve()
