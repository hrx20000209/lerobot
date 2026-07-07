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
Example command:
```shell
python src/lerobot/async_inference/robot_client.py \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
    --robot.id=black \
    --task="dummy" \
    --server_address=127.0.0.1:8080 \
    --policy_type=act \
    --pretrained_name_or_path=user/model \
    --policy_device=mps \
    --client_device=cpu \
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True
```
"""

import logging
import os
import pickle  # nosec
import threading
import time
from collections.abc import Callable
from dataclasses import asdict
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import grpc
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.transport import (
    services_pb2,  # type: ignore
    services_pb2_grpc,  # type: ignore
)
from lerobot.transport.utils import grpc_channel_options, send_bytes_in_chunks
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data, shutdown_rerun

from .configs import RobotClientConfig
from .helpers import (
    Action,
    FPSTracker,
    LatencyRecorder,
    Observation,
    RawObservation,
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    save_observation_images,
    visualize_action_queue_size,
)
from .system_resources import SystemResourceRecorder


class RobotClient:
    prefix = "robot_client"
    logger = get_logger(prefix)

    def __init__(self, config: RobotClientConfig):
        """Initialize RobotClient with unified configuration.

        Args:
            config: RobotClientConfig containing all configuration parameters
        """
        # Store configuration
        self.config = config
        self.robot = make_robot_from_config(config.robot)
        self.robot.connect()

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)

        # Use environment variable if server_address is not provided in config
        self.server_address = config.server_address

        self.policy_config = RemotePolicyConfig(
            policy_type=config.policy_type,
            pretrained_name_or_path=config.pretrained_name_or_path,
            lerobot_features=lerobot_features,
            actions_per_chunk=config.actions_per_chunk,
            device=config.policy_device,
            task=config.task,
        )
        self.channel = grpc.insecure_channel(
            self.server_address, grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s")
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info(f"Initializing client to connect to server at {self.server_address}")

        self.shutdown_event = threading.Event()
        self._stopped = False
        self.latency_recorder = LatencyRecorder("robot_client", log_dir=config.timeline_log_dir)
        self.system_resource_recorder = SystemResourceRecorder(
            "robot_client",
            log_dir=config.timeline_log_dir,
            interval_s=config.system_resource_interval_s,
            enabled=config.record_system_resources,
            sample_nvidia_smi=config.system_resource_sample_nvidia_smi,
        )
        self.system_resource_recorder.start()
        self.timeline_image_mode = config.timeline_save_images or os.getenv("LEROBOT_TIMELINE_SAVE_IMAGES", "")
        self.timeline_image_dir = (
            self.latency_recorder.log_dir / f"robot_client_observation_images_{self.latency_recorder.run_id}"
        )

        # Initialize client side variables
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.latest_action_wall_time = None
        self.action_chunk_size = -1

        self._chunk_size_threshold = config.chunk_size_threshold

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()  # Protect queue operations
        self.action_queue_size = []
        self.start_barrier = threading.Barrier(2)  # 2 threads: action receiver, control loop

        # FPS measurement
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)

        self.logger.info("Robot connected and ready")

        # Use an event for thread-safe coordination
        self.must_go = threading.Event()
        self.must_go.set()  # Initially set - observations qualify for direct processing

    def _record_timeline(self, kind: str, **metrics: Any) -> dict[str, Any] | None:
        if not self.config.record_timeline:
            return None
        return self.latency_recorder.record(kind, **metrics)

    @property
    def running(self):
        return not self.shutdown_event.is_set()

    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # client-server handshake
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            end_time = time.perf_counter()
            self.latency_recorder.record("client_start", ready_rpc_ms=(end_time - start_time) * 1000)
            self.logger.debug(f"Connected to policy server in {end_time - start_time:.4f}s")

            # send policy instructions
            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)

            self.logger.info("Sending policy instructions to policy server")
            self.logger.debug(
                f"Policy type: {self.policy_config.policy_type} | "
                f"Pretrained name or path: {self.policy_config.pretrained_name_or_path} | "
                f"Device: {self.policy_config.device}"
            )

            setup_start = time.perf_counter()
            self.stub.SendPolicyInstructions(policy_setup)
            self.latency_recorder.record(
                "client_start",
                send_policy_instructions_ms=(time.perf_counter() - setup_start) * 1000,
            )

            self.shutdown_event.clear()

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Failed to connect to policy server: {e}")
            return False

    def stop(self):
        """Stop the robot client"""
        if self._stopped:
            return
        self._stopped = True
        self.shutdown_event.set()

        self.robot.disconnect()
        self.logger.debug("Robot disconnected")

        self.channel.close()
        self.logger.debug("Client stopped, channel closed")
        resource_summary = self.system_resource_recorder.stop()
        if resource_summary is not None:
            self.logger.info(
                "System resource summary written to %s",
                self.system_resource_recorder.summary_path,
            )
        self.latency_recorder.log_summary(self.logger)
        self.latency_recorder.close()

    def send_observation(
        self,
        obs: TimedObservation,
    ) -> bool:
        """Send observation to the policy server.
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            raise RuntimeError("Client not running. Run RobotClient.start() before sending observations.")

        if not isinstance(obs, TimedObservation):
            raise ValueError("Input observation needs to be a TimedObservation!")

        start_time = time.perf_counter()
        observation_bytes = pickle.dumps(obs)
        serialize_time = time.perf_counter() - start_time
        self.logger.debug(f"Observation serialization time: {serialize_time:.6f}s")

        try:
            send_start = time.perf_counter()
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            send_rpc_time = time.perf_counter() - send_start
            obs_timestep = obs.get_timestep()
            self.logger.debug(f"Sent observation #{obs_timestep} | ")
            self.latency_recorder.record(
                "client_send_observation",
                timestep=obs_timestep,
                serialize_ms=serialize_time * 1000,
                send_rpc_ms=send_rpc_time * 1000,
                total_ms=(serialize_time + send_rpc_time) * 1000,
                must_go=int(obs.must_go),
            )

            return True

        except grpc.RpcError as e:
            self.logger.error(f"Error sending observation #{obs.get_timestep()}: {e}")
            return False

    def _inspect_action_queue(self):
        with self.action_queue_lock:
            queue_size = self.action_queue.qsize()
            timestamps = sorted([action.get_timestep() for action in self.action_queue.queue])
        self.logger.debug(f"Queue size: {queue_size}, Queue contents: {timestamps}")
        return queue_size, timestamps

    def _aggregate_action_queues(
        self,
        incoming_actions: list[TimedAction],
        aggregate_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ):
        """Finds the same timestep actions in the queue and aggregates them using the aggregate_fn"""
        if aggregate_fn is None:
            # default aggregate function: take the latest action
            def aggregate_fn(x1, x2):
                return x2

        future_action_queue = Queue()
        with self.action_queue_lock:
            internal_queue = self.action_queue.queue

        current_action_queue = {action.get_timestep(): action.get_action() for action in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action

            # New action is older than the latest action in the queue, skip it
            if new_action.get_timestep() <= latest_action:
                continue

            # If the new action's timestep is not in the current action queue, add it directly
            elif new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
                continue

            # If the new action's timestep is in the current action queue, aggregate it
            # TODO: There is probably a way to do this with broadcasting of the two action tensors
            future_action_queue.put(
                TimedAction(
                    timestamp=new_action.get_timestamp(),
                    timestep=new_action.get_timestep(),
                    action=aggregate_fn(
                        current_action_queue[new_action.get_timestep()], new_action.get_action()
                    ),
                    metadata=dict(new_action.get_metadata()),
                )
            )

        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def receive_actions(self, verbose: bool = False):
        """Receive actions from the policy server"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")

        while self.running:
            try:
                # Use StreamActions to get a stream of actions from the server
                get_actions_start = time.perf_counter()
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                get_actions_rpc_time = time.perf_counter() - get_actions_start
                if len(actions_chunk.data) == 0:
                    continue  # received `Empty` from server, wait for next call

                receive_time = time.time()

                # Deserialize bytes back into list[TimedAction]
                deserialize_start = time.perf_counter()
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                deserialize_time = time.perf_counter() - deserialize_start

                # Log device type of received actions
                if len(timed_actions) > 0:
                    received_device = timed_actions[0].get_action().device.type
                    self.logger.debug(f"Received actions on device: {received_device}")

                # Move actions to client_device (e.g., for downstream planners that need GPU)
                client_device = self.config.client_device
                if client_device != "cpu":
                    for timed_action in timed_actions:
                        if timed_action.get_action().device.type != client_device:
                            timed_action.action = timed_action.get_action().to(client_device)
                    self.logger.debug(f"Converted actions to device: {client_device}")
                else:
                    self.logger.debug(f"Actions kept on device: {client_device}")

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))
                first_action_timestep = timed_actions[0].get_timestep() if timed_actions else -1
                last_action_timestep = timed_actions[-1].get_timestep() if timed_actions else -1
                incoming_timesteps = [a.get_timestep() for a in timed_actions]
                observation_to_actions_latency = (
                    (receive_time - timed_actions[0].get_timestamp()) * 1000 if timed_actions else 0
                )
                server_latency = {}
                server_timestamp = None
                if timed_actions:
                    metadata = timed_actions[0].get_metadata()
                    if isinstance(metadata, dict):
                        server_latency = metadata.get("server_latency", {}) or {}
                        server_timestamp = metadata.get("server_timestamp")
                server_to_client_latency = (
                    (receive_time - server_timestamp) * 1000
                    if isinstance(server_timestamp, int | float) and not isinstance(server_timestamp, bool)
                    else observation_to_actions_latency
                )
                if server_latency:
                    self.latency_recorder.record(
                        "server_action_chunk",
                        first_timestep=first_action_timestep,
                        last_timestep=last_action_timestep,
                        **server_latency,
                    )
                if timed_actions:
                    metadata = timed_actions[0].get_metadata()
                    server_timeline = metadata.get("server_timeline", {}) if isinstance(metadata, dict) else {}
                    inference_start_time = server_timeline.get("inference_start_time")
                    inference_end_time = server_timeline.get("inference_end_time")
                    if isinstance(inference_start_time, int | float) and isinstance(
                        inference_end_time, int | float
                    ):
                        self._record_timeline(
                            "timeline_server_inference",
                            timestep=server_timeline.get("observation_timestep", first_action_timestep),
                            observation_time=server_timeline.get("observation_timestamp"),
                            start_time=inference_start_time,
                            end_time=inference_end_time,
                            first_timestep=first_action_timestep,
                            last_timestep=last_action_timestep,
                            actions_count=len(timed_actions),
                            queue_size_at_observation=server_timeline.get("queue_size_at_observation"),
                            must_go=server_timeline.get("must_go"),
                            received_time=receive_time,
                            server_to_client_ms=server_to_client_latency,
                        )

                # Calculate network latency if we have matching observations
                old_size = -1
                old_timesteps: list[int] = []
                if self.config.record_timeline or verbose:
                    old_size, old_timesteps = self._inspect_action_queue()
                    if not old_timesteps:
                        with self.latest_action_lock:
                            old_timesteps = [self.latest_action]

                if len(timed_actions) > 0 and verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    self.logger.debug(f"Current latest action: {latest_action}")

                    self.logger.info(
                        f"Received action chunk for step #{first_action_timestep} | "
                        f"Latest action: #{latest_action} | "
                        f"Incoming actions: {incoming_timesteps[0]}:{incoming_timesteps[-1]} | "
                        f"Network latency (server->client): {server_to_client_latency:.2f}ms | "
                        f"Deserialization time: {deserialize_time * 1000:.2f}ms"
                    )

                # Update action queue
                start_time = time.perf_counter()
                self._aggregate_action_queues(timed_actions, self.config.aggregate_fn)
                queue_update_time = time.perf_counter() - start_time
                new_size = -1
                new_timesteps: list[int] = []
                if self.config.record_timeline or verbose:
                    new_size, new_timesteps = self._inspect_action_queue()
                self.latency_recorder.record(
                    "client_receive_actions",
                    first_timestep=first_action_timestep,
                    last_timestep=last_action_timestep,
                    actions_count=len(timed_actions),
                    get_actions_rpc_ms=get_actions_rpc_time * 1000,
                    server_to_client_ms=server_to_client_latency,
                    observation_to_actions_ms=observation_to_actions_latency,
                    deserialize_ms=deserialize_time * 1000,
                    queue_update_ms=queue_update_time * 1000,
                    total_ms=(get_actions_rpc_time + deserialize_time + queue_update_time) * 1000,
                )
                self._record_timeline(
                    "timeline_receive_actions",
                    first_timestep=first_action_timestep,
                    last_timestep=last_action_timestep,
                    actions_count=len(timed_actions),
                    receive_time=receive_time,
                    queue_size_before=old_size,
                    queue_size_after=new_size,
                    server_to_client_ms=server_to_client_latency,
                )

                self.must_go.set()  # after receiving actions, next empty queue triggers must-go processing!

                if verbose:
                    with self.latest_action_lock:
                        latest_action = self.latest_action

                    old_steps = f"{old_timesteps[0]}:{old_timesteps[-1]}" if old_timesteps else "empty"
                    incoming_steps = (
                        f"{incoming_timesteps[0]}:{incoming_timesteps[-1]}"
                        if incoming_timesteps
                        else "empty"
                    )
                    new_steps = f"{new_timesteps[0]}:{new_timesteps[-1]}" if new_timesteps else "empty"
                    self.logger.info(
                        f"Latest action: {latest_action} | "
                        f"Old action steps: {old_steps} | "
                        f"Incoming action steps: {incoming_steps} | "
                        f"Updated action steps: {new_steps}"
                    )
                    self.logger.debug(
                        f"Queue update complete ({queue_update_time:.6f}s) | "
                        f"Before: {old_size} items | "
                        f"After: {new_size} items | "
                    )

            except grpc.RpcError as e:
                self.logger.error(f"Error receiving actions: {e}")

    def actions_available(self):
        """Check if there are actions available in the queue"""
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _action_tensor_to_action_dict(self, action_tensor: torch.Tensor) -> dict[str, float]:
        action = {key: action_tensor[i].item() for i, key in enumerate(self.robot.action_features)}
        return action

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any]:
        """Reading and performing actions in local queue"""

        # Lock only for queue operations
        get_start = time.perf_counter()
        with self.action_queue_lock:
            queue_size_before_pop = self.action_queue.qsize()
            self.action_queue_size.append(queue_size_before_pop)
            # Get action from queue
            timed_action = self.action_queue.get_nowait()
            queue_size_after_pop = self.action_queue.qsize()
        get_end = time.perf_counter() - get_start

        execution_start_time = time.time()
        send_start = time.perf_counter()
        commanded_action = self._action_tensor_to_action_dict(timed_action.get_action())
        _performed_action = self.robot.send_action(commanded_action)
        execution_end_time = time.time()
        send_action_time = time.perf_counter() - send_start
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()
            self.latest_action_wall_time = execution_end_time
        action_metadata = timed_action.get_metadata()
        source_observation_timestamp = None
        source_observation_timestep = None
        inference_start_time = None
        inference_end_time = None
        if isinstance(action_metadata, dict):
            source_observation_timestamp = action_metadata.get("source_observation_timestamp")
            source_observation_timestep = action_metadata.get("source_observation_timestep")
            inference_start_time = action_metadata.get("inference_start_time")
            inference_end_time = action_metadata.get("inference_end_time")
        source_observation_age_ms = (
            (execution_start_time - source_observation_timestamp) * 1000
            if isinstance(source_observation_timestamp, int | float)
            else None
        )
        self.latency_recorder.record(
            "client_execute_action",
            timestep=timed_action.get_timestep(),
            queue_pop_ms=get_end * 1000,
            robot_send_action_ms=send_action_time * 1000,
            action_age_ms=(time.time() - timed_action.get_timestamp()) * 1000,
            total_ms=(get_end + send_action_time) * 1000,
        )
        self._record_timeline(
            "timeline_action",
            timestep=timed_action.get_timestep(),
            start_time=execution_start_time,
            end_time=execution_end_time,
            action_timestamp=timed_action.get_timestamp(),
            source_observation_timestamp=source_observation_timestamp,
            source_observation_timestep=source_observation_timestep,
            source_observation_age_ms=source_observation_age_ms,
            inference_start_time=inference_start_time,
            inference_end_time=inference_end_time,
            queue_size_before=queue_size_before_pop,
            queue_size_after=queue_size_after_pop,
            commanded_action=commanded_action,
            performed_action=_performed_action,
        )
        if self.config.display_data:
            log_rerun_data(action=_performed_action)

        if verbose:
            with self.action_queue_lock:
                current_queue_size = self.action_queue.qsize()

            self.logger.debug(
                f"Ts={timed_action.get_timestamp()} | "
                f"Action #{timed_action.get_timestep()} performed | "
                f"Queue size: {current_queue_size}"
            )

            self.logger.debug(
                f"Popping action from queue to perform took {get_end:.6f}s | Queue size: {current_queue_size}"
            )

        return _performed_action

    def _ready_to_send_observation(self):
        """Flags when the client is ready to send an observation"""
        with self.action_queue_lock:
            return self.action_queue.qsize() / self.action_chunk_size <= self._chunk_size_threshold

    def control_loop_observation(self, task: str, verbose: bool = False) -> RawObservation:
        try:
            # Get serialized observation bytes from the function
            start_time = time.perf_counter()
            capture_start_time = time.time()

            raw_observation: RawObservation = self.robot.get_observation()
            raw_observation["task"] = task

            if self.config.display_data:
                log_rerun_data(observation=raw_observation, compress_images=True)

            with self.latest_action_lock:
                latest_action = self.latest_action
                latest_action_wall_time = self.latest_action_wall_time

            observation = TimedObservation(
                timestamp=time.time(),  # need time.time() to compare timestamps across client and server
                observation=raw_observation,
                timestep=max(latest_action, 0),
            )

            obs_capture_time = time.perf_counter() - start_time
            capture_end_time = observation.get_timestamp()

            # If there are no actions left in the queue, the observation must go through processing!
            with self.action_queue_lock:
                observation.must_go = self.must_go.is_set() and self.action_queue.empty()
                current_queue_size = self.action_queue.qsize()
            observation.metadata.update(
                {
                    "capture_start_time": capture_start_time,
                    "capture_end_time": capture_end_time,
                    "queue_size_at_capture": current_queue_size,
                    "latest_action": latest_action,
                    "latest_action_time": latest_action_wall_time,
                    "must_go": int(observation.must_go),
                }
            )

            image_paths = save_observation_images(
                raw_observation,
                self.timeline_image_dir,
                observation.get_timestep(),
                mode=self.timeline_image_mode if self.config.record_timeline else "",
                key_frame=observation.must_go,
            )
            _ = self.send_observation(observation)
            self.latency_recorder.record(
                "client_capture_observation",
                timestep=observation.get_timestep(),
                capture_ms=obs_capture_time * 1000,
                queue_size=current_queue_size,
                must_go=int(observation.must_go),
            )
            state_lag_ms = (
                (capture_end_time - latest_action_wall_time) * 1000
                if isinstance(latest_action_wall_time, int | float)
                else None
            )
            self._record_timeline(
                "timeline_observation",
                timestep=observation.get_timestep(),
                start_time=capture_start_time,
                end_time=capture_end_time,
                observation_time=observation.get_timestamp(),
                latest_action=latest_action,
                latest_action_time=latest_action_wall_time,
                state_lag_ms=state_lag_ms,
                queue_size=current_queue_size,
                must_go=int(observation.must_go),
                image_paths=image_paths,
            )

            self.logger.debug(f"QUEUE SIZE: {current_queue_size} (Must go: {observation.must_go})")
            if observation.must_go:
                # must-go event will be set again after receiving actions
                self.must_go.clear()

            if verbose:
                # Calculate comprehensive FPS metrics
                fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())

                self.logger.info(
                    f"Obs #{observation.get_timestep()} | "
                    f"Avg FPS: {fps_metrics['avg_fps']:.2f} | "
                    f"Target: {fps_metrics['target_fps']:.2f}"
                )

                self.logger.debug(
                    f"Ts={observation.get_timestamp():.6f} | Capturing observation took {obs_capture_time:.6f}s"
                )

            return raw_observation

        except Exception as e:
            # Observation capture failures usually mean the robot or a camera is no longer providing
            # trustworthy state. Stop the async client immediately instead of continuing to execute
            # queued actions from stale observations.
            self.shutdown_event.set()
            self.logger.exception("Fatal error in observation sender; stopping robot client: %s", e)

    def control_loop(self, task: str, verbose: bool = False) -> tuple[Observation, Action]:
        """Combined function for executing actions and streaming observations"""
        # Wait at barrier for synchronized start
        self.start_barrier.wait()
        self.logger.info("Control loop thread starting")

        _performed_action = None
        _captured_observation = None

        while self.running:
            control_loop_start = time.perf_counter()
            """Control loop: (1) Performing actions, when available"""
            if self.actions_available():
                _performed_action = self.control_loop_action(verbose)

            """Control loop: (2) Streaming observations to the remote policy server"""
            if self._ready_to_send_observation():
                _captured_observation = self.control_loop_observation(task, verbose)

            self.logger.debug(f"Control loop (ms): {(time.perf_counter() - control_loop_start) * 1000:.2f}")
            self.latency_recorder.record(
                "client_control_loop",
                total_ms=(time.perf_counter() - control_loop_start) * 1000,
            )
            # Dynamically adjust sleep time to maintain the desired control frequency
            time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


@draccus.wrap()
def async_client(cfg: RobotClientConfig):
    logging.info(pformat(asdict(cfg)))

    if cfg.display_data:
        init_rerun(session_name="async_inference", ip=cfg.display_ip, port=cfg.display_port)

    # TODO: Assert if checking robot support is still needed with the plugin system
    # if cfg.robot.type not in SUPPORTED_ROBOTS:
    #     raise ValueError(f"Robot {cfg.robot.type} not yet supported!")

    client = RobotClient(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")

        # Create and start action receiver thread
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)

        # Start action receiver thread
        action_receiver_thread.start()

        try:
            # The main thread runs the control loop
            client.control_loop(task=cfg.task)

        finally:
            client.stop()
            action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            if cfg.display_data:
                shutdown_rerun()
            client.logger.info("Client stopped")
    else:
        client.stop()
        if cfg.display_data:
            shutdown_rerun()


if __name__ == "__main__":
    register_third_party_plugins()
    async_client()  # run the client
