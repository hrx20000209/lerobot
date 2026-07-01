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

import json
import logging
import logging.handlers
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from lerobot.configs import PolicyFeature

# NOTE: Configs need to be loaded for the client to be able to instantiate the policy config
from lerobot.policies import (  # noqa: F401
    ACTConfig,
    DiffusionConfig,
    FastWAMConfig,
    GigaWorldConfig,
    PI0Config,
    PI05Config,
    SmolVLAConfig,
    VLAJEPAConfig,
    VQBeTConfig,
)
from lerobot.robots.robot import Robot
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features
from lerobot.utils.utils import init_logging

Action = torch.Tensor

# observation as received from the robot (can be numpy arrays, floats, etc.)
RawObservation = dict[str, Any]

# observation as those recorded in LeRobot dataset (keys are different)
LeRobotObservation = dict[str, torch.Tensor]

# observation, ready for policy inference (image keys resized)
Observation = dict[str, torch.Tensor]


class LatencyRecorder:
    """Thread-safe JSONL latency recorder with simple aggregate summaries."""

    def __init__(self, name: str, log_dir: str | Path = "logs", enabled: bool = True):
        self.name = name
        self.enabled = enabled
        self.records: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = time.strftime("%Y%m%d_%H%M%S")
        self.path = self.log_dir / f"{name}_latency_{self.run_id}.jsonl"
        self.summary_path = self.log_dir / f"{name}_latency_summary_{self.run_id}.json"
        self._file = self.path.open("a", encoding="utf-8") if enabled else None

    def _sync_cuda(self, device: str | torch.device | None = None) -> None:
        if not torch.cuda.is_available():
            return
        if device is None:
            torch.cuda.synchronize()
            return
        device_str = str(device)
        if device_str.startswith("cuda"):
            torch.cuda.synchronize(torch.device(device_str))

    def now(self, device: str | torch.device | None = None) -> float:
        self._sync_cuda(device)
        return time.perf_counter()

    def elapsed_ms(self, start: float, device: str | torch.device | None = None) -> float:
        self._sync_cuda(device)
        return (time.perf_counter() - start) * 1000

    def record(self, kind: str, **metrics: Any) -> dict[str, Any]:
        record = {
            "time": time.time(),
            "kind": kind,
            **metrics,
        }
        if not self.enabled:
            return record

        with self._lock:
            self.records.append(record)
            if self._file is not None and not self._file.closed:
                self._file.write(json.dumps(record, default=str) + "\n")
                self._file.flush()
        return record

    @staticmethod
    def _is_summary_number(key: str, value: Any) -> bool:
        time_like_keys = {
            "time",
            "timestamp",
            "start_time",
            "end_time",
            "capture_start_time",
            "capture_end_time",
            "observation_time",
            "execution_start_time",
            "execution_end_time",
            "inference_start_time",
            "inference_end_time",
            "received_time",
            "latest_action_time",
            "source_observation_timestamp",
        }
        step_like_keys = {
            "timestep",
            "first_timestep",
            "last_timestep",
            "latest_action",
            "source_observation_timestep",
        }
        if key in time_like_keys or key in step_like_keys or key.endswith("_timestamp"):
            return False
        return isinstance(value, int | float) and not isinstance(value, bool)

    def summary(self) -> dict[str, dict[str, dict[str, float | int]]]:
        grouped: dict[str, dict[str, list[float]]] = {}
        with self._lock:
            records = list(self.records)
        for record in records:
            kind = str(record.get("kind", "unknown"))
            grouped.setdefault(kind, {})
            for key, value in record.items():
                if self._is_summary_number(key, value):
                    grouped[kind].setdefault(key, []).append(float(value))

        summary: dict[str, dict[str, dict[str, float | int]]] = {}
        for kind, metrics in grouped.items():
            summary[kind] = {}
            for key, values in metrics.items():
                values_sorted = sorted(values)
                count = len(values_sorted)
                if count == 0:
                    continue
                p50 = values_sorted[count // 2]
                p90 = values_sorted[min(count - 1, int(count * 0.9))]
                summary[kind][key] = {
                    "count": count,
                    "mean": sum(values_sorted) / count,
                    "p50": p50,
                    "p90": p90,
                    "max": max(values_sorted),
                }
        return summary

    def write_summary(self) -> Path:
        summary = self.summary()
        self.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return self.summary_path

    def log_summary(self, logger: logging.Logger) -> None:
        summary_path = self.write_summary()
        logger.info(f"Latency records saved to {self.path}")
        logger.info(f"Latency summary saved to {summary_path}")
        summary = self.summary()
        for kind, metrics in summary.items():
            logger.info(f"Latency summary: {kind}")
            for key, stats in sorted(metrics.items()):
                logger.info(
                    f"  {key}: mean={stats['mean']:.2f}, p50={stats['p50']:.2f}, "
                    f"p90={stats['p90']:.2f}, max={stats['max']:.2f}, n={stats['count']}"
                )

    def close(self) -> None:
        with self._lock:
            if self._file is not None and not self._file.closed:
                self._file.close()


def visualize_action_queue_size(action_queue_size: list[int]) -> None:
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    ax.set_title("Action Queue Size Over Time")
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Action Queue Size")
    ax.set_ylim(0, max(action_queue_size) * 1.1)
    ax.grid(True, alpha=0.3)
    ax.plot(range(len(action_queue_size)), action_queue_size)
    plt.show()


def map_robot_keys_to_lerobot_features(robot: Robot) -> dict[str, dict]:
    return hw_to_dataset_features(robot.observation_features, OBS_STR, use_video=False)


def is_image_key(k: str) -> bool:
    return k.startswith(OBS_IMAGES)


def _safe_filename_fragment(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "image"


def _image_array_or_none(value: Any):
    try:
        import numpy as np
    except ImportError:
        return None

    if isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
    elif isinstance(value, np.ndarray):
        array = value
    else:
        return None

    if array.ndim != 3:
        return None

    if array.shape[0] in {1, 3, 4} and array.shape[-1] not in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)

    if array.shape[-1] not in {1, 3, 4}:
        return None

    if array.dtype.kind == "f" and array.size > 0 and array.max() <= 1.0:
        array = np.clip(array, 0.0, 1.0) * 255.0
    else:
        array = np.clip(array, 0, 255)
    return array.astype(np.uint8)


def save_observation_images(
    raw_observation: RawObservation,
    output_dir: str | Path,
    timestep: int,
    mode: str | None = None,
    key_frame: bool = False,
) -> list[str]:
    """Optionally persist raw observation images for timeline inspection.

    mode:
        off/empty: do not save images
        key: save only key frames, currently observations marked must_go
        all/true/1: save every observation with image-like arrays
    """
    save_mode = (mode or "").strip().lower()
    if save_mode in {"", "0", "false", "off", "none", "no"}:
        return []
    if save_mode == "key" and not key_frame:
        return []

    try:
        from PIL import Image
    except ImportError:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for key, value in raw_observation.items():
        array = _image_array_or_none(value)
        if array is None:
            continue

        filename = f"obs_{timestep:06d}_{_safe_filename_fragment(str(key))}.jpg"
        path = output_dir / filename
        if array.shape[-1] == 1:
            array = array[..., 0]
        Image.fromarray(array).save(path, quality=90)
        image_paths.append(str(path))

    return image_paths


def resize_robot_observation_image(image: torch.tensor, resize_dims: tuple[int, int, int]) -> torch.tensor:
    assert image.ndim == 3, f"Image must be (C, H, W)! Received {image.shape}"
    # (H, W, C) -> (C, H, W) for resizing from robot obsevation resolution to policy image resolution
    image = image.permute(2, 0, 1)
    dims = (resize_dims[1], resize_dims[2])
    # Add batch dimension for interpolate: (C, H, W) -> (1, C, H, W)
    image_batched = image.unsqueeze(0)
    # Interpolate and remove batch dimension: (1, C, H, W) -> (C, H, W)
    resized = torch.nn.functional.interpolate(image_batched, size=dims, mode="bilinear", align_corners=False)

    return resized.squeeze(0)


# TODO(Steven): Consider implementing a pipeline step for this
def raw_observation_to_observation(
    raw_observation: RawObservation,
    lerobot_features: dict[str, dict],
    policy_image_features: dict[str, PolicyFeature],
) -> Observation:
    observation = {}

    observation = prepare_raw_observation(raw_observation, lerobot_features, policy_image_features)
    for k, v in observation.items():
        if isinstance(v, torch.Tensor):  # VLAs present natural-language instructions in observations
            if "image" in k:
                # Policy expects images in shape (B, C, H, W)
                observation[k] = prepare_image(v).unsqueeze(0)
        else:
            observation[k] = v

    return observation


def prepare_image(image: torch.Tensor) -> torch.Tensor:
    """Minimal preprocessing to turn int8 images to float32 in [0, 1], and create a memory-contiguous tensor"""
    image = image.type(torch.float32) / 255
    image = image.contiguous()

    return image


def extract_state_from_raw_observation(
    lerobot_obs: RawObservation,
) -> torch.Tensor:
    """Extract the state from a raw observation."""
    state = torch.tensor(lerobot_obs[OBS_STATE])

    if state.ndim == 1:
        state = state.unsqueeze(0)

    return state


def extract_images_from_raw_observation(
    lerobot_obs: RawObservation,
    camera_key: str,
) -> dict[str, torch.Tensor]:
    """Extract the images from a raw observation."""
    return torch.tensor(lerobot_obs[camera_key])


def make_lerobot_observation(
    robot_obs: RawObservation,
    lerobot_features: dict[str, dict],
) -> LeRobotObservation:
    """Make a lerobot observation from a raw observation."""
    return build_dataset_frame(lerobot_features, robot_obs, prefix=OBS_STR)


def prepare_raw_observation(
    robot_obs: RawObservation,
    lerobot_features: dict[str, dict],
    policy_image_features: dict[str, PolicyFeature],
) -> Observation:
    """Matches keys from the raw robot_obs dict to the keys expected by a given policy (passed as
    policy_image_features)."""
    # 1. {motor.pos1:value1, motor.pos2:value2, ..., laptop:np.ndarray} ->
    # -> {observation.state:[value1,value2,...], observation.images.laptop:np.ndarray}
    lerobot_obs = make_lerobot_observation(robot_obs, lerobot_features)

    # 2. Keep only the image keys the active policy expects. The robot may expose
    # extra cameras (for example right/front/wrist) while a checkpoint was trained
    # with only a subset (for example front+wrist).
    robot_image_keys = set(filter(is_image_key, lerobot_obs))
    image_keys = list(policy_image_features.keys())
    missing_image_keys = [key for key in image_keys if key not in robot_image_keys]
    if missing_image_keys:
        raise KeyError(
            f"Robot observation is missing image keys required by the policy: {missing_image_keys}. "
            f"Available image keys: {sorted(robot_image_keys)}"
        )
    # state's shape is expected as (B, state_dim)
    state_dict = {OBS_STATE: extract_state_from_raw_observation(lerobot_obs)}
    image_dict = {
        image_k: extract_images_from_raw_observation(lerobot_obs, image_k) for image_k in image_keys
    }

    # Turns the image features to (C, H, W) with H, W matching the policy image features.
    # This reduces the resolution of the images
    image_dict = {
        key: resize_robot_observation_image(torch.tensor(lerobot_obs[key]), policy_image_features[key].shape)
        for key in image_keys
    }

    if "task" in robot_obs:
        state_dict["task"] = robot_obs["task"]

    return {**state_dict, **image_dict}


def get_logger(name: str, log_to_file: bool = True) -> logging.Logger:
    """
    Get a logger using the standardized logging setup from utils.py.

    Args:
        name: Logger name (e.g., 'policy_server', 'robot_client')
        log_to_file: Whether to also log to a file

    Returns:
        Configured logger instance
    """
    # Create logs directory if logging to file
    if log_to_file:
        os.makedirs("logs", exist_ok=True)
        log_file = Path(f"logs/{name}_{int(time.time())}.log")
    else:
        log_file = None

    # Initialize the standardized logging
    init_logging(log_file=log_file, display_pid=False)

    # Return a named logger
    return logging.getLogger(name)


@dataclass
class TimedData:
    """A data object with timestamp and timestep information.

    Args:
        timestamp: Unix timestamp relative to data's creation.
        data: The actual data to wrap a timestamp around.
        timestep: The timestep of the data.
    """

    timestamp: float
    timestep: int

    def get_timestamp(self):
        return self.timestamp

    def get_timestep(self):
        return self.timestep


@dataclass
class TimedAction(TimedData):
    action: Action
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_action(self):
        return self.action

    def get_metadata(self):
        return self.metadata


@dataclass
class TimedObservation(TimedData):
    observation: RawObservation
    must_go: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_observation(self):
        return self.observation

    def get_metadata(self):
        return self.metadata


@dataclass
class FPSTracker:
    """Utility class to track FPS metrics over time."""

    target_fps: float
    first_timestamp: float = None
    total_obs_count: int = 0

    def calculate_fps_metrics(self, current_timestamp: float) -> dict[str, float]:
        """Calculate average FPS vs target"""
        self.total_obs_count += 1

        # Initialize first observation time
        if self.first_timestamp is None:
            self.first_timestamp = current_timestamp

        # Calculate overall average FPS (since start)
        total_duration = current_timestamp - self.first_timestamp
        avg_fps = (self.total_obs_count - 1) / total_duration if total_duration > 1e-6 else 0.0

        return {"avg_fps": avg_fps, "target_fps": self.target_fps}

    def reset(self):
        """Reset the FPS tracker state"""
        self.first_timestamp = None
        self.total_obs_count = 0


@dataclass
class RemotePolicyConfig:
    policy_type: str
    pretrained_name_or_path: str
    lerobot_features: dict[str, PolicyFeature]
    actions_per_chunk: int
    device: str = "cpu"
    rename_map: dict[str, str] = field(default_factory=dict)
    task: str = ""


def _compare_observation_states(obs1_state: torch.Tensor, obs2_state: torch.Tensor, atol: float) -> bool:
    """Check if two observation states are similar, under a tolerance threshold"""
    return bool(torch.linalg.norm(obs1_state - obs2_state) < atol)


def observations_similar(
    obs1: TimedObservation, obs2: TimedObservation, lerobot_features: dict[str, dict], atol: float = 1
) -> bool:
    """Check if two observations are similar, under a tolerance threshold. Measures distance between
    observations as the difference in joint-space between the two observations.

    NOTE(fracapuano): This is a very simple check, and it is enough for the current use case.
    An immediate next step is to use (fast) perceptual difference metrics comparing some camera views,
    to surpass this joint-space similarity check.
    """
    obs1_state = extract_state_from_raw_observation(
        make_lerobot_observation(obs1.get_observation(), lerobot_features)
    )
    obs2_state = extract_state_from_raw_observation(
        make_lerobot_observation(obs2.get_observation(), lerobot_features)
    )

    return _compare_observation_states(obs1_state, obs2_state, atol=atol)
