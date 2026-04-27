#!/usr/bin/env python

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


@PreTrainedConfig.register_subclass("fast_wam")
@dataclass
class FastWAMConfig(PreTrainedConfig):
    """LeRobot policy wrapper for FastWAM checkpoints and training.

    The released FastWAM RoboTwin checkpoint predicts 32 normalized actions from
    three RGB views and a 14-D proprioceptive state. This config keeps those
    defaults while allowing LeRobot datasets such as SO101 recordings to infer
    their own action/state dimensions at training time.
    """

    n_obs_steps: int = 1
    chunk_size: int = 32
    n_action_steps: int = 8
    num_frames: int = 33
    action_video_freq_ratio: int = 4

    action_dim: int = 14
    proprio_dim: int = 14
    image_size: tuple[int, int] = (384, 320)
    concat_multi_camera: str = "horizontal"
    camera_keys: list[str] | None = None
    auto_configure_dims: bool = True

    high_camera_key: str = f"{OBS_IMAGES}.cam_high"
    left_camera_key: str = f"{OBS_IMAGES}.cam_left_wrist"
    right_camera_key: str = f"{OBS_IMAGES}.cam_right_wrist"
    composite_image_key: str | None = None
    state_key: str = OBS_STATE
    task_key: str = "task"

    prompt_template: str = (
        "A video recorded from a robot's point of view executing the following instruction: {task}"
    )
    default_task: str = ""

    fastwam_root: Path | None = None
    fastwam_config_name: str = "sim_robotwin.yaml"
    fastwam_task: str | None = None
    fastwam_checkpoint_path: Path | None = None
    dataset_stats_path: Path | None = None
    release_checkpoint_name: str = "robotwin_uncond_3cam_384"
    auto_find_fastwam_artifacts: bool = True

    mixed_precision: str = "bf16"
    dtype: str | None = None
    num_inference_steps: int = 10
    sigma_shift: float | None = None
    seed: int | None = None
    text_cfg_scale: float = 1.0
    negative_prompt: str = ""
    rand_device: str = "cpu"
    tiled: bool = False
    collect_timing: bool = False

    load_text_encoder: bool = True
    text_encoder_device: str | None = None
    vae_device: str | None = None
    mot_layer_devices: str | None = None
    skip_dit_load_from_pretrain: bool = True
    action_dit_pretrained_path: Path | None = None
    diffsynth_model_base_path: Path | None = None
    gradient_checkpointing: bool = False
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    ignore_mismatched_checkpoint_shapes: bool = True
    norm_default_mode: str = "z-score"

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 1e-2

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) cannot be greater than "
                f"chunk_size ({self.chunk_size})."
            )
        if self.chunk_size != self.num_frames - 1:
            raise ValueError(
                f"FastWAM expects chunk_size ({self.chunk_size}) to equal num_frames - 1 "
                f"({self.num_frames - 1})."
            )
        if (self.num_frames - 1) % self.action_video_freq_ratio != 0:
            raise ValueError(
                "num_frames - 1 must be divisible by action_video_freq_ratio, got "
                f"{self.num_frames - 1} and {self.action_video_freq_ratio}."
            )
        num_video_frames = (self.num_frames - 1) // self.action_video_freq_ratio + 1
        if num_video_frames % 4 != 1:
            raise ValueError(
                "FastWAM video frames must satisfy T % 4 == 1 after temporal sampling, got "
                f"{num_video_frames}."
            )
        if self.dtype is not None:
            dtype = self.dtype.lower()
            dtype_to_precision = {
                "bfloat16": "bf16",
                "bf16": "bf16",
                "float16": "fp16",
                "fp16": "fp16",
                "float32": "no",
                "fp32": "no",
                "no": "no",
            }
            if dtype not in dtype_to_precision:
                raise ValueError(
                    "dtype must be one of: 'bfloat16', 'bf16', 'float16', 'fp16', 'float32', 'fp32', 'no'."
                )
            self.mixed_precision = dtype_to_precision[dtype]
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("mixed_precision must be one of: 'no', 'fp16', 'bf16'.")
        if self.concat_multi_camera not in {"horizontal", "vertical", "robotwin"}:
            raise ValueError("concat_multi_camera must be one of: 'horizontal', 'vertical', 'robotwin'.")
        if self.norm_default_mode not in {"min/max", "q01/q99", "z-score"}:
            raise ValueError("norm_default_mode must be one of: 'min/max', 'q01/q99', 'z-score'.")

    def validate_features(self) -> None:
        if not self.input_features:
            self.input_features = {
                self.high_camera_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
                self.left_camera_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
                self.right_camera_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
                self.state_key: PolicyFeature(type=FeatureType.STATE, shape=(self.proprio_dim,)),
            }

        if not self.output_features:
            self.output_features = {
                ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,)),
            }

        if self.auto_configure_dims and self.action_feature is not None:
            self.action_dim = int(self.action_feature.shape[-1])

        state_feature = self.input_features.get(self.state_key) if self.input_features else None
        if self.auto_configure_dims and state_feature is not None:
            self.proprio_dim = int(state_feature.shape[-1])

        if self.action_feature is not None and self.action_feature.shape[-1] != self.action_dim:
            raise ValueError(
                f"FastWAM action_dim={self.action_dim}, but output feature '{ACTION}' has "
                f"shape {self.action_feature.shape}."
            )

        if state_feature is not None and state_feature.shape[-1] != self.proprio_dim:
            raise ValueError(
                f"FastWAM proprio_dim={self.proprio_dim}, but state feature '{self.state_key}' has "
                f"shape {state_feature.shape}."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay)

    def get_scheduler_preset(self) -> None:
        return None

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(self.num_frames))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
