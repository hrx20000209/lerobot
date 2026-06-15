#!/usr/bin/env python

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


@PreTrainedConfig.register_subclass("lingbo_va")
@dataclass
class LingBoVAConfig(PreTrainedConfig):
    """LeRobot wrapper for LingBot-VA video-action models.

    LingBot-VA's released transformer uses a 30-channel action tensor. The
    LeRobot wrapper maps the dataset action dimensions into configurable model
    channels, leaving the remaining channels masked out.
    """

    n_obs_steps: int = 1
    chunk_size: int = 32
    n_action_steps: int = 8
    num_frames: int = 33
    frame_chunk_size: int = 4
    action_per_frame: int = 8
    vae_temporal_stride: int = 4
    video_frame_stride: int | None = None

    action_dim: int = 6
    lingbo_action_dim: int = 30
    used_action_channel_ids: list[int] | None = None
    auto_configure_dims: bool = True

    image_size: tuple[int, int] = (256, 256)
    camera_keys: list[str] | None = None
    server_camera_keys: list[str] | None = None
    top_camera_key: str = f"{OBS_IMAGES}.top"
    wrist_camera_key: str = f"{OBS_IMAGES}.wrist"
    front_camera_key: str = f"{OBS_IMAGES}.front"
    state_key: str = OBS_STATE
    task_key: str = "task"

    prompt_template: str = "{task}"
    default_task: str = ""

    lingbo_root: Path | None = Path("/home/rxhuang/Projects/lingbot-va")
    model_root: Path | None = Path("/home/rxhuang/Projects/models/lingbot-va-posttrain-robotwin")
    transformer_path: Path | None = None
    va_config_name: str = "demo"
    env_type: str = "none"
    attn_window: int = 30
    save_root: Path = Path("./lingbo_va_runtime")
    dtype: str = "bfloat16"
    train_attn_mode: str = "flex"
    inference_attn_mode: str = "torch"

    inference_backend: str = "server"  # server or local
    server_host: str = "127.0.0.1"
    server_port: int = 29056
    connect_retry_s: float = 2.0
    video_guidance_scale: float = 5.0
    action_guidance_scale: float = 1.0
    video_inference_steps: int | None = None
    action_inference_steps: int | None = None
    video_exec_step: int | None = None
    return_server_profile: bool = False

    train_action_head_only: bool = True
    freeze_vision_text_encoder: bool = True
    vae_device: str | None = None
    text_encoder_device: str | None = "cpu"
    transformer_device: str | None = None
    offload_vae_after_encode: bool = True
    offload_text_encoder_after_encode: bool = True
    gradient_checkpointing: bool = True

    norm_default_mode: str = "q01/q99"
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-1

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.chunk_size != self.frame_chunk_size * self.action_per_frame:
            raise ValueError(
                "chunk_size must equal frame_chunk_size * action_per_frame, got "
                f"{self.chunk_size} vs {self.frame_chunk_size} * {self.action_per_frame}."
            )
        if self.vae_temporal_stride <= 0:
            raise ValueError("vae_temporal_stride must be positive.")
        if self.video_frame_stride is None:
            if self.action_per_frame % self.vae_temporal_stride != 0:
                raise ValueError(
                    "action_per_frame must be divisible by vae_temporal_stride when "
                    "video_frame_stride is not set."
                )
            self.video_frame_stride = max(1, self.action_per_frame // self.vae_temporal_stride)
        if self.video_frame_stride <= 0:
            raise ValueError("video_frame_stride must be positive.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps cannot be greater than chunk_size.")
        required_frames = (self.frame_chunk_size - 1) * self.video_frame_stride * self.vae_temporal_stride + 1
        if self.num_frames < required_frames:
            raise ValueError(
                "num_frames must cover the video frames needed by the Wan VAE, got "
                f"{self.num_frames} but need at least {required_frames}."
            )
        if self.inference_backend not in {"server", "local"}:
            raise ValueError("inference_backend must be one of: 'server', 'local'.")
        if self.dtype not in {"bfloat16", "bf16", "float16", "fp16", "float32", "fp32"}:
            raise ValueError("dtype must be one of: bfloat16/bf16/float16/fp16/float32/fp32.")
        if self.train_attn_mode not in {"flex", "torch", "flashattn"}:
            raise ValueError("train_attn_mode must be one of: flex, torch, flashattn.")
        if self.inference_attn_mode not in {"torch", "flashattn", "flex"}:
            raise ValueError("inference_attn_mode must be one of: torch, flashattn, flex.")
        if self.norm_default_mode not in {"q01/q99", "min/max", "z-score"}:
            raise ValueError("norm_default_mode must be one of: q01/q99, min/max, z-score.")

    def validate_features(self) -> None:
        if not self.input_features:
            self.input_features = {
                self.top_camera_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
                self.wrist_camera_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
                self.state_key: PolicyFeature(type=FeatureType.STATE, shape=(self.action_dim,)),
            }

        if not self.output_features:
            self.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))}

        if self.auto_configure_dims and self.action_feature is not None:
            self.action_dim = int(self.action_feature.shape[-1])

        if self.used_action_channel_ids is None:
            self.used_action_channel_ids = list(range(self.action_dim))

        if len(self.used_action_channel_ids) != self.action_dim:
            raise ValueError(
                "used_action_channel_ids length must match dataset action_dim, got "
                f"{len(self.used_action_channel_ids)} and {self.action_dim}."
            )
        if max(self.used_action_channel_ids, default=-1) >= self.lingbo_action_dim:
            raise ValueError("used_action_channel_ids must be smaller than lingbo_action_dim.")

        if self.action_feature is not None and self.action_feature.shape[-1] != self.action_dim:
            raise ValueError(
                f"LingBoVA action_dim={self.action_dim}, but output feature '{ACTION}' has "
                f"shape {self.action_feature.shape}."
            )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(lr=self.optimizer_lr, weight_decay=self.optimizer_weight_decay, betas=(0.9, 0.95))

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
