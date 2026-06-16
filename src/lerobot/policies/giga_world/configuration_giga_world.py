#!/usr/bin/env python

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


@PreTrainedConfig.register_subclass("giga_world")
@dataclass
class GigaWorldConfig(PreTrainedConfig):
    """LeRobot wrapper around GigaWorld-Policy's action-centered WAM.

    GigaWorld trains on a short video condition and a longer action chunk.  The
    defaults mirror the public example config in ``giga-world-policy``:
    5 visual frames sampled over a 48-step action horizon, a WAN2.2 TI2V VAE/text
    stack, and a causal world-action transformer.
    """

    n_obs_steps: int = 1
    chunk_size: int = 48
    n_action_steps: int = 16

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    # Dataset/key configuration.
    state_key: str = OBS_STATE
    action_key: str = ACTION
    task_key: str = "task"
    view_keys: list[str] | None = None
    default_task: str = ""
    visual_frame_offsets: list[int] | None = None

    # Action/state dimensions are inferred from the LeRobot dataset when possible.
    action_dim: int = 14
    state_dim: int = 14
    auto_configure_dims: bool = True
    delta_mask: list[bool] | None = None

    # GigaWorld / WAN runtime.
    giga_world_root: Path | None = Path("~/Projects/giga-world-policy")
    model_cache_dir: Path = Path("~/Projects/models/giga-world-policy")
    wan_model_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    transformer_path: Path | None = None
    base_transformer_path: Path | None = None
    transformer_checkpoint_path: Path | None = None
    norm_stats_path: Path | None = None

    # Image/prompt/inference sizing.
    per_view_size: tuple[int, int] = (256, 192)  # width, height per camera view
    crop_mode: str = "center"
    t5_len: int = 64
    text_encoder_max_length: int = 512
    num_inference_frames: int = 5
    num_inference_steps: int = 10
    guidance_scale: float = 0.0
    flow_shift: float = 5.0
    expand_timesteps: bool = True
    torch_dtype: str = "bfloat16"

    # Training objective weights.
    action_loss_weight: float = 1.0
    visual_loss_weight: float = 1.0

    # Fine-tuning defaults: train action heads plus LoRA on WAN transformer blocks.
    freeze_vae: bool = True
    train_action_heads: bool = True
    freeze_transformer_backbone: bool = True
    use_transformer_lora: bool = True
    transformer_lora_path: Path | None = None
    lora_rank: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(default_factory=lambda: ["to_q", "to_k", "to_v"])
    reinit_action_heads: bool = True
    gradient_checkpointing: bool = False

    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-2
    optimizer_grad_clip_norm: float = 10.0
    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.n_action_steps > self.chunk_size:
            raise ValueError("n_action_steps must be <= chunk_size.")
        if self.crop_mode not in {"center", "random"}:
            raise ValueError("crop_mode must be either 'center' or 'random'.")
        if self.torch_dtype not in {"bfloat16", "float16", "float32", "bf16", "fp16", "fp32"}:
            raise ValueError("torch_dtype must be one of bfloat16/float16/float32 or bf16/fp16/fp32.")

    def validate_features(self) -> None:
        if not self.input_features:
            self.input_features = {
                f"{OBS_IMAGES}.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
                f"{OBS_IMAGES}.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
                self.state_key: PolicyFeature(type=FeatureType.STATE, shape=(self.state_dim,)),
            }
        if not self.output_features:
            self.output_features = {
                self.action_key: PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,)),
            }

        if self.auto_configure_dims and self.action_feature is not None:
            self.action_dim = int(self.action_feature.shape[-1])

        state_feature = self.input_features.get(self.state_key) if self.input_features else None
        if self.auto_configure_dims and state_feature is not None:
            self.state_dim = int(state_feature.shape[-1])

        if self.view_keys is None:
            self.view_keys = list(self.image_features.keys())
        if not self.view_keys:
            raise ValueError("GigaWorld requires at least one visual feature.")
        if self.state_key not in self.input_features:
            raise ValueError(f"GigaWorld requires state feature '{self.state_key}'.")
        if self.action_feature is None:
            raise ValueError("GigaWorld requires an action output feature.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self) -> CosineDecayWithWarmupSchedulerConfig:
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list[int]:
        if self.visual_frame_offsets is not None:
            return list(self.visual_frame_offsets)
        return sorted({0, self.chunk_size // 4, self.chunk_size // 2, 3 * self.chunk_size // 4, self.chunk_size})

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
