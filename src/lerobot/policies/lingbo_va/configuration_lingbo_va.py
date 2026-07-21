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
    num_frames: int = 25
    frame_chunk_size: int = 4
    action_per_frame: int = 8
    vae_temporal_stride: int = 4
    video_frame_stride: int | None = None

    action_dim: int = 6
    lingbo_action_dim: int = 30
    used_action_channel_ids: list[int] | None = None
    auto_configure_dims: bool = True

    image_size: tuple[int, int] = (256, 256)
    camera_keys: list[str] | None = field(
        default_factory=lambda: [f"{OBS_IMAGES}.front", f"{OBS_IMAGES}.right"]
    )
    server_camera_keys: list[str] | None = None
    top_camera_key: str = f"{OBS_IMAGES}.top"
    wrist_camera_key: str = f"{OBS_IMAGES}.wrist"
    front_camera_key: str = f"{OBS_IMAGES}.front"
    right_camera_key: str = f"{OBS_IMAGES}.right"
    state_key: str = OBS_STATE
    task_key: str = "task"

    prompt_template: str = "{task}"
    default_task: str = ""

    lingbo_root: Path | None = Path("/home/rxhuang/Projects/lingbot-va")
    model_root: Path | None = Path("/home/rxhuang/Projects/models/lingbot-va-posttrain-robotwin")
    transformer_path: Path | None = None
    va_config_name: str = "so101"
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
    train_video_head: bool = False
    train_last_n_blocks: int = 0
    use_transformer_lora: bool = False
    transformer_lora_path: Path | None = None
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(default_factory=lambda: ["to_q", "to_v"])
    lora_optimizer_lr: float = 1e-4
    lora_optimizer_weight_decay: float = 0.0
    freeze_vision_text_encoder: bool = True
    vae_device: str | None = None
    vae_encode_batch_size: int = 1
    text_encoder_device: str | None = "cpu"
    transformer_device: str | None = None
    offload_vae_after_encode: bool = True
    offload_text_encoder_after_encode: bool = True
    gradient_checkpointing: bool = True

    norm_default_mode: str = "q01/q99"
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-1
    loss_log_freq: int = 20

    # Classifier-free-guidance text dropout probability during training. Upstream
    # (`lerobot_latent_dataset.py`) applies this per-sample in the dataset; this port applies it
    # inline in `_build_training_input` since there is no offline-cached-latent dataset stage.
    cfg_prob: float = 0.0
    # Loss combination weights. Upstream default (robotwin/libero, unset) is 1.0/1.0; the
    # already-validated SO101 config overrides to 0.1/1.0 to keep video reconstruction loss from
    # dominating the (much smaller) action loss on a small, single-task dataset.
    video_loss_weight: float = 1.0
    action_loss_weight: float = 1.0

    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    # Optional explicit path to a `lingbo_va_action_stats.json`-formatted file (see
    # `_save_action_stats`/`_load_action_stats`) to use instead of `dataset_stats`. Used to force
    # training to use train-only-scoped normalization stats (e.g. computed via
    # `scripts/compute_lingbo_va_train_stats.py` over a subset of episodes), and to guarantee
    # deploy-time inference loads the exact same file via the checkpoint's saved copy.
    action_stats_path: Path | None = None

    # --- Training-time validation (see LingBoVAPolicy.run_lingbo_va_validation / eval_utils.py) ---
    # Episode indices held out from training, used only by the validation callback. Never
    # included in `--dataset.episodes`. Validation is disabled entirely when this is None/empty.
    val_episodes: list[int] | None = None
    # Fixed-timestep val loss + teacher-forced action-curve eval run every `val_freq` steps.
    val_freq: int = 500
    val_num_samples: int = 32
    val_seed: int = 12345
    val_timestep_fractions: tuple[float, ...] = (0.25, 0.5, 0.75)
    teacher_forced_episode: int = 95
    teacher_forced_offsets: list[int] = field(
        default_factory=lambda: [0, 50, 100, 150, 200, 250, 300, 350]
    )
    # Full-episode open-loop stride-requery eval runs every `open_loop_freq` steps (and, driven
    # by the caller, on every checkpoint save).
    open_loop_freq: int = 2000
    open_loop_episodes: list[int] = field(default_factory=lambda: [95, 97])
    open_loop_stride: int = 8

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
        if self.vae_encode_batch_size <= 0:
            raise ValueError("vae_encode_batch_size must be positive.")
        if self.loss_log_freq < 0:
            raise ValueError("loss_log_freq cannot be negative.")
        if self.lora_rank <= 0:
            raise ValueError("lora_rank must be positive.")
        if self.lora_alpha <= 0:
            raise ValueError("lora_alpha must be positive.")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError("lora_dropout must be in [0, 1).")
        if self.lora_optimizer_lr <= 0:
            raise ValueError("lora_optimizer_lr must be positive.")
        if self.lora_optimizer_weight_decay < 0:
            raise ValueError("lora_optimizer_weight_decay cannot be negative.")
        if self.use_transformer_lora and not self.lora_target_modules:
            raise ValueError("lora_target_modules cannot be empty when use_transformer_lora is enabled.")
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
        if self.train_last_n_blocks < 0:
            raise ValueError("train_last_n_blocks cannot be negative.")
        if not 0.0 <= self.cfg_prob < 1.0:
            raise ValueError("cfg_prob must be in [0, 1).")
        if self.video_loss_weight < 0:
            raise ValueError("video_loss_weight cannot be negative.")
        if self.action_loss_weight < 0:
            raise ValueError("action_loss_weight cannot be negative.")
        if self.scheduler_warmup_steps < 0:
            raise ValueError("scheduler_warmup_steps cannot be negative.")
        if self.val_freq < 0 or self.open_loop_freq < 0:
            raise ValueError("val_freq and open_loop_freq cannot be negative.")
        if self.val_num_samples <= 0:
            raise ValueError("val_num_samples must be positive.")

    def validate_features(self) -> None:
        if not self.input_features:
            camera_keys = self.camera_keys or [self.front_camera_key, self.right_camera_key]
            self.input_features = {
                key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)) for key in camera_keys
            }
            self.input_features[self.state_key] = PolicyFeature(
                type=FeatureType.STATE, shape=(self.action_dim,)
            )

        if not self.output_features:
            self.output_features = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))}

        if self.auto_configure_dims and self.action_feature is not None:
            self.action_dim = int(self.action_feature.shape[-1])

        if self.used_action_channel_ids is None:
            self.used_action_channel_ids = list(range(self.action_dim))

        if self.camera_keys is not None and len(set(self.camera_keys)) != len(self.camera_keys):
            raise ValueError("camera_keys must not contain duplicates.")
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

    def get_scheduler_preset(self):
        from lerobot.optim.schedulers import DiffuserSchedulerConfig

        return DiffuserSchedulerConfig(name=self.scheduler_name, num_warmup_steps=self.scheduler_warmup_steps)

    @property
    def observation_delta_indices(self) -> list[int]:
        return list(range(self.num_frames))

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
