#!/usr/bin/env python

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


@PreTrainedConfig.register_subclass("cosmos")
@dataclass
class CosmosConfig(PreTrainedConfig):
    """LeRobot policy wrapper for the local Cosmos SO101 action model."""

    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 16
    actions_per_chunk: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        }
    )

    cosmos_repo: str = "/home/rxhuang/Projects/cosmos-policy"
    model_path: str | None = None
    ckpt_path: str | None = None
    fine_tuned_weights_path: str | None = None
    config_name: str = "cosmos_predict2_2b_480p_so101_lerobot"
    config_file: str = "cosmos_policy/config/config.py"
    dataset_root: str = "/data/rxhuang/three_cubes_1"
    dataset_repo_id: str = "local/three_cubes_1"
    dataset_stats_path: str = "/data/rxhuang/three_cubes_1/so101_dataset_statistics.json"
    t5_text_embeddings_path: str = "/data/rxhuang/three_cubes_1/so101_t5_embeddings.pkl"

    action_dim: int = 6
    state_dim: int = 6
    final_image_size: int = 224
    num_denoising_steps_action: int = 10
    camera_keys: list[str] = field(
        default_factory=lambda: [
            "observation.images.front",
            "observation.images.right",
            "observation.images.wrist",
        ]
    )
    primary_camera_key: str = "observation.images.front"
    wrist_camera_key: str = "observation.images.wrist"
    wrist_left_camera_key: str = "observation.images.right"
    state_key: str = OBS_STATE
    action_key: str = ACTION
    use_t5_embeddings: bool = True
    default_task: str = "Pick the green cube and place it inside the blue box."

    train_mode: str = "full_dit"
    train_last_n_dit_blocks: int = 8
    freeze_vae: bool = True
    freeze_text_encoder: bool = True
    freeze_tokenizer: bool = True
    action_loss_weight: float = 1.0
    visual_loss_weight: float = 0.0
    future_state_loss_weight: float = 0.0
    dtype: str = "bfloat16"

    max_delta_from_observation: float = 8.0
    max_gripper_delta_from_observation: float = 8.0
    max_step_delta: float = 4.0
    max_gripper_step_delta: float = 5.0
    safety_clip_actions: bool = True
    seed: int = 195

    optimizer_lr: float = 1e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-4
    optimizer_grad_clip_norm: float = 10.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.chunk_size != 50:
            raise ValueError("Current SO101 Cosmos checkpoint schema expects chunk_size=50.")
        if self.actions_per_chunk != self.chunk_size:
            self.actions_per_chunk = self.chunk_size
        if self.n_action_steps > self.actions_per_chunk:
            raise ValueError("n_action_steps must be <= actions_per_chunk.")
        if self.action_dim != 6 or self.state_dim != 6:
            raise ValueError("SO101 Cosmos policy currently expects 6D state/action.")
        if self.dtype not in {"bfloat16", "bf16", "float16", "fp16", "float32", "fp32"}:
            raise ValueError("dtype must be bfloat16/bf16, float16/fp16, or float32/fp32.")

    def validate_features(self) -> None:
        if not self.input_features:
            self.input_features = {
                self.state_key: PolicyFeature(type=FeatureType.STATE, shape=(self.state_dim,)),
                self.primary_camera_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
                self.wrist_camera_key: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            }
        if not self.output_features:
            self.output_features = {self.action_key: PolicyFeature(type=FeatureType.ACTION, shape=(self.action_dim,))}

        if self.action_feature is not None:
            self.action_dim = int(self.action_feature.shape[-1])
        state_feature = self.input_features.get(self.state_key) if self.input_features else None
        if state_feature is not None:
            self.state_dim = int(state_feature.shape[-1])

        image_keys = list(self.image_features.keys())
        if self.primary_camera_key not in image_keys:
            raise ValueError(f"Cosmos requires primary camera key {self.primary_camera_key!r}; images={image_keys}")
        if self.wrist_camera_key not in image_keys:
            raise ValueError(f"Cosmos requires wrist camera key {self.wrist_camera_key!r}; images={image_keys}")
        if self.state_key not in self.input_features:
            raise ValueError(f"Cosmos requires state key {self.state_key!r}.")
        if self.action_feature is None:
            raise ValueError("Cosmos requires LeRobot action output feature.")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return None

    @property
    def observation_delta_indices(self) -> list[int]:
        return [0, self.chunk_size]

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
