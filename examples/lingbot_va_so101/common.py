"""Shared config/helpers for the LeRobot-native lingbot_va SO-101 fine-tune.

See ~/Projects/lingbot_va_lerobot_native/NOTES.md for the Step 0 findings this is
built on: joint-space (not EEF) action channels, width_concat camera layout, and the
manual quantile-normalization gap (the shipped preprocessor uses IDENTITY for ACTION;
LingBot-VA's flow-matching loss expects the training target's scale to already be
close to the noise's, i.e. roughly [-1, 1]).
"""

from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig

DATASET_ROOT = "/data/rxhuang/three_cubes_1"
WAN_PRETRAINED_PATH = "/home/rxhuang/Projects/models/lingbot-va-base"
# Base (non-EEF-specialized) LeRobot-format checkpoint -- per user's confirmed choice of
# joint-channel fine-tuning over FK-to-EEF, we start from this rather than
# lingbot_va_libero_long/robotwin (which are EEF-pose specialized on channels we aren't using).
BASE_CHECKPOINT = "lerobot/lingbot_va_base"
RUNS_ROOT = Path("/data/rxhuang/lingbot_va_runs")

OBS_CAM_KEYS = ["observation.images.front", "observation.images.right", "observation.images.wrist"]
# 5 arm joints + gripper, in the dataset's own column order (shoulder_pan, shoulder_lift,
# elbow_flex, wrist_flex, wrist_roll, gripper) -> channels 14-18 ("left-arm joints,
# unused by released checkpoints") + 28 ("left gripper"). Joint-space fine-tune of the
# base (non-EEF-specialized) checkpoint, per user's explicit choice over FK-to-EEF.
USED_ACTION_CHANNEL_IDS = [14, 15, 16, 17, 18, 28]
ACTION_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Held-out episodes for periodic open-loop eval (same 5-episode convention used by the
# earlier wan_va/RoboTwin-side SO-101 post-train config for this same dataset).
HELD_OUT_EPISODES = [95, 96, 97, 98, 99]
TRAIN_EPISODES = list(range(95))

FPS = 30


def build_config(pretrained: bool = True, **overrides) -> LingBotVAConfig:
    kwargs = dict(
        obs_cam_keys=OBS_CAM_KEYS,
        camera_layout="width_concat",
        used_action_channel_ids=USED_ACTION_CHANNEL_IDS,
        attn_mode="flex",
        wan_pretrained_path=WAN_PRETRAINED_PATH,
        text_encoder_device="cpu",
        device="cuda",
    )
    if pretrained:
        kwargs["pretrained_path"] = BASE_CHECKPOINT
    kwargs.update(overrides)
    return LingBotVAConfig(**kwargs)


def delta_timestamps_for(config: LingBotVAConfig) -> dict:
    dt = {k: [i / FPS for i in config.observation_delta_indices] for k in OBS_CAM_KEYS}
    dt["action"] = [i / FPS for i in config.action_delta_indices]
    return dt


def make_dataset(config: LingBotVAConfig, episodes: list[int] | None) -> LeRobotDataset:
    return LeRobotDataset(
        "three_cubes_1",
        root=DATASET_ROOT,
        delta_timestamps=delta_timestamps_for(config),
        episodes=episodes,
    )


def make_ds_meta() -> LeRobotDatasetMetadata:
    return LeRobotDatasetMetadata("three_cubes_1", root=DATASET_ROOT)


def action_quantiles(ds_meta: LeRobotDatasetMetadata) -> tuple[torch.Tensor, torch.Tensor]:
    q01 = torch.tensor(ds_meta.stats["action"]["q01"], dtype=torch.float32)
    q99 = torch.tensor(ds_meta.stats["action"]["q99"], dtype=torch.float32)
    return q01, q99


def normalize_action(action: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    """Map raw physical-unit actions into [-1, 1] via quantiles (see module docstring)."""
    q01 = q01.to(action.device, action.dtype)
    q99 = q99.to(action.device, action.dtype)
    denom = (q99 - q01).clamp_min(1e-6)
    return 2 * (action - q01) / denom - 1


def denormalize_action(action: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    q01 = q01.to(action.device, action.dtype)
    q99 = q99.to(action.device, action.dtype)
    denom = q99 - q01
    return (action + 1) / 2 * denom + q01


def attach_text_embed_cache(policy):
    """Memoize policy._get_t5_prompt_embeds by (prompt tuple, max_sequence_length).

    UMT5-XXL runs on CPU (config.text_encoder_device) and takes ~150s per call for a
    512-token prompt (measured), vs ~4s for the rest of a training forward pass combined
    (VAE encode + transformer fwd/bwd) -- by far the dominant cost. three_cubes_1 has a
    single constant task string, so this turns an O(150s/step) bottleneck into a one-time
    cost. Safe for multi-task datasets too: caches per distinct prompt tuple, not globally.
    Instance-level monkeypatch on our own policy object, not a lerobot source-code change.
    """
    orig = policy._get_t5_prompt_embeds
    cache = {}

    def cached(prompt, max_sequence_length):
        prompt_list = [prompt] if isinstance(prompt, str) else list(prompt)
        key = (tuple(prompt_list), max_sequence_length)
        if key not in cache:
            cache[key] = orig(prompt_list, max_sequence_length).detach().clone()
        cached_emb = cache[key]
        if cached_emb.shape[0] != len(prompt_list):
            # Shouldn't happen (key includes the prompt tuple), but guard shape mismatches.
            cache.pop(key, None)
            return orig(prompt_list, max_sequence_length)
        return cached_emb

    policy._get_t5_prompt_embeds = cached
    return policy
