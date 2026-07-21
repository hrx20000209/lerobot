import numpy as np
import pytest
import torch
from torch import nn

from lerobot.policies.lingbo_va.configuration_lingbo_va import LingBoVAConfig
from lerobot.policies.lingbo_va.modeling_lingbo_va import LingBoVAPolicy
from lerobot.utils.constants import ACTION


def _stats(values: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
    return {
        ACTION: {
            "q01": values - 10,
            "q99": values + 10,
            "min": values - 20,
            "max": values + 20,
            "mean": values,
            "std": torch.ones_like(values),
        }
    }


def test_so101_defaults_match_recorded_camera_and_action_layout() -> None:
    config = LingBoVAConfig()
    config.validate_features()

    assert config.camera_keys == ["observation.images.front", "observation.images.right"]
    assert config.used_action_channel_ids == list(range(6))
    assert config.num_frames == 25


def test_action_channels_round_trip_through_lingbo_layout() -> None:
    centers = torch.arange(6, dtype=torch.float32) * 10
    config = LingBoVAConfig(device="cpu")
    policy = LingBoVAPolicy(config, dataset_stats=_stats(centers))
    object.__setattr__(policy, "_transformer", nn.Linear(1, 1, bias=False))

    action = centers.view(1, 1, 6).repeat(1, config.chunk_size, 1)
    model_action, mask = policy._build_model_actions({ACTION: action})

    assert model_action.shape == (1, 30, 4, 8, 1)
    assert torch.allclose(model_action[:, :6], torch.zeros_like(model_action[:, :6]), atol=1e-6)
    assert not mask[:, 6:].any()


def test_server_full_action_uses_configured_channel_ids() -> None:
    config = LingBoVAConfig(
        device="cpu",
        action_dim=6,
        used_action_channel_ids=[0, 1, 2, 3, 4, 28],
    )
    policy = LingBoVAPolicy(config)
    full_action = np.arange(30, dtype=np.float32)[None].repeat(config.chunk_size, axis=0)

    action = policy._server_action_to_tensor(full_action)

    assert action.shape == (1, config.chunk_size, 6)
    assert action[0, 0].tolist() == [0, 1, 2, 3, 4, 28]


def test_video_head_scope_unfreezes_video_io_modules() -> None:
    config = LingBoVAConfig(device="cpu", train_video_head=True)
    policy = LingBoVAPolicy(config)

    class Transformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_embedder = nn.Linear(2, 2)
            self.action_proj_out = nn.Linear(2, 2)
            self.condition_embedder_action = nn.Linear(2, 2)
            self.patch_embedding_mlp = nn.Linear(2, 2)
            self.condition_embedder = nn.Linear(2, 2)
            self.proj_out = nn.Linear(2, 2)
            self.block = nn.Linear(2, 2)
            self.scale_shift_table = nn.Parameter(torch.ones(1))

    transformer = Transformer()
    object.__setattr__(policy, "_transformer", transformer)
    policy._apply_training_freeze()

    assert all(param.requires_grad for param in transformer.proj_out.parameters())
    assert all(param.requires_grad for param in transformer.patch_embedding_mlp.parameters())
    assert all(param.requires_grad for param in transformer.condition_embedder.parameters())
    assert transformer.scale_shift_table.requires_grad
    assert not any(param.requires_grad for param in transformer.block.parameters())


def test_lora_scope_trains_adapters_and_saved_heads_only() -> None:
    from peft import LoraConfig, get_peft_model

    config = LingBoVAConfig(device="cpu", train_video_head=True, use_transformer_lora=True)
    policy = LingBoVAPolicy(config)

    class Transformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.to_q = nn.Linear(2, 2)
            self.action_embedder = nn.Linear(2, 2)
            self.action_proj_out = nn.Linear(2, 2)
            self.condition_embedder_action = nn.Linear(2, 2)
            self.patch_embedding_mlp = nn.Linear(2, 2)
            self.condition_embedder = nn.Linear(2, 2)
            self.proj_out = nn.Linear(2, 2)
            self.block = nn.Linear(2, 2)

    transformer = get_peft_model(
        Transformer(),
        LoraConfig(
            r=1,
            target_modules=["to_q"],
            modules_to_save=policy._lora_modules_to_save(),
        ),
    )
    object.__setattr__(policy, "_transformer", transformer)
    policy._apply_training_freeze()

    trainable = {name for name, param in transformer.named_parameters() if param.requires_grad}
    assert any("lora_" in name for name in trainable)
    assert any("action_embedder.modules_to_save" in name for name in trainable)
    assert any("proj_out.modules_to_save" in name for name in trainable)
    assert not any("block" in name for name in trainable)
    assert not any("original_module" in name for name in trainable)

    named_trainable = [(name, param) for name, param in transformer.named_parameters() if param.requires_grad]
    lora_params = [param for name, param in named_trainable if "lora_" in name]
    head_params = [param for name, param in named_trainable if "lora_" not in name]
    groups = [
        {"params": head_params},
        {"params": lora_params, "lr": config.lora_optimizer_lr},
    ]
    assert groups[0]["params"]
    assert groups[1]["params"]
    assert groups[1]["lr"] == 1e-4


def _fake_transformer_with_blocks(num_blocks: int = 4) -> nn.Module:
    class Transformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_embedder = nn.Linear(2, 2)
            self.action_proj_out = nn.Linear(2, 2)
            self.condition_embedder_action = nn.Linear(2, 2)
            self.blocks = nn.ModuleList([nn.Linear(2, 2) for _ in range(num_blocks)])

    return Transformer()


def test_train_last_n_blocks_unfreezes_only_the_last_n_blocks() -> None:
    config = LingBoVAConfig(device="cpu", train_action_head_only=True, train_last_n_blocks=2)
    policy = LingBoVAPolicy(config)
    transformer = _fake_transformer_with_blocks(num_blocks=4)
    object.__setattr__(policy, "_transformer", transformer)

    policy._apply_training_freeze()

    assert not any(p.requires_grad for p in transformer.blocks[0].parameters())
    assert not any(p.requires_grad for p in transformer.blocks[1].parameters())
    assert all(p.requires_grad for p in transformer.blocks[2].parameters())
    assert all(p.requires_grad for p in transformer.blocks[3].parameters())
    assert all(p.requires_grad for p in transformer.action_embedder.parameters())


def test_train_last_n_blocks_zero_leaves_all_blocks_frozen() -> None:
    config = LingBoVAConfig(device="cpu", train_action_head_only=True, train_last_n_blocks=0)
    policy = LingBoVAPolicy(config)
    transformer = _fake_transformer_with_blocks(num_blocks=4)
    object.__setattr__(policy, "_transformer", transformer)

    policy._apply_training_freeze()

    assert not any(p.requires_grad for block in transformer.blocks for p in block.parameters())


def test_trainable_param_self_check_raises_below_half_percent() -> None:
    # A transformer with a huge frozen "backbone" and only a tiny trainable head: trainable% < 0.5%.
    config = LingBoVAConfig(device="cpu", train_action_head_only=True, train_video_head=False)
    policy = LingBoVAPolicy(config)

    class Transformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action_embedder = nn.Linear(2, 2)  # tiny trainable head: 6 params
            self.action_proj_out = nn.Linear(2, 2)
            self.condition_embedder_action = nn.Linear(2, 2)
            self.backbone = nn.Linear(10_000, 10_000)  # huge frozen backbone: >1e8 params

    transformer = Transformer()
    object.__setattr__(policy, "_transformer", transformer)

    with pytest.raises(RuntimeError, match="below the 0.5%"):
        policy._apply_training_freeze()


def test_trainable_param_self_check_passes_for_full_finetune() -> None:
    config = LingBoVAConfig(device="cpu", train_action_head_only=False, use_transformer_lora=False)
    policy = LingBoVAPolicy(config)
    transformer = _fake_transformer_with_blocks(num_blocks=2)
    object.__setattr__(policy, "_transformer", transformer)

    policy._apply_training_freeze()  # should not raise

    assert all(p.requires_grad for p in transformer.parameters())


def test_config_rejects_invalid_new_fields() -> None:
    with pytest.raises(ValueError, match="cfg_prob"):
        LingBoVAConfig(device="cpu", cfg_prob=1.5)
    with pytest.raises(ValueError, match="train_last_n_blocks"):
        LingBoVAConfig(device="cpu", train_last_n_blocks=-1)
    with pytest.raises(ValueError, match="video_loss_weight"):
        LingBoVAConfig(device="cpu", video_loss_weight=-0.1)


def test_commit_executed_action_validates_shape_and_alignment() -> None:
    config = LingBoVAConfig(device="cpu", action_per_frame=8, action_dim=6)
    config.validate_features()
    policy = LingBoVAPolicy(config)

    wrong_action_dim = torch.zeros(8, 5)
    with pytest.raises(ValueError, match="action_dim"):
        policy.commit_executed_action(wrong_action_dim, obs={})

    not_multiple_of_action_per_frame = torch.zeros(5, 6)
    with pytest.raises(ValueError, match="multiple of action_per_frame"):
        policy.commit_executed_action(not_multiple_of_action_per_frame, obs={})


def test_forward_rejects_non_flex_train_attn_mode() -> None:
    config = LingBoVAConfig(device="cpu", train_attn_mode="torch")
    policy = LingBoVAPolicy(config)
    with pytest.raises(ValueError, match="train_attn_mode='flex'"):
        policy.forward({})


def test_scheduler_preset_is_cosine_with_configured_warmup() -> None:
    config = LingBoVAConfig(device="cpu", scheduler_name="cosine", scheduler_warmup_steps=123)
    preset = config.get_scheduler_preset()
    assert preset.name == "cosine"
    assert preset.num_warmup_steps == 123
