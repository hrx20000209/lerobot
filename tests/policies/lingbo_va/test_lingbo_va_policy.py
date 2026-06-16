import numpy as np
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
