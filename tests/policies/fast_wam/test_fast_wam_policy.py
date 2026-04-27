#!/usr/bin/env python

from pathlib import Path

import pytest
import torch

from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.fast_wam import FastWAMConfig, FastWAMPolicy
from lerobot.policies.factory import get_policy_class, make_policy_config, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE


def test_fast_wam_factory_registration():
    policy_cls = get_policy_class("fast_wam")
    config = make_policy_config("fast_wam", device="cpu", push_to_hub=False)

    assert policy_cls is FastWAMPolicy
    assert isinstance(config, FastWAMConfig)
    assert config.type == "fast_wam"


def test_fast_wam_policy_is_lazy_without_runtime_deps(monkeypatch):
    config = FastWAMConfig(device="cpu", push_to_hub=False)
    policy = FastWAMPolicy(config)

    assert policy._fastwam_model is None

    def fail_fastwam_resolution():
        raise FileNotFoundError("blocked in unit test")

    monkeypatch.setattr(policy, "_resolve_fastwam_root", fail_fastwam_resolution)
    with pytest.raises(FileNotFoundError):
        policy.predict_action_chunk({})


def test_fast_wam_pre_post_processors_are_identity():
    config = FastWAMConfig(device="cpu", push_to_hub=False)
    preprocessor, postprocessor = make_pre_post_processors(config)

    batch = {
        config.state_key: torch.zeros(config.proprio_dim),
        "task": "pick up the object",
    }
    action = torch.zeros(config.action_dim)

    processed = preprocessor(batch)
    assert processed["task"] == batch["task"]
    assert torch.equal(processed[config.state_key], batch[config.state_key])
    assert torch.equal(postprocessor(action), action)


def test_fast_wam_native_checkpoint_uses_identity_processors(tmp_path):
    checkpoint = tmp_path / "fastwam.pt"
    checkpoint.touch()
    config = FastWAMConfig(device="cpu", push_to_hub=False)

    preprocessor, postprocessor = make_pre_post_processors(config, pretrained_path=checkpoint)

    action = torch.zeros(config.action_dim)
    assert torch.equal(postprocessor(action), action)
    assert preprocessor({"task": "pick"})["task"] == "pick"


def test_fast_wam_from_pretrained_accepts_native_checkpoint(monkeypatch, tmp_path):
    checkpoint = tmp_path / "fastwam.pt"
    checkpoint.touch()
    calls = []

    def fake_init(self, config, **kwargs):
        calls.append((config.fastwam_checkpoint_path, kwargs))

    monkeypatch.setattr(FastWAMPolicy, "__init__", fake_init)
    monkeypatch.setattr(FastWAMPolicy, "eval", lambda self: self)

    policy = FastWAMPolicy.from_pretrained(checkpoint, config=FastWAMConfig(device="cpu", push_to_hub=False))

    assert isinstance(policy, FastWAMPolicy)
    assert calls == [(Path(checkpoint), {})]


def test_fast_wam_forward_builds_training_sample(monkeypatch):
    class FakeFastWAM:
        device = torch.device("cpu")
        torch_dtype = torch.float32

        def train(self):
            return self

        def requires_grad_(self, value):
            return self

        def encode_prompt(self, prompts):
            return torch.zeros(len(prompts), 4, 4096), torch.ones(len(prompts), 4, dtype=torch.bool)

        def training_loss(self, sample, tiled=False):
            assert not tiled
            assert sample["video"].shape == (1, 3, 9, 32, 32)
            assert sample["action"].shape == (1, 32, 6)
            assert sample["proprio"].shape == (1, 32, 6)
            assert sample["context"].shape == (1, 4, 4096)
            return sample["action"].mean(), {"loss_action": 0.0, "loss_video": 0.0}

    stats = {
        ACTION: {
            "mean": torch.zeros(6),
            "std": torch.ones(6),
            "min": -torch.ones(6),
            "max": torch.ones(6),
            "q01": -torch.ones(6),
            "q99": torch.ones(6),
        },
        OBS_STATE: {
            "mean": torch.zeros(6),
            "std": torch.ones(6),
            "min": -torch.ones(6),
            "max": torch.ones(6),
            "q01": -torch.ones(6),
            "q99": torch.ones(6),
        },
    }
    config = FastWAMConfig(
        device="cpu",
        push_to_hub=False,
        action_dim=6,
        proprio_dim=6,
        image_size=(32, 32),
        train_expert_only=False,
        freeze_vision_encoder=False,
        camera_keys=["observation.images.front", "observation.images.wrist"],
        input_features={
            "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 32, 32)),
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,))},
    )
    policy = FastWAMPolicy(config, dataset_stats=stats)

    def fake_ensure_runtime():
        object.__setattr__(policy, "_fastwam_model", FakeFastWAM())

    monkeypatch.setattr(policy, "_ensure_fastwam_runtime", fake_ensure_runtime)
    batch = {
        "observation.images.front": torch.randint(0, 255, (1, 33, 3, 32, 32), dtype=torch.uint8),
        "observation.images.wrist": torch.randint(0, 255, (1, 33, 3, 32, 32), dtype=torch.uint8),
        OBS_STATE: torch.zeros(1, 33, 6),
        ACTION: torch.zeros(1, 32, 6),
        ACTION + "_is_pad": torch.zeros(1, 32, dtype=torch.bool),
        "observation.images.front_is_pad": torch.zeros(1, 33, dtype=torch.bool),
        "observation.images.wrist_is_pad": torch.zeros(1, 33, dtype=torch.bool),
        "task": ["Grab the blue cube"],
    }

    loss, loss_dict = policy.forward(batch)

    assert torch.isclose(loss, torch.tensor(0.0))
    assert loss_dict == {"loss_action": 0.0, "loss_video": 0.0}
