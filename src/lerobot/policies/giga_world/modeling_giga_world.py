#!/usr/bin/env python

from __future__ import annotations

import json
import logging
import os
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from PIL import Image
from torch import Tensor
from torchvision.transforms import InterpolationMode, functional as tvf

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION

from .configuration_giga_world import GigaWorldConfig


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _as_path(value: str | Path | None) -> Path | None:
    if _is_none_like(value):
        return None
    return Path(value).expanduser()


def _torch_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if name in {"float16", "fp16"}:
        return torch.float16
    if name in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {name}")


class GigaWorldPolicy(PreTrainedPolicy):
    config_class = GigaWorldConfig
    name = "giga_world"

    def __init__(self, config: GigaWorldConfig, **kwargs: Any) -> None:
        super().__init__(config)
        dataset_stats = kwargs.pop("dataset_stats", None)
        del kwargs

        config.validate_features()
        self.config = config
        self._dataset_stats = dataset_stats
        self._action_queue: deque[Tensor] = deque(maxlen=config.n_action_steps)
        self._prompt_cache: dict[str, Tensor] = {}

        self.vae: nn.Module | None = None
        self.transformer: nn.Module | None = None
        self.text_encoder: nn.Module | None = None
        self.tokenizer: Any | None = None
        self.pipeline: Any | None = None

        self._latents_mean: Tensor | None = None
        self._latents_std: Tensor | None = None
        self._stats = self._load_normalization_stats()
        self._runtime_loaded = False
        self._lora_applied = False

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: GigaWorldConfig | None = None,
        **kwargs: Any,
    ) -> GigaWorldPolicy:
        path = Path(pretrained_name_or_path).expanduser()
        if path.is_dir():
            if config is None:
                loaded = PreTrainedConfig.from_pretrained(path, **kwargs)
                if not isinstance(loaded, GigaWorldConfig):
                    raise TypeError(f"Expected GigaWorldConfig in {path}, got {type(loaded)}.")
                config = loaded

            transformer_dir = path / "transformer"
            if transformer_dir.exists():
                config.transformer_path = Path("transformer")
                config.reinit_action_heads = False
            norm_stats_path = path / "giga_world_norm_stats.json"
            if norm_stats_path.exists():
                config.norm_stats_path = Path("giga_world_norm_stats.json")

            config.pretrained_path = path
            policy = cls(config, **kwargs)
            policy.eval()
            return policy

        return super().from_pretrained(pretrained_name_or_path, config=config, **kwargs)

    def _save_pretrained(self, save_directory: Path) -> None:
        old_transformer_path = self.config.transformer_path
        old_norm_stats_path = self.config.norm_stats_path
        try:
            if self.transformer is not None:
                self.config.transformer_path = Path("transformer")
                self.config.reinit_action_heads = False
            if self._stats is not None:
                self.config.norm_stats_path = Path("giga_world_norm_stats.json")
            self.config._save_pretrained(save_directory)
        finally:
            self.config.transformer_path = old_transformer_path
            self.config.norm_stats_path = old_norm_stats_path

        if self.transformer is not None:
            transformer_dir = save_directory / "transformer"
            transformer_dir.mkdir(parents=True, exist_ok=True)
            self.transformer.save_pretrained(transformer_dir)
        if self._stats is not None:
            with open(save_directory / "giga_world_norm_stats.json", "w", encoding="utf-8") as f:
                json.dump(self._stats, f, indent=2)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.vae is not None:
            self.vae.eval()
        if self.text_encoder is not None:
            self.text_encoder.eval()
        if self.transformer is not None:
            self._apply_training_freeze()
        return self

    def get_optim_params(self) -> list:
        self._ensure_runtime(for_training=True)
        self._apply_training_freeze()
        return [p for p in self.transformer.parameters() if p.requires_grad]

    def reset(self) -> None:
        self._action_queue.clear()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        self._ensure_runtime(for_training=True)
        self._apply_training_freeze()
        prepared = self._prepare_training_batch(batch)
        losses = self._training_losses(prepared)
        action_loss = losses["action_loss"]
        visual_loss = losses["visual_loss"]
        loss = self.config.action_loss_weight * action_loss + self.config.visual_loss_weight * visual_loss
        return loss, {
            "loss_action": float(action_loss.detach().item()),
            "loss_visual": float(visual_loss.detach().item()),
        }

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any], **kwargs: Any) -> Tensor:
        del kwargs
        self.eval()
        if not self._action_queue:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any], **kwargs: Any) -> Tensor:
        del kwargs
        self.eval()
        self._ensure_runtime(for_training=False)
        self._ensure_pipeline()

        image_batches = self._build_visual_condition(batch)
        raw_state = self._current_state(batch).to(device=self.device, dtype=torch.float32)
        norm_state = self._normalize_state(raw_state)
        prompt_embeds = self._prompt_embeds(batch, batch_size=raw_state.shape[0]).to(
            device=self.device, dtype=torch.float32
        )

        chunks: list[Tensor] = []
        for idx in range(raw_state.shape[0]):
            image = self._composite_pil_from_condition(image_batches[idx])
            _, action = self.pipeline(
                height=self._image_height,
                width=self._image_width,
                action_chunk=self.config.chunk_size,
                state=norm_state[idx].unsqueeze(0),
                num_frames=self.config.num_inference_frames,
                guidance_scale=self.config.guidance_scale,
                num_inference_steps=self.config.num_inference_steps,
                image=image,
                action_only=True,
                return_dict=False,
                prompt_embeds=prompt_embeds[idx].unsqueeze(0),
                max_sequence_length=self.config.t5_len,
            )
            action = action[0].float()
            action = self._denormalize_action(action)
            action = self._add_state_to_delta_action(action, raw_state[idx])
            chunks.append(action.cpu())
        return torch.stack(chunks, dim=0)

    @property
    def device(self) -> torch.device:
        return torch.device(self.config.device)

    @property
    def _dtype(self) -> torch.dtype:
        return _torch_dtype(self.config.torch_dtype)

    @property
    def _view_keys(self) -> list[str]:
        if self.config.view_keys:
            return list(self.config.view_keys)
        return list(self.config.image_features.keys())

    @property
    def _num_visual_frames(self) -> int:
        return len(self.config.observation_delta_indices)

    @property
    def _image_width(self) -> int:
        return int(self.config.per_view_size[0]) * len(self._view_keys)

    @property
    def _image_height(self) -> int:
        return int(self.config.per_view_size[1])

    def _resolve_giga_world_root(self) -> Path:
        candidates = [
            self.config.giga_world_root,
            os.environ.get("GIGA_WORLD_POLICY_ROOT"),
            Path("~/Projects/giga-world-policy"),
        ]
        for candidate in candidates:
            if _is_none_like(candidate):
                continue
            path = Path(candidate).expanduser().resolve()
            if (path / "world_action_model").exists():
                return path
        raise FileNotFoundError(
            "Could not find giga-world-policy checkout. Set policy.giga_world_root or GIGA_WORLD_POLICY_ROOT."
        )

    def _install_giga_import_path(self) -> Path:
        root = self._resolve_giga_world_root()
        for path in (root,):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
        return root

    def _model_cache_dir(self) -> Path:
        env_cache_dir = os.environ.get("GIGA_WORLD_MODEL_CACHE_DIR")
        path = Path(env_cache_dir if env_cache_dir else self.config.model_cache_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve_relative_to_checkpoint(self, path: Path | None) -> Path | None:
        if path is None:
            return None
        path = Path(path).expanduser()
        if not path.is_absolute() and self.config.pretrained_path is not None:
            path = Path(self.config.pretrained_path).expanduser() / path
        return path.resolve()

    def _ensure_runtime(self, for_training: bool) -> None:
        if self._runtime_loaded:
            return

        self._install_giga_import_path()
        try:
            from diffusers.models import AutoencoderKLWan
            from world_action_model.models.transformer_wa_casual import (
                CasualWorldActionTransformer,
                WanRotaryPosEmbed1D,
            )
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "GigaWorldPolicy needs the local giga-world-policy checkout and WAN-capable diffusers. "
                "Set policy.giga_world_root and run in the lerobot environment with diffusers>=0.36."
            ) from exc

        dtype = self._dtype
        cache_dir = self._model_cache_dir()
        device = self.device

        self.vae = AutoencoderKLWan.from_pretrained(
            self.config.wan_model_id,
            subfolder="vae",
            torch_dtype=dtype,
            cache_dir=cache_dir,
        )
        self.vae.requires_grad_(False)
        self.vae.to(device=device, dtype=dtype)

        transformer_path = self._resolve_relative_to_checkpoint(_as_path(self.config.transformer_path))
        if transformer_path is not None and transformer_path.exists():
            if (transformer_path / "adapter_config.json").exists():
                base = self._load_base_transformer(CasualWorldActionTransformer, dtype, cache_dir)
                self._install_action_heads(base, WanRotaryPosEmbed1D)
                from peft import PeftModel

                transformer = PeftModel.from_pretrained(base, transformer_path, is_trainable=for_training)
                self._lora_applied = True
            else:
                transformer = CasualWorldActionTransformer.from_pretrained(
                    transformer_path,
                    torch_dtype=dtype,
                )
        else:
            transformer = self._load_base_transformer(CasualWorldActionTransformer, dtype, cache_dir)

        if self.config.reinit_action_heads:
            self._install_action_heads(transformer, WanRotaryPosEmbed1D)

        checkpoint_path = self._resolve_relative_to_checkpoint(_as_path(self.config.transformer_checkpoint_path))
        if checkpoint_path is not None:
            self._load_transformer_checkpoint(transformer, checkpoint_path)

        if self.config.gradient_checkpointing and hasattr(transformer, "enable_gradient_checkpointing"):
            transformer.enable_gradient_checkpointing()

        transformer.to(device=device, dtype=dtype)
        self.transformer = transformer
        self._setup_latent_stats()
        self._apply_transformer_lora(for_training=for_training)
        self._apply_training_freeze()
        self._runtime_loaded = True

    def _install_action_heads(self, transformer: nn.Module, rope_cls: Any) -> None:
        transformer.action_encoder = nn.Sequential(
            nn.Linear(self.config.action_dim, 128),
            nn.GELU(),
            nn.Linear(128, 256),
            nn.GELU(),
            nn.Linear(256, 3072),
        )
        transformer.action_decoder = nn.Sequential(
            nn.Linear(3072, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, self.config.action_dim),
        )
        transformer.action_rope = rope_cls(128, 1024)

    def _load_base_transformer(self, cls: Any, dtype: torch.dtype, cache_dir: Path) -> nn.Module:
        base_path = self._resolve_relative_to_checkpoint(_as_path(self.config.base_transformer_path))
        if base_path is not None and base_path.exists():
            return cls.from_pretrained(base_path, torch_dtype=dtype)
        return cls.from_pretrained(
            self.config.wan_model_id,
            subfolder="transformer",
            torch_dtype=dtype,
            cache_dir=cache_dir,
        )

    def _load_transformer_checkpoint(self, transformer: nn.Module, checkpoint_path: Path) -> None:
        if checkpoint_path.is_dir():
            loaded = type(transformer).from_pretrained(checkpoint_path, torch_dtype=self._dtype)
            transformer.load_state_dict(loaded.state_dict(), strict=False)
            return
        state = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state, dict):
            for key in ("state_dict", "model", "module"):
                if key in state and isinstance(state[key], dict):
                    state = state[key]
                    break
        missing, unexpected = transformer.load_state_dict(state, strict=False)
        logging.info(
            "Loaded GigaWorld transformer checkpoint %s | missing=%d unexpected=%d",
            checkpoint_path,
            len(missing),
            len(unexpected),
        )

    def _apply_transformer_lora(self, for_training: bool) -> None:
        if self.transformer is None or self._lora_applied:
            return
        lora_path = self._resolve_relative_to_checkpoint(_as_path(self.config.transformer_lora_path))
        if lora_path is not None:
            from peft import PeftModel

            self.transformer = PeftModel.from_pretrained(self.transformer, lora_path, is_trainable=for_training)
            self._lora_applied = True
            return
        if not self.config.use_transformer_lora:
            return

        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            target_modules=list(self.config.lora_target_modules),
            modules_to_save=["action_encoder", "action_decoder"],
        )
        self.transformer = get_peft_model(self.transformer, lora_config)
        self._lora_applied = True

    def _apply_training_freeze(self) -> None:
        if self.transformer is None:
            return
        if not self.training:
            self.transformer.eval()
            return
        self.transformer.train()
        if self.config.use_transformer_lora or self.config.transformer_lora_path is not None:
            for name, param in self.transformer.named_parameters():
                trainable = "lora_" in name or "modules_to_save" in name
                param.requires_grad_(trainable)
            return
        if self.config.freeze_transformer_backbone:
            for param in self.transformer.parameters():
                param.requires_grad_(False)
        if self.config.train_action_heads:
            for module_name in ("action_encoder", "action_decoder", "action_rope"):
                module = getattr(self.transformer, module_name, None)
                if module is not None:
                    for param in module.parameters():
                        param.requires_grad_(True)

    def _setup_latent_stats(self) -> None:
        if self.vae is None:
            return
        dtype = self._dtype
        device = self.device
        self._latents_mean = torch.tensor(self.vae.config.latents_mean).view(
            1, self.vae.config.z_dim, 1, 1, 1
        ).to(device=device, dtype=dtype)
        self._latents_std = (
            1.0
            / torch.tensor(self.vae.config.latents_std)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(device=device, dtype=dtype)
        )

    def _ensure_text_encoder(self) -> None:
        if self.text_encoder is not None and self.tokenizer is not None:
            return
        from transformers import AutoTokenizer, UMT5EncoderModel

        cache_dir = self._model_cache_dir()
        text_device = torch.device(self.config.text_encoder_device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.wan_model_id,
            subfolder="tokenizer",
            cache_dir=cache_dir,
        )
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            self.config.wan_model_id,
            subfolder="text_encoder",
            torch_dtype=torch.float32,
            cache_dir=cache_dir,
        ).to(text_device)
        self.text_encoder.eval()
        for param in self.text_encoder.parameters():
            param.requires_grad_(False)

    def _ensure_pipeline(self) -> None:
        if self.pipeline is not None:
            return
        self._install_giga_import_path()
        from world_action_model.pipeline.wa_pipeline import WAPipeline

        self._ensure_text_encoder()
        self.pipeline = WAPipeline.from_pretrained(
            self.config.wan_model_id,
            vae=self.vae,
            transformer=self.transformer,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            torch_dtype=self._dtype,
            cache_dir=self._model_cache_dir(),
        )
        self.pipeline.to(self.device)

    def _load_normalization_stats(self) -> dict[str, Any] | None:
        path = self._resolve_relative_to_checkpoint(_as_path(self.config.norm_stats_path))
        if path is None or not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def _stat_tensor(self, key: str, stat: str, dim: int, default: float) -> Tensor:
        source = None
        if self._stats is not None:
            source = self._stats.get("norm_stats", {}).get(key, {}).get(stat)
        if source is None and self._dataset_stats is not None:
            source = self._dataset_stats.get(key, {}).get(stat)
        if source is None:
            out = torch.full((dim,), default, dtype=torch.float32)
        else:
            out = torch.as_tensor(source, dtype=torch.float32).flatten()
            out = F.pad(out, (0, dim - out.numel()), value=default) if out.numel() < dim else out[:dim]
        return out.to(self.device)

    def _delta_mask(self) -> Tensor:
        if self.config.delta_mask is not None:
            mask = torch.tensor(self.config.delta_mask, dtype=torch.bool)
        else:
            values = [True] * int(self.config.action_dim)
            if values:
                values[-1] = False
            mask = torch.tensor(values, dtype=torch.bool)
        if mask.numel() < self.config.action_dim:
            mask = F.pad(mask, (0, self.config.action_dim - mask.numel()), value=False)
        return mask[: self.config.action_dim].to(self.device)

    def _current_state(self, batch: dict[str, Any]) -> Tensor:
        state = torch.as_tensor(batch[self.config.state_key])
        if state.ndim == 3:
            state = state[:, 0]
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state = state[..., : self.config.state_dim]
        if state.shape[-1] < self.config.state_dim:
            state = F.pad(state, (0, self.config.state_dim - state.shape[-1]))
        return state

    def _action_chunk(self, batch: dict[str, Any]) -> Tensor:
        action = torch.as_tensor(batch.get(self.config.action_key, batch[ACTION]))
        if action.ndim == 2:
            action = action[:, None, :]
        action = action[:, : self.config.chunk_size, : self.config.action_dim]
        if action.shape[1] < self.config.chunk_size:
            pad_t = self.config.chunk_size - action.shape[1]
            action = F.pad(action, (0, 0, 0, pad_t))
        if action.shape[-1] < self.config.action_dim:
            action = F.pad(action, (0, self.config.action_dim - action.shape[-1]))
        return action

    def _normalize_state(self, state: Tensor) -> Tensor:
        mean = self._stat_tensor(self.config.state_key, "mean", self.config.state_dim, 0.0)
        std = self._stat_tensor(self.config.state_key, "std", self.config.state_dim, 1.0).clamp_min(1e-8)
        return (state.to(self.device, dtype=torch.float32) - mean) / std

    def _normalize_action_delta(self, action: Tensor, state: Tensor) -> Tensor:
        action = action.to(self.device, dtype=torch.float32)
        state = state.to(self.device, dtype=torch.float32)
        mask = self._delta_mask()
        if state.shape[-1] >= action.shape[-1]:
            state_rep = state[:, None, : action.shape[-1]].expand_as(action)
            action = torch.where(mask[None, None, :], action - state_rep, action)
        mean = self._stat_tensor(self.config.action_key, "mean", self.config.action_dim, 0.0)
        std = self._stat_tensor(self.config.action_key, "std", self.config.action_dim, 1.0).clamp_min(1e-8)
        return (action - mean) / std

    def _denormalize_action(self, action: Tensor) -> Tensor:
        mean = self._stat_tensor(self.config.action_key, "mean", self.config.action_dim, 0.0)
        std = self._stat_tensor(self.config.action_key, "std", self.config.action_dim, 1.0).clamp_min(1e-8)
        return action.to(self.device, dtype=torch.float32) * std + mean

    def _add_state_to_delta_action(self, action: Tensor, state: Tensor) -> Tensor:
        mask = self._delta_mask()
        state = state.to(self.device, dtype=torch.float32)
        if state.shape[-1] < action.shape[-1]:
            return action
        state_rep = state[None, : action.shape[-1]].expand_as(action)
        return torch.where(mask[None, :], action + state_rep, action)

    def _image_sequence(self, batch: dict[str, Any], key: str) -> Tensor:
        x = torch.as_tensor(batch[key])
        if x.ndim == 3:
            x = x.unsqueeze(0).unsqueeze(1)
        elif x.ndim == 4:
            x = x.unsqueeze(1)
        elif x.ndim != 5:
            raise ValueError(f"Unexpected image tensor shape for {key}: {tuple(x.shape)}")
        if x.shape[2] not in (1, 3) and x.shape[-1] in (1, 3):
            x = x.permute(0, 1, 4, 2, 3).contiguous()
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        else:
            x = x.float()
            if float(x.detach().amax().cpu()) > 2.0:
                x = x / 255.0
        if x.shape[1] == 1 and self._num_visual_frames > 1:
            x = x.repeat(1, self._num_visual_frames, 1, 1, 1)
        elif x.shape[1] != self._num_visual_frames:
            idx = torch.linspace(0, x.shape[1] - 1, steps=self._num_visual_frames, device=x.device).long()
            x = x.index_select(1, idx)
        return x

    def _resize_crop_normalize(self, seq: Tensor) -> Tensor:
        b, t, c, h, w = seq.shape
        dst_w, dst_h = map(int, self.config.per_view_size)
        flat = seq.reshape(b * t, c, h, w)
        if float(dst_h) / h < float(dst_w) / w:
            new_h = int(round(float(dst_w) / w * h))
            new_w = dst_w
        else:
            new_h = dst_h
            new_w = int(round(float(dst_h) / h * w))
        flat = tvf.resize(flat, [new_h, new_w], interpolation=InterpolationMode.BILINEAR)
        max_x = max(0, new_w - dst_w)
        max_y = max(0, new_h - dst_h)
        if self.training and self.config.crop_mode == "random":
            x1 = random.randint(0, max_x) if max_x else 0
            y1 = random.randint(0, max_y) if max_y else 0
        else:
            x1 = max_x // 2
            y1 = max_y // 2
        flat = tvf.crop(flat, y1, x1, dst_h, dst_w)
        flat = flat.clamp(0.0, 1.0)
        flat = (flat - 0.5) / 0.5
        return flat.reshape(b, t, c, dst_h, dst_w)

    def _build_visual_condition(self, batch: dict[str, Any]) -> Tensor:
        views = []
        for key in self._view_keys:
            if key not in batch:
                raise KeyError(f"Missing GigaWorld image key: {key}")
            views.append(self._resize_crop_normalize(self._image_sequence(batch, key)))
        return torch.cat(views, dim=-1).to(self.device, dtype=torch.float32)

    def _prompt_embeds(self, batch: dict[str, Any], batch_size: int) -> Tensor:
        if "t5_embedding" in batch:
            embeds = torch.as_tensor(batch["t5_embedding"], dtype=torch.float32)
            if embeds.ndim == 2:
                embeds = embeds.unsqueeze(0)
            return self._pad_prompt_embeds(embeds, batch_size=batch_size)
        prompts = self._prompts_from_batch(batch, batch_size)
        return self._encode_prompts(prompts)

    def _prompts_from_batch(self, batch: dict[str, Any], batch_size: int) -> list[str]:
        value = batch.get(self.config.task_key, self.config.default_task)
        if isinstance(value, str):
            return [value] * batch_size
        if isinstance(value, list | tuple):
            prompts = [str(v) for v in value]
        elif isinstance(value, np.ndarray):
            prompts = [str(v) for v in value.tolist()]
        else:
            prompts = [str(value)] * batch_size
        if len(prompts) < batch_size:
            prompts.extend([self.config.default_task] * (batch_size - len(prompts)))
        return prompts[:batch_size]

    def _pad_prompt_embeds(self, embeds: Tensor, batch_size: int) -> Tensor:
        if embeds.shape[0] == 1 and batch_size > 1:
            embeds = embeds.repeat(batch_size, 1, 1)
        embeds = embeds[:batch_size]
        if embeds.shape[1] >= self.config.t5_len:
            return embeds[:, : self.config.t5_len]
        return F.pad(embeds, (0, 0, 0, self.config.t5_len - embeds.shape[1]))

    @torch.no_grad()
    def _encode_prompts(self, prompts: list[str]) -> Tensor:
        self._ensure_text_encoder()
        missing = [prompt for prompt in prompts if prompt not in self._prompt_cache]
        if missing:
            inputs = self.tokenizer(
                missing,
                padding="max_length",
                max_length=self.config.text_encoder_max_length,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            text_device = next(self.text_encoder.parameters()).device
            input_ids = inputs.input_ids.to(text_device)
            attention_mask = inputs.attention_mask.to(text_device)
            hidden = self.text_encoder(input_ids, attention_mask).last_hidden_state.float()
            seq_lens = attention_mask.gt(0).sum(dim=1).long()
            for prompt, emb, seq_len in zip(missing, hidden, seq_lens, strict=False):
                trimmed = emb[: int(seq_len.item())].detach().cpu()
                self._prompt_cache[prompt] = trimmed
        stacked = [self._prompt_cache[prompt] for prompt in prompts]
        return self._pad_prompt_embeds(torch.stack([self._pad_one_prompt(x) for x in stacked], dim=0), len(prompts))

    def _pad_one_prompt(self, embed: Tensor) -> Tensor:
        if embed.shape[0] >= self.config.t5_len:
            return embed[: self.config.t5_len]
        return F.pad(embed, (0, 0, 0, self.config.t5_len - embed.shape[0]))

    def _prepare_training_batch(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        images = self._build_visual_condition(batch).to(self.device)
        ref_images = torch.zeros_like(images)
        ref_images[:, :1] = images[:, :1]
        state_raw = self._current_state(batch)
        action_raw = self._action_chunk(batch)
        state = self._normalize_state(state_raw).unsqueeze(1)
        action = self._normalize_action_delta(action_raw, state_raw)
        prompt_embeds = self._prompt_embeds(batch, batch_size=images.shape[0]).to(self.device)
        return {
            "images": images,
            "ref_images": ref_images,
            "state": state,
            "action": action,
            "prompt_embeds": prompt_embeds,
        }

    def _forward_vae(self, images: Tensor) -> Tensor:
        images = images.to(self.device, dtype=self.vae.dtype)
        with torch.no_grad():
            latents = self.vae.encode(images.permute(0, 2, 1, 3, 4).contiguous()).latent_dist.mode()
        return (latents - self._latents_mean) * self._latents_std

    def _get_timestep_and_sigma(self, batch_size: int) -> tuple[Tensor, Tensor]:
        sigma = torch.rand(batch_size, device=self.device)
        sigma = self.config.flow_shift * sigma / (1 + (self.config.flow_shift - 1) * sigma)
        timestep = torch.round(sigma * 1000).long()
        sigma = timestep.float() / 1000
        return timestep, sigma

    def _training_losses(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        images = batch["images"]
        bs = images.shape[0]
        prompt_embeds = batch["prompt_embeds"].to(device=self.device, dtype=self._dtype)
        action = batch["action"].to(device=self.device, dtype=torch.float32)
        state = batch["state"].to(device=self.device, dtype=torch.float32)

        timestep, sigma = self._get_timestep_and_sigma(bs)
        sigma_img = sigma.view(bs, 1, 1, 1, 1)
        visual_latents = self._forward_vae(images)
        visual_noise = torch.randn_like(visual_latents)
        visual_target = visual_noise - visual_latents
        noisy_latents = visual_noise * sigma_img + visual_latents * (1 - sigma_img)

        action_sigma = sigma.view(bs, 1, 1)
        action_noise = torch.randn_like(action)
        action_target = action_noise - action
        noisy_action = action_noise * action_sigma + action * (1 - action_sigma)

        ref_images = batch["ref_images"]
        ref_latents = self._forward_vae(ref_images[:, :1])
        num_latent_frames = visual_latents.shape[2]
        latent_height = visual_latents.shape[-2]
        latent_width = visual_latents.shape[-1]
        first_frame_mask = torch.ones(
            bs,
            1,
            num_latent_frames,
            latent_height,
            latent_width,
            dtype=visual_latents.dtype,
            device=visual_latents.device,
        )
        first_frame_mask[:, :, 0] = 0
        insert_noisy_latents = (1 - first_frame_mask) * ref_latents + first_frame_mask * noisy_latents
        insert_noisy_latents = insert_noisy_latents.to(self._dtype)

        ref_latents = insert_noisy_latents[:, :, :1]
        noisy_latents = insert_noisy_latents[:, :, 1:]
        num_state_tokens = state.shape[1]
        num_action_tokens = action.shape[1]
        frame_per_tokens = first_frame_mask.shape[-1] * first_frame_mask.shape[-2] // 4
        num_latent_tokens = frame_per_tokens * first_frame_mask.shape[2]
        timestep_full = torch.zeros(
            bs,
            num_state_tokens + num_action_tokens + num_latent_tokens,
            device=noisy_latents.device,
            dtype=noisy_latents.dtype,
        )
        num_clean_latent_tokens = frame_per_tokens
        timestep_full[:, num_state_tokens + num_clean_latent_tokens :] = timestep[:, None].to(timestep_full.dtype)

        visual_pred, action_pred = self.transformer(
            ref_latents=ref_latents,
            noisy_latents=noisy_latents,
            timestep=timestep_full,
            encoder_hidden_states=prompt_embeds,
            return_dict=False,
            action=noisy_action.to(self._dtype),
            state=state.to(self._dtype),
        )

        visual_loss = ((visual_pred.float() - visual_target.float()) * first_frame_mask).pow(2).mean()
        action_loss = (action_pred.float() - action_target.float()).pow(2).mean()
        return {"visual_loss": visual_loss, "action_loss": action_loss}

    def _composite_pil_from_condition(self, image_seq: Tensor) -> Image.Image:
        frame = image_seq[0].detach().float().cpu()
        frame = ((frame + 1.0) / 2.0).clamp(0.0, 1.0)
        frame = (frame.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(frame)
