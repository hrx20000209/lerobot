#!/usr/bin/env python

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from einops import rearrange
from torch import Tensor

from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import ACTION

from ..pretrained import PreTrainedPolicy
from .configuration_lingbo_va import LingBoVAConfig


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _dtype_from_config(dtype: str) -> torch.dtype:
    dtype = dtype.lower()
    if dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if dtype in {"float16", "fp16"}:
        return torch.float16
    if dtype in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported LingBoVA dtype: {dtype}")


class LingBoVAPolicy(PreTrainedPolicy):
    config_class = LingBoVAConfig
    name = "lingbo_va"

    _ACTION_HEAD_MODULES = ("action_embedder", "action_proj_out", "condition_embedder_action")
    _VIDEO_HEAD_MODULES = ("patch_embedding_mlp", "condition_embedder", "proj_out")

    def __init__(self, config: LingBoVAConfig, **kwargs):
        super().__init__(config)
        dataset_stats = kwargs.pop("dataset_stats", None)
        del kwargs
        config.validate_features()
        self.config = config
        self._action_queue: deque[Tensor] = deque(maxlen=config.n_action_steps)
        self._dataset_stats = dataset_stats
        self._action_stats = self._extract_action_stats(dataset_stats)
        self._model_action_stats = self._build_model_action_stats()
        if config.action_stats_path is not None:
            self._load_action_stats(Path(config.action_stats_path).expanduser())

        object.__setattr__(self, "_transformer", None)
        object.__setattr__(self, "_vae", None)
        object.__setattr__(self, "_streaming_vae", None)
        object.__setattr__(self, "_text_encoder", None)
        object.__setattr__(self, "_tokenizer", None)
        object.__setattr__(self, "_train_scheduler_latent", None)
        object.__setattr__(self, "_train_scheduler_action", None)
        object.__setattr__(self, "_flow_scheduler", None)
        object.__setattr__(self, "_action_scheduler", None)
        object.__setattr__(self, "_lingbo_utils", None)
        object.__setattr__(self, "_server_client", None)
        object.__setattr__(self, "_local_server", None)
        object.__setattr__(self, "_last_prompt", None)
        object.__setattr__(self, "_forward_calls", 0)
        object.__setattr__(self, "_freeze_report_done", False)
        object.__setattr__(self, "_empty_prompt_emb", None)
        object.__setattr__(self, "_mean_collapse_tracker", None)
        self._prompt_cache: dict[str, Tensor] = {}

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: LingBoVAConfig | None = None,
        **kwargs,
    ) -> LingBoVAPolicy:
        path = Path(pretrained_name_or_path).expanduser()
        if path.is_dir():
            if config is None:
                loaded = PreTrainedConfig.from_pretrained(path, **kwargs)
                if not isinstance(loaded, LingBoVAConfig):
                    raise TypeError(f"Expected LingBoVAConfig in {path}, got {type(loaded)}.")
                config = loaded
            if (path / "transformer").exists():
                config.transformer_path = path / "transformer"
            if (path / "transformer_lora").exists():
                config.use_transformer_lora = True
                config.transformer_lora_path = path / "transformer_lora"
            stats_path = path / "lingbo_va_action_stats.json"
            policy = cls(config, **kwargs)
            if stats_path.exists():
                policy._load_action_stats(stats_path)
            policy.eval()
            return policy
        return super().from_pretrained(pretrained_name_or_path, config=config, **kwargs)

    def _save_pretrained(self, save_directory: Path) -> None:
        old_transformer_path = self.config.transformer_path
        old_transformer_lora_path = self.config.transformer_lora_path
        try:
            if self._transformer is not None and self.config.use_transformer_lora:
                self.config.transformer_lora_path = Path("transformer_lora")
            elif self._transformer is not None:
                self.config.transformer_path = Path("transformer")
            self.config._save_pretrained(save_directory)
        finally:
            self.config.transformer_path = old_transformer_path
            self.config.transformer_lora_path = old_transformer_lora_path

        if self._transformer is not None:
            subdir = "transformer_lora" if self.config.use_transformer_lora else "transformer"
            self._transformer.save_pretrained(save_directory / subdir)
        self._save_action_stats(save_directory / "lingbo_va_action_stats.json")

    def train(self, mode: bool = True):
        super().train(mode)
        transformer = getattr(self, "_transformer", None)
        if transformer is not None:
            if mode:
                self._apply_training_freeze()
            else:
                transformer.eval()
        return self

    def parameters(self, recurse: bool = True):
        seen: set[int] = set()
        transformer = getattr(self, "_transformer", None)
        if transformer is not None:
            for param in transformer.parameters(recurse=recurse):
                seen.add(id(param))
                yield param
        for param in super().parameters(recurse=recurse):
            if id(param) not in seen:
                yield param

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ):
        seen: set[int] = set()
        transformer = getattr(self, "_transformer", None)
        if transformer is not None:
            model_prefix = f"{prefix}._transformer" if prefix else "_transformer"
            for name, param in transformer.named_parameters(
                prefix=model_prefix,
                recurse=recurse,
                remove_duplicate=remove_duplicate,
            ):
                seen.add(id(param))
                yield name, param
        for name, param in super().named_parameters(
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        ):
            if id(param) not in seen:
                yield name, param

    def get_optim_params(self) -> list:
        self._ensure_local_runtime(for_training=True)
        self._apply_training_freeze()
        trainable = [
            (name, param) for name, param in self._transformer.named_parameters() if param.requires_grad
        ]
        if not self.config.use_transformer_lora:
            return [param for _, param in trainable]

        lora_params = [param for name, param in trainable if "lora_" in name]
        head_params = [param for name, param in trainable if "lora_" not in name]
        if not lora_params:
            raise RuntimeError(
                "use_transformer_lora is enabled, but no trainable LoRA parameters were found."
            )
        logging.info(
            "LingBoVA optimizer groups: heads=%d params at %.2e, LoRA=%d params at %.2e",
            sum(param.numel() for param in head_params),
            self.config.optimizer_lr,
            sum(param.numel() for param in lora_params),
            self.config.lora_optimizer_lr,
        )
        return [
            {"params": head_params},
            {
                "params": lora_params,
                "lr": self.config.lora_optimizer_lr,
                "weight_decay": self.config.lora_optimizer_weight_decay,
            },
        ]

    def reset(self) -> None:
        self._action_queue.clear()
        object.__setattr__(self, "_last_prompt", None)
        local_server = getattr(self, "_local_server", None)
        if local_server is not None:
            local_server.transformer.clear_cache(local_server.cache_name)

    def forward(self, batch: dict[str, Tensor], fixed_timesteps: Tensor | None = None) -> tuple[Tensor, dict]:
        if self.config.train_attn_mode != "flex":
            raise ValueError(
                f"LingBoVA training requires train_attn_mode='flex' (got '{self.config.train_attn_mode}'). "
                "'torch'/'flashattn' attention ops ignore the block-causal training mask entirely "
                "(FlexAttnFunc.init_mask is only built inside WanTransformer3DModel.forward_train), so "
                "training with anything other than 'flex' would silently run full bidirectional "
                "attention across noisy/clean/video/action tokens — a correctness bug, not a perf knob."
            )
        self._ensure_local_runtime(for_training=True)
        self._apply_training_freeze()
        input_dict = self._build_training_input(batch, fixed_timesteps=fixed_timesteps)
        latent_pred, action_pred = self._transformer(input_dict, train_mode=True)
        latent_loss, action_loss = self._compute_loss(input_dict, latent_pred, action_pred)
        loss = self.config.video_loss_weight * latent_loss + self.config.action_loss_weight * action_loss
        self._forward_calls += 1
        if self._forward_calls == 1 or (
            self.config.loss_log_freq > 0 and self._forward_calls % self.config.loss_log_freq == 0
        ):
            logging.info(
                "LingBoVA losses at forward %d: video=%.6f action=%.6f total=%.6f",
                self._forward_calls,
                latent_loss.detach().item(),
                action_loss.detach().item(),
                loss.detach().item(),
            )
        return loss, {
            "loss_video": float(latent_loss.detach().item()),
            "loss_action": float(action_loss.detach().item()),
        }

    @torch.no_grad()
    def select_action(self, batch: dict[str, Any]) -> Tensor:
        self.eval()
        if not self._action_queue:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Any], **kwargs) -> Tensor:
        del kwargs
        self.eval()
        if self.config.inference_backend == "server":
            action_chunk = self._predict_action_chunk_server(batch)
        else:
            action_chunk = self._predict_action_chunk_local(batch)
        return action_chunk[:, : self.config.chunk_size]

    def _install_lingbo_import_path(self) -> Path:
        root = self._resolve_lingbo_root()
        for path in (root, root / "wan_va"):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)
        return root

    def _resolve_lingbo_root(self) -> Path:
        candidates = [
            self.config.lingbo_root,
            os.environ.get("LINGBOT_ROOT"),
            Path("/home/rxhuang/Projects/lingbot-va"),
        ]
        for candidate in candidates:
            if _is_none_like(candidate):
                continue
            path = Path(candidate).expanduser().resolve()
            if (path / "wan_va").exists():
                return path
        raise FileNotFoundError("Could not find LingBot-VA checkout. Set policy.lingbo_root.")

    def _resolve_model_root(self) -> Path:
        if _is_none_like(self.config.model_root):
            raise FileNotFoundError("LingBoVA model_root is required.")
        root = Path(self.config.model_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"LingBoVA model_root not found: {root}")
        return root

    def _resolve_transformer_path(self) -> Path:
        if self.config.transformer_path is not None:
            path = Path(self.config.transformer_path).expanduser()
            if not path.is_absolute() and self.config.pretrained_path is not None:
                path = Path(self.config.pretrained_path).expanduser() / path
            path = path.resolve()
            if path.exists():
                return path
            raise FileNotFoundError(f"LingBoVA transformer path not found: {path}")
        return self._resolve_model_root() / "transformer"

    def _resolve_transformer_lora_path(self) -> Path | None:
        if self.config.transformer_lora_path is None:
            return None
        path = Path(self.config.transformer_lora_path).expanduser()
        if not path.is_absolute() and self.config.pretrained_path is not None:
            path = Path(self.config.pretrained_path).expanduser() / path
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"LingBoVA Transformer LoRA path not found: {path}")
        return path

    def _lora_modules_to_save(self) -> list[str]:
        modules = list(self._ACTION_HEAD_MODULES)
        if self.config.train_video_head:
            modules.extend(self._VIDEO_HEAD_MODULES)
        return modules

    def _apply_transformer_lora(self, transformer, for_training: bool):
        from peft import LoraConfig, PeftModel, get_peft_model

        adapter_path = self._resolve_transformer_lora_path()
        if adapter_path is not None:
            transformer = PeftModel.from_pretrained(
                transformer,
                adapter_path,
                is_trainable=for_training,
            )
            logging.info("Loaded LingBoVA Transformer LoRA from %s", adapter_path)
            return transformer

        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            target_modules=list(self.config.lora_target_modules),
            modules_to_save=self._lora_modules_to_save(),
        )
        transformer = get_peft_model(transformer, lora_config)
        logging.info(
            "Initialized LingBoVA Transformer LoRA: rank=%d alpha=%d targets=%s",
            self.config.lora_rank,
            self.config.lora_alpha,
            self.config.lora_target_modules,
        )
        return transformer

    def _ensure_local_runtime(self, for_training: bool) -> None:
        if self._transformer is not None and self._vae is not None and self._text_encoder is not None:
            return

        self._install_lingbo_import_path()
        try:
            from wan_va.distributed.fsdp import apply_ac
            from wan_va.modules.utils import (
                WanVAEStreamingWrapper,
                load_text_encoder,
                load_tokenizer,
                load_transformer,
                load_vae,
            )
            from wan_va.utils import data_seq_to_patch, get_mesh_id, sample_timestep_id
            from wan_va.utils.scheduler import FlowMatchScheduler
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "LingBoVA policy needs LingBot-VA dependencies. Install diffusers/transformers/easydict "
                "in the environment running LeRobot, or use inference_backend=server for deployment."
            ) from exc

        model_root = self._resolve_model_root()
        dtype = _dtype_from_config(self.config.dtype)
        transformer_device = self.config.transformer_device or self.config.device
        vae_device = self.config.vae_device or transformer_device
        text_device = self.config.text_encoder_device or transformer_device

        transformer = load_transformer(
            str(self._resolve_transformer_path()),
            torch_dtype=dtype,
            torch_device=transformer_device,
            attn_mode=self.config.train_attn_mode if for_training else self.config.inference_attn_mode,
        )
        if for_training and self.config.gradient_checkpointing:
            try:
                apply_ac(transformer)
            except Exception as exc:
                if hasattr(transformer, "enable_gradient_checkpointing"):
                    try:
                        transformer.enable_gradient_checkpointing()
                    except ValueError:
                        logging.warning("LingBoVA gradient checkpointing is unavailable: %s", exc)
                else:
                    logging.warning("LingBoVA gradient checkpointing is unavailable: %s", exc)
        if self.config.use_transformer_lora:
            transformer = self._apply_transformer_lora(transformer, for_training=for_training)

        vae = load_vae(str(model_root / "vae"), torch_dtype=dtype, torch_device=vae_device)
        text_encoder = load_text_encoder(
            str(model_root / "text_encoder"),
            torch_dtype=dtype,
            torch_device=text_device,
        )
        tokenizer = load_tokenizer(str(model_root / "tokenizer"))

        object.__setattr__(self, "_transformer", transformer)
        object.__setattr__(self, "_vae", vae)
        object.__setattr__(self, "_streaming_vae", WanVAEStreamingWrapper(vae))
        object.__setattr__(self, "_text_encoder", text_encoder)
        object.__setattr__(self, "_tokenizer", tokenizer)
        object.__setattr__(
            self,
            "_train_scheduler_latent",
            FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True),
        )
        object.__setattr__(
            self,
            "_train_scheduler_action",
            FlowMatchScheduler(shift=1.0, sigma_min=0.0, extra_one_step=True),
        )
        self._train_scheduler_latent.set_timesteps(1000, training=True)
        self._train_scheduler_action.set_timesteps(1000, training=True)
        object.__setattr__(
            self,
            "_lingbo_utils",
            {
                "data_seq_to_patch": data_seq_to_patch,
                "get_mesh_id": get_mesh_id,
                "sample_timestep_id": sample_timestep_id,
            },
        )

        if for_training:
            self._apply_training_freeze()
        else:
            transformer.eval().requires_grad_(False)
            vae.eval().requires_grad_(False)
            text_encoder.eval().requires_grad_(False)

    def _apply_training_freeze(self) -> None:
        transformer = getattr(self, "_transformer", None)
        if transformer is None:
            return
        transformer.train()
        if self.config.use_transformer_lora:
            transformer.requires_grad_(False)
            transformer.set_adapter(transformer.active_adapter)
        elif not self.config.train_action_head_only:
            transformer.requires_grad_(True)
        else:
            transformer.requires_grad_(False)
            for name in self._ACTION_HEAD_MODULES:
                module = getattr(transformer, name, None)
                if module is not None:
                    module.train()
                    module.requires_grad_(True)
            if self.config.train_video_head:
                for name in self._VIDEO_HEAD_MODULES:
                    module = getattr(transformer, name, None)
                    if module is not None:
                        module.train()
                        module.requires_grad_(True)
                transformer.scale_shift_table.requires_grad_(True)
            if self.config.train_last_n_blocks > 0:
                blocks = getattr(transformer, "blocks", None)
                if blocks is None:
                    raise AttributeError(
                        "LingBoVA transformer has no `.blocks` attribute; train_last_n_blocks is unsupported."
                    )
                for block in blocks[-self.config.train_last_n_blocks :]:
                    block.train()
                    block.requires_grad_(True)
        for module in (getattr(self, "_vae", None), getattr(self, "_text_encoder", None)):
            if module is not None and self.config.freeze_vision_text_encoder:
                module.eval()
                module.requires_grad_(False)
        if not getattr(self, "_freeze_report_done", False):
            self._report_trainable_params(transformer)
            object.__setattr__(self, "_freeze_report_done", True)

    def _report_trainable_params(self, transformer: Any) -> None:
        total = sum(p.numel() for p in transformer.parameters())
        trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
        pct = 100.0 * trainable / max(total, 1)
        trainable_module_prefixes = sorted(
            {name.split(".")[0] for name, p in transformer.named_parameters() if p.requires_grad}
        )
        logging.info(
            "LingBoVA trainable params: %d / %d (%.4f%%). Trainable top-level modules: %s",
            trainable,
            total,
            pct,
            trainable_module_prefixes,
        )
        if pct < 0.5:
            raise RuntimeError(
                f"LingBoVA trainable parameters are {pct:.4f}% of the transformer, below the 0.5% "
                "sanity floor. This almost always means the freeze configuration froze everything "
                "that matters (e.g. train_action_head_only=true with use_transformer_lora=false and "
                "train_last_n_blocks=0, which never unfreezes any transformer block). Check "
                "train_action_head_only / use_transformer_lora / train_last_n_blocks."
            )

    def _build_training_input(
        self, batch: dict[str, Any], fixed_timesteps: Tensor | None = None
    ) -> dict[str, Any]:
        latents = self._encode_video_latents(batch)
        actions, actions_mask = self._build_model_actions(batch)
        prompts = self._build_prompts(batch, batch_size=latents.shape[0])
        text_emb = self._encode_prompts(prompts).to(device=latents.device, dtype=latents.dtype)

        if self.training and self.config.cfg_prob > 0.0 and fixed_timesteps is None:
            drop_mask = torch.rand(text_emb.shape[0], device=text_emb.device) < self.config.cfg_prob
            if drop_mask.any():
                empty_emb = self._empty_prompt_embedding().to(device=text_emb.device, dtype=text_emb.dtype)
                text_emb = torch.where(drop_mask[:, None, None], empty_emb[None], text_emb)

        latent_dict = self._add_noise(
            latents,
            self._train_scheduler_latent,
            action_mask=None,
            action_mode=False,
            fixed_timesteps=fixed_timesteps,
        )
        action_dict = self._add_noise(
            actions,
            self._train_scheduler_action,
            action_mask=actions_mask,
            action_mode=True,
            fixed_timesteps=fixed_timesteps,
        )
        latent_dict["text_emb"] = text_emb
        action_dict["text_emb"] = text_emb
        action_dict["actions_mask"] = actions_mask
        return {
            "latent_dict": latent_dict,
            "action_dict": action_dict,
            "chunk_size": self.config.frame_chunk_size,
            "window_size": self.config.attn_window,
        }

    def _empty_prompt_embedding(self) -> Tensor:
        if self._empty_prompt_emb is None:
            with torch.no_grad():
                emb = self._encode_prompts([""])[0]
            object.__setattr__(self, "_empty_prompt_emb", emb)
        return self._empty_prompt_emb

    @torch.no_grad()
    def _add_noise(
        self,
        latent: Tensor,
        train_scheduler: Any,
        action_mask: Tensor | None,
        action_mode: bool,
        fixed_timesteps: Tensor | None = None,
    ) -> dict[str, Tensor]:
        batch_size, _, num_frames, height, width = latent.shape
        if fixed_timesteps is not None:
            fractions = fixed_timesteps.to(torch.float64)
            timestep_ids = (fractions * train_scheduler.num_train_timesteps).long()
            timestep_ids = timestep_ids.clamp(min=0, max=train_scheduler.num_train_timesteps - 1)
            if timestep_ids.numel() == 1:
                timestep_ids = timestep_ids.expand(num_frames)
            elif timestep_ids.numel() != num_frames:
                timestep_ids = timestep_ids[torch.arange(num_frames) % timestep_ids.numel()]
        else:
            sample_timestep_id = self._lingbo_utils["sample_timestep_id"]
            timestep_ids = sample_timestep_id(
                batch_size=num_frames, num_train_timesteps=train_scheduler.num_train_timesteps
            )
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=latent.device)
        noisy_latents = train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets = train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = (1, 1, 1) if action_mode else self._transformer.patch_size
        get_mesh_id = self._lingbo_utils["get_mesh_id"]
        grid_id = get_mesh_id(
            num_frames // patch_f,
            height // patch_h,
            width // patch_w,
            t=1 if action_mode else 0,
            f_w=1,
            f_shift=0,
            action=action_mode,
        ).to(latent.device)
        grid_id = grid_id[None].repeat(batch_size, 1, 1)

        cond_timesteps = torch.zeros_like(timesteps)
        if action_mask is not None:
            noisy_latents = noisy_latents * action_mask.float()
            targets = targets * action_mask.float()
            latent = latent * action_mask.float()

        return {
            "timesteps": timesteps[None].repeat(batch_size, 1),
            "noisy_latents": noisy_latents,
            "targets": targets,
            "latent": latent,
            "cond_timesteps": cond_timesteps[None].repeat(batch_size, 1),
            "grid_id": grid_id,
        }

    def _compute_loss(
        self, input_dict: dict[str, Any], latent_pred: Tensor, action_pred: Tensor
    ) -> tuple[Tensor, Tensor]:
        action_targets = input_dict["action_dict"]["targets"]
        latent_targets = input_dict["latent_dict"]["targets"]
        action_pred = rearrange(action_pred, "b (f n) c -> b c f n 1", f=action_targets.shape[-3])
        data_seq_to_patch = self._lingbo_utils["data_seq_to_patch"]
        latent_pred = data_seq_to_patch(
            self._transformer.patch_size,
            latent_pred,
            latent_targets.shape[-3],
            latent_targets.shape[-2],
            latent_targets.shape[-1],
            batch_size=latent_pred.shape[0],
        )
        batch_size, num_frames = input_dict["latent_dict"]["timesteps"].shape
        latent_weight = self._train_scheduler_latent.training_weight(
            input_dict["latent_dict"]["timesteps"].flatten()
        ).reshape(batch_size, num_frames)
        action_weight = self._train_scheduler_action.training_weight(
            input_dict["action_dict"]["timesteps"].flatten()
        ).reshape(batch_size, num_frames)

        latent_loss = F.mse_loss(latent_pred.float(), latent_targets.float().detach(), reduction="none")
        latent_loss = latent_loss * latent_weight[:, None, :, None, None]
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1).mean(dim=1).mean()

        action_loss = F.mse_loss(action_pred.float(), action_targets.float().detach(), reduction="none")
        action_loss = action_loss * action_weight[:, None, :, None, None]
        action_mask = input_dict["action_dict"]["actions_mask"].float()
        action_loss = action_loss * action_mask
        action_loss = action_loss.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        action_mask = action_mask.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1)
        action_loss = (action_loss.sum(dim=1) / (action_mask.sum(dim=1) + 1e-6)).mean()
        return latent_loss, action_loss

    @torch.no_grad()
    def _encode_video_latents(self, batch: dict[str, Any]) -> Tensor:
        camera_keys = self._resolve_camera_keys(batch)
        num_vae_input_frames = (self.config.frame_chunk_size - 1) * self.config.vae_temporal_stride + 1
        frame_ids = [idx * self.config.video_frame_stride for idx in range(num_vae_input_frames)]
        videos = []
        for key in camera_keys:
            seq = self._as_image_sequence(batch[key])[:, frame_ids]
            videos.append(self._resize_image_sequence(seq, self.config.image_size))
        video = torch.stack(videos, dim=1)
        batch_size, num_cameras, num_frames, channels, height, width = video.shape
        video = video.reshape(batch_size * num_cameras, num_frames, channels, height, width)
        video = video.permute(0, 2, 1, 3, 4).contiguous()

        target_vae_device = torch.device(self.config.vae_device or self.config.device)
        if next(self._vae.parameters()).device != target_vae_device:
            self._vae.to(target_vae_device)
        vae_device = next(self._vae.parameters()).device
        encoded_chunks = []
        for video_chunk in video.split(self.config.vae_encode_batch_size):
            self._streaming_vae.clear_cache()
            encoded_chunks.append(
                self._encode_vae_chunk(video_chunk.to(device=vae_device, dtype=self._vae.dtype))
            )
        enc = torch.cat(encoded_chunks, dim=0)
        mu, _ = torch.chunk(enc, 2, dim=1)
        latents_mean = torch.tensor(self._vae.config.latents_mean, device=mu.device, dtype=torch.float32)
        latents_std = torch.tensor(self._vae.config.latents_std, device=mu.device, dtype=torch.float32)
        mu = ((mu.float() - latents_mean.view(1, -1, 1, 1, 1)) / latents_std.view(1, -1, 1, 1, 1)).to(
            mu.dtype
        )
        _, channels_latent, frames_latent, height_latent, width_latent = mu.shape
        mu = mu.reshape(batch_size, num_cameras, channels_latent, frames_latent, height_latent, width_latent)
        latents = torch.cat([mu[:, idx] for idx in range(num_cameras)], dim=-1)
        if latents.shape[2] != self.config.frame_chunk_size:
            raise ValueError(
                "LingBoVA VAE produced an unexpected number of latent frames: "
                f"{latents.shape[2]} vs frame_chunk_size={self.config.frame_chunk_size}. "
                "Check vae_temporal_stride/video_frame_stride/num_frames."
            )
        latents = latents.to(
            device=next(self._transformer.parameters()).device, dtype=_dtype_from_config(self.config.dtype)
        )
        if self.config.offload_vae_after_encode and target_vae_device.type == "cuda":
            self._vae.to("cpu")
            torch.cuda.empty_cache()
        return latents

    def _encode_vae_chunk(self, video: Tensor) -> Tensor:
        try:
            return self._vae.encode(video).latent_dist.parameters
        except torch.OutOfMemoryError:
            raise
        except Exception:
            return self._streaming_vae.encode_chunk(video)

    def _build_model_actions(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor]:
        action = torch.as_tensor(batch[ACTION], dtype=torch.float32)
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.shape[1] < self.config.chunk_size:
            raise ValueError(f"LingBoVA needs {self.config.chunk_size} actions, got {action.shape[1]}.")
        action = action[:, : self.config.chunk_size]
        if action.shape[-1] != self.config.action_dim:
            raise ValueError(f"Expected action_dim={self.config.action_dim}, got {action.shape[-1]}.")

        batch_size = action.shape[0]
        action_device = action.device
        padded = torch.zeros(
            batch_size,
            self.config.chunk_size,
            self.config.action_dim + 1,
            dtype=torch.float32,
            device=action_device,
        )
        padded[:, :, : self.config.action_dim] = action
        inverse = self._inverse_used_action_channel_ids()
        aligned = padded[:, :, inverse]

        normalized = self._normalize_model_action(aligned)
        normalized = rearrange(
            normalized,
            "b (f n) c -> b c f n 1",
            f=self.config.frame_chunk_size,
            n=self.config.action_per_frame,
        )

        valid = torch.ones(
            batch_size,
            self.config.chunk_size,
            self.config.lingbo_action_dim,
            dtype=torch.bool,
            device=action_device,
        )
        if ACTION + "_is_pad" in batch:
            valid = (
                valid
                & ~torch.as_tensor(batch[ACTION + "_is_pad"], dtype=torch.bool)[
                    :, : self.config.chunk_size, None
                ]
            )
        channel_mask = torch.zeros(self.config.lingbo_action_dim, dtype=torch.bool, device=action_device)
        channel_mask[self.config.used_action_channel_ids] = True
        valid = valid & channel_mask.view(1, 1, -1)
        valid = rearrange(
            valid, "b (f n) c -> b c f n 1", f=self.config.frame_chunk_size, n=self.config.action_per_frame
        )

        device = next(self._transformer.parameters()).device
        dtype = _dtype_from_config(self.config.dtype)
        return normalized.to(device=device, dtype=dtype), valid.to(device=device)

    def _normalize_model_action(self, action: Tensor) -> Tensor:
        stats = self._model_action_stats
        q01 = stats["q01"].to(action.device)
        q99 = stats["q99"].to(action.device)
        if self.config.norm_default_mode == "q01/q99":
            return torch.clamp((action - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0, -5.0, 5.0)
        if self.config.norm_default_mode == "min/max":
            low, high = stats["min"].to(action.device), stats["max"].to(action.device)
            return torch.clamp((action - low) / (high - low + 1e-6) * 2.0 - 1.0, -5.0, 5.0)
        mean, std = stats["mean"].to(action.device), stats["std"].to(action.device)
        return torch.clamp((action - mean) / (std + 1e-8), -5.0, 5.0)

    def _inverse_used_action_channel_ids(self) -> list[int]:
        inverse = [self.config.action_dim] * self.config.lingbo_action_dim
        for compact_idx, model_idx in enumerate(self.config.used_action_channel_ids):
            inverse[int(model_idx)] = compact_idx
        return inverse

    def _resolve_camera_keys(self, batch: dict[str, Any]) -> list[str]:
        if self.config.camera_keys:
            missing = [key for key in self.config.camera_keys if key not in batch]
            if missing:
                raise KeyError(f"Configured LingBoVA camera_keys missing from batch: {missing}")
            return list(self.config.camera_keys)
        for candidates in (
            [self.config.front_camera_key, self.config.right_camera_key],
            [self.config.front_camera_key, self.config.wrist_camera_key],
        ):
            if all(key in batch for key in candidates):
                return candidates
        visual_keys = [key for key in self.config.image_features if key in batch]
        if not visual_keys:
            raise KeyError("LingBoVA could not find visual observation keys in the batch.")
        return visual_keys

    def _server_camera_keys(self, camera_keys: list[str]) -> list[str]:
        if self.config.server_camera_keys:
            if len(self.config.server_camera_keys) != len(camera_keys):
                raise ValueError("server_camera_keys length must match camera_keys length.")
            return list(self.config.server_camera_keys)
        return camera_keys

    def _as_image_sequence(self, value: Any) -> Tensor:
        image = torch.as_tensor(value)
        if image.ndim == 4:
            image = image.unsqueeze(0)
        if image.ndim != 5:
            raise ValueError(f"Expected image sequence [B,T,C,H,W] or [B,T,H,W,C], got {tuple(image.shape)}")
        if image.shape[-1] in {1, 3, 4} and image.shape[2] not in {1, 3, 4}:
            image = image.permute(0, 1, 4, 2, 3)
        if image.shape[2] > 3:
            image = image[:, :, :3]
        if image.shape[2] != 3:
            raise ValueError(f"LingBoVA expects RGB images, got shape {tuple(image.shape)}")
        image = image.to(dtype=torch.float32)
        if float(image.detach().amax().cpu()) > 2.0:
            image = image / 255.0
        if float(image.detach().amin().cpu()) >= 0.0:
            image = image * 2.0 - 1.0
        if image.shape[1] < self.config.num_frames:
            raise ValueError(
                f"LingBoVA needs {self.config.num_frames} observation frames, got {image.shape[1]}."
            )
        return image[:, : self.config.num_frames]

    def _as_current_image_uint8(self, value: Any) -> np.ndarray:
        image = torch.as_tensor(value)
        if image.ndim == 5:
            image = image[:, -1]
        if image.ndim == 4:
            image = image[0]
        if image.shape[0] in {1, 3, 4}:
            image = image[:3].permute(1, 2, 0)
        image = image.detach().cpu()
        if image.dtype != torch.uint8:
            if float(image.max()) <= 2.0:
                if float(image.min()) < 0.0:
                    image = (image + 1.0) / 2.0
                image = image * 255.0
            image = image.clamp(0, 255).to(torch.uint8)
        return image.numpy()

    def _resize_image_sequence(self, video: Tensor, size_hw: tuple[int, int]) -> Tensor:
        batch_size, num_frames, channels, height, width = video.shape
        flat = video.reshape(batch_size * num_frames, channels, height, width)
        resized = F.interpolate(flat, size=size_hw, mode="bilinear", align_corners=False)
        return resized.reshape(batch_size, num_frames, channels, *size_hw)

    def _build_prompts(self, batch: dict[str, Any], batch_size: int) -> list[str]:
        tasks = batch.get(self.config.task_key, self.config.default_task)
        if isinstance(tasks, str):
            task_list = [tasks] * batch_size
        elif isinstance(tasks, list | tuple):
            task_list = list(tasks)
            if len(task_list) == 1 and batch_size > 1:
                task_list = task_list * batch_size
        else:
            task_list = [str(tasks)] * batch_size
        if len(task_list) != batch_size:
            raise ValueError(f"Expected {batch_size} prompts, got {len(task_list)}.")
        return [self.config.prompt_template.format(task=str(task)) for task in task_list]

    @torch.no_grad()
    def _encode_prompts(self, prompts: list[str]) -> Tensor:
        uncached = list(dict.fromkeys(prompt for prompt in prompts if prompt not in self._prompt_cache))
        if uncached:
            from diffusers.pipelines.wan.pipeline_wan import prompt_clean

            cleaned = [prompt_clean(prompt) for prompt in uncached]
            inputs = self._tokenizer(
                cleaned,
                padding=True,
                max_length=512,
                truncation=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            target_text_device = torch.device(self.config.text_encoder_device or self.config.device)
            if next(self._text_encoder.parameters()).device != target_text_device:
                self._text_encoder.to(target_text_device)
            text_device = next(self._text_encoder.parameters()).device
            embeds = self._text_encoder(
                inputs.input_ids.to(text_device),
                inputs.attention_mask.to(text_device),
            ).last_hidden_state
            seq_lens = inputs.attention_mask.gt(0).sum(dim=1).long()
            padded = []
            for emb, valid in zip(embeds, seq_lens, strict=False):
                cur = emb[:valid]
                cur = torch.cat([cur, cur.new_zeros(512 - cur.shape[0], cur.shape[1])], dim=0)
                padded.append(cur.detach().cpu())
            for prompt, emb in zip(uncached, padded, strict=False):
                self._prompt_cache[prompt] = emb
            if self.config.offload_text_encoder_after_encode and target_text_device.type == "cuda":
                self._text_encoder.to("cpu")
                torch.cuda.empty_cache()
        out = torch.stack([self._prompt_cache[prompt] for prompt in prompts], dim=0)
        return out

    def _extract_action_stats(self, dataset_stats: dict[str, dict[str, Any]] | None) -> dict[str, Tensor]:
        if dataset_stats is None or ACTION not in dataset_stats:
            zeros = torch.zeros(self.config.action_dim)
            ones = torch.ones(self.config.action_dim)
            return {"q01": -ones, "q99": ones, "min": -ones, "max": ones, "mean": zeros, "std": ones}
        stats = dataset_stats[ACTION]
        return {
            "q01": self._to_float_tensor(stats.get("q01", stats.get("min"))),
            "q99": self._to_float_tensor(stats.get("q99", stats.get("max"))),
            "min": self._to_float_tensor(stats.get("min", stats.get("q01"))),
            "max": self._to_float_tensor(stats.get("max", stats.get("q99"))),
            "mean": self._to_float_tensor(stats.get("mean", torch.zeros(self.config.action_dim))),
            "std": self._to_float_tensor(stats.get("std", torch.ones(self.config.action_dim))),
        }

    def _build_model_action_stats(self) -> dict[str, Tensor]:
        out = {
            "q01": torch.zeros(self.config.lingbo_action_dim),
            "q99": torch.ones(self.config.lingbo_action_dim),
            "min": torch.zeros(self.config.lingbo_action_dim),
            "max": torch.ones(self.config.lingbo_action_dim),
            "mean": torch.zeros(self.config.lingbo_action_dim),
            "std": torch.ones(self.config.lingbo_action_dim),
        }
        for compact_idx, model_idx in enumerate(self.config.used_action_channel_ids):
            for key in out:
                out[key][int(model_idx)] = self._action_stats[key][compact_idx]
        return out

    def _to_float_tensor(self, value: Any) -> Tensor:
        if isinstance(value, Tensor):
            return value.detach().cpu().to(dtype=torch.float32)
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).to(dtype=torch.float32)
        return torch.as_tensor(value, dtype=torch.float32)

    def _save_action_stats(self, path: Path) -> None:
        data = {key: value.detach().cpu().tolist() for key, value in self._action_stats.items()}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load_action_stats(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self._action_stats = {key: torch.as_tensor(value, dtype=torch.float32) for key, value in data.items()}
        self._model_action_stats = self._build_model_action_stats()

    def _make_server_payload(self, batch: dict[str, Any], reset: bool = False) -> dict[str, Any]:
        camera_keys = self._resolve_camera_keys(batch)
        server_keys = self._server_camera_keys(camera_keys)
        obs = {
            out_key: self._as_current_image_uint8(batch[in_key])
            for in_key, out_key in zip(camera_keys, server_keys, strict=False)
        }
        payload = {
            "obs": [obs],
            "reset": reset,
            "video_guidance_scale": self.config.video_guidance_scale,
            "action_guidance_scale": self.config.action_guidance_scale,
            "return_profile": self.config.return_server_profile,
        }
        if self.config.video_inference_steps is not None:
            payload["video_inference_steps"] = int(self.config.video_inference_steps)
        if self.config.action_inference_steps is not None:
            payload["action_inference_steps"] = int(self.config.action_inference_steps)
        if self.config.video_exec_step is not None:
            payload["video_exec_step"] = int(self.config.video_exec_step)
        return payload

    def _predict_action_chunk_server(self, batch: dict[str, Any]) -> Tensor:
        # NOTE: this call intentionally does NOT send `compute_kv_cache` — that would require
        # knowing which actions were *actually executed* (post safety-clip) on the robot, which
        # is only known outside this policy, by the rollout loop. Committing the previous
        # chunk's executed actions (and thereby advancing the server's `frame_st_id`) is the
        # caller's responsibility via `commit_executed_action()`, which must be invoked once per
        # chunk boundary before the next `predict_action_chunk` call. See
        # `LingboVaKvCacheStrategy` in `src/lerobot/rollout/strategies/lingbo_va_kv_cache.py`.
        client = self._ensure_server_client()
        prompt = self._build_prompts(batch, batch_size=1)[0]
        if prompt != self._last_prompt:
            reset_payload = self._make_server_payload(batch, reset=True)
            reset_payload["prompt"] = prompt
            client.infer(reset_payload)
            object.__setattr__(self, "_last_prompt", prompt)
        response = client.infer(self._make_server_payload(batch, reset=False))
        return self._server_action_to_tensor(response["action"])

    def _predict_action_chunk_local(self, batch: dict[str, Any]) -> Tensor:
        # See the NOTE in `_predict_action_chunk_server` above — same caveat applies here.
        server = self._ensure_local_server()
        prompt = self._build_prompts(batch, batch_size=1)[0]
        if prompt != self._last_prompt:
            reset_payload = self._make_server_payload(batch, reset=True)
            reset_payload["prompt"] = prompt
            server.infer(reset_payload)
            object.__setattr__(self, "_last_prompt", prompt)
        response = server.infer(self._make_server_payload(batch, reset=False))
        return self._server_action_to_tensor(response["action"])

    def _build_kv_cache_payload(self, executed_action_chunk: Tensor, obs: dict[str, Any]) -> dict[str, Any]:
        """Pure `compute_kv_cache` payload builder — no dispatch. Shared by `commit_executed_action`
        (dispatches via the policy's configured server/local backend) and by `eval_utils`'
        in-process weight-sharing server (Task 2.4's open-loop eval), which must send this payload
        directly to ITS OWN server object rather than through `commit_executed_action`'s dispatch
        — that would open a websocket to `server_host`/`server_port` (nothing listens there during
        training) or spin up a second, independent, disk-loaded `VA_Server` (doubling GPU memory).

        `executed_action_chunk`: Tensor of shape (T, action_dim) in raw (un-normalized) per-step
        dataset-action units. `T` must be a multiple of `config.action_per_frame` (commonly
        `T == config.n_action_steps`), since the server's KV cache operates on whole latent-frame
        granularity. `obs`: the observation batch used to encode the camera frames committed
        alongside the actions.
        """
        if executed_action_chunk.ndim != 2 or executed_action_chunk.shape[-1] != self.config.action_dim:
            raise ValueError(
                "commit_executed_action expects shape (T, action_dim="
                f"{self.config.action_dim}), got {tuple(executed_action_chunk.shape)}."
            )
        num_steps = executed_action_chunk.shape[0]
        if num_steps % self.config.action_per_frame != 0:
            raise ValueError(
                f"commit_executed_action received {num_steps} executed steps, which is not a "
                f"multiple of action_per_frame={self.config.action_per_frame}. Configure "
                "n_action_steps to be a multiple of action_per_frame so executed chunks align "
                "with the server's KV-cache latent-frame granularity."
            )
        num_frames = num_steps // self.config.action_per_frame
        arr = executed_action_chunk.detach().cpu().numpy().astype(np.float32)
        arr = arr.reshape(num_frames, self.config.action_per_frame, self.config.action_dim)
        arr = np.transpose(arr, (2, 0, 1))  # -> [action_dim, num_frames, action_per_frame]

        payload = self._make_server_payload(obs, reset=False)
        payload["compute_kv_cache"] = True
        payload["imagine"] = False
        payload["pred_action"] = arr
        return payload

    def commit_executed_action(self, executed_action_chunk: Tensor, obs: dict[str, Any]) -> None:
        """Commit actually-executed actions to the inference server's KV cache.

        Must be called exactly once per action-chunk boundary — after the robot has executed the
        actions `select_action` returned for the current chunk, and before the next
        `predict_action_chunk` call fires — mirroring the upstream protocol: reset once -> infer
        chunk -> execute chunk -> compute_kv_cache(executed actions) -> infer next chunk -> ...
        Without this call, the server's `frame_st_id` never advances and every
        `predict_action_chunk` call silently replans from scratch instead of streaming.

        See `_build_kv_cache_payload` for the argument contract. This method dispatches the
        resulting payload via `config.inference_backend` (server websocket or local disk-loaded
        `VA_Server`) — for the in-process training-time eval server, call
        `server.infer(policy._build_kv_cache_payload(...))` directly instead.
        """
        payload = self._build_kv_cache_payload(executed_action_chunk, obs)
        if self.config.inference_backend == "server":
            self._ensure_server_client().infer(payload)
        else:
            self._ensure_local_server().infer(payload)

    def _server_action_to_tensor(self, action: Any) -> Tensor:
        arr = np.asarray(action, dtype=np.float32)
        if arr.ndim == 3:
            arr = np.transpose(arr, (1, 2, 0)).reshape(-1, arr.shape[0])
        elif arr.ndim != 2:
            raise ValueError(f"Unexpected LingBoVA action response shape: {arr.shape}")
        if arr.shape[-1] == self.config.lingbo_action_dim:
            arr = arr[:, self.config.used_action_channel_ids]
        elif arr.shape[-1] != self.config.action_dim:
            raise ValueError(
                "Unexpected LingBoVA action channel count: "
                f"{arr.shape[-1]} (expected {self.config.action_dim} or {self.config.lingbo_action_dim})."
            )
        return torch.from_numpy(arr).unsqueeze(0)

    def _ensure_server_client(self):
        if self._server_client is not None:
            return self._server_client

        try:
            import msgpack
            from websockets.sync.client import connect
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("LingBoVA server inference needs `websockets` and `msgpack`.") from exc

        class Client:
            def __init__(self, host: str, port: int, retry_s: float):
                self.uri = f"ws://{host}:{port}"
                self.retry_s = retry_s
                while True:
                    try:
                        self.websocket = connect(self.uri, max_size=None)
                        self.websocket.recv()
                        break
                    except Exception:
                        time.sleep(self.retry_s)

            def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
                self.websocket.send(msgpack.packb(_pack_array(payload), use_bin_type=True))
                response = self.websocket.recv()
                if isinstance(response, str):
                    raise RuntimeError(response)
                return msgpack.unpackb(response, raw=False, object_hook=_unpack_array)

        client = Client(self.config.server_host, self.config.server_port, self.config.connect_retry_s)
        object.__setattr__(self, "_server_client", client)
        return client

    def _make_va_job_config(self):
        self._install_lingbo_import_path()
        from easydict import EasyDict
        from wan_va.configs import VA_CONFIGS

        base = deepcopy(VA_CONFIGS.get(self.config.va_config_name, VA_CONFIGS["demo"]))
        cfg = EasyDict(base)
        cfg.wan22_pretrained_model_name_or_path = str(self._resolve_model_root())
        cfg.transformer_path = str(self._resolve_transformer_path())
        cfg.attn_window = int(self.config.attn_window)
        cfg.frame_chunk_size = int(self.config.frame_chunk_size)
        cfg.env_type = self.config.env_type
        cfg.height, cfg.width = self.config.image_size
        cfg.action_dim = int(self.config.lingbo_action_dim)
        cfg.action_per_frame = int(self.config.action_per_frame)
        cfg.obs_cam_keys = self._server_camera_keys(
            self.config.camera_keys or [self.config.front_camera_key, self.config.right_camera_key]
        )
        cfg.guidance_scale = float(self.config.video_guidance_scale)
        cfg.action_guidance_scale = float(self.config.action_guidance_scale)
        cfg.num_inference_steps = int(self.config.video_inference_steps or 5)
        cfg.video_exec_step = int(
            self.config.video_exec_step if self.config.video_exec_step is not None else -1
        )
        cfg.action_num_inference_steps = int(self.config.action_inference_steps or 10)
        cfg.used_action_channel_ids = list(self.config.used_action_channel_ids)
        inverse = self._inverse_used_action_channel_ids()
        cfg.inverse_used_action_channel_ids = inverse
        cfg.action_norm_method = "quantiles"
        cfg.norm_stat = {
            "q01": self._model_action_stats["q01"].tolist(),
            "q99": self._model_action_stats["q99"].tolist(),
        }
        cfg.param_dtype = _dtype_from_config(self.config.dtype)
        cfg.save_root = str(self.config.save_root)
        cfg.enable_offload = False
        cfg.rank = 0
        cfg.local_rank = 0
        cfg.world_size = 1
        return cfg

    def _ensure_local_server(self):
        if self._local_server is not None:
            return self._local_server
        self._install_lingbo_import_path()
        try:
            from wan_va.wan_va_server import VA_Server
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Local LingBoVA inference needs LingBot-VA dependencies.") from exc
        server = VA_Server(self._make_va_job_config())
        object.__setattr__(self, "_local_server", server)
        return server

    def run_lingbo_va_validation(
        self,
        step: int,
        output_dir: Path,
        dataset_repo_id: str,
        dataset_root: Any,
        dataset_revision: Any,
        run_open_loop: bool,
    ) -> dict[str, Any]:
        """Entry point for `lerobot_train.py`'s duck-typed validation hook (see the `hasattr`
        check there — this method existing at all is what turns the hook on for this policy).

        Dispatches to `eval_utils`'s fixed-timestep / teacher-forced / open-loop routines based
        on `step` and `self.config.val_freq`/`open_loop_freq`, writes plots/CSV/markdown, and
        returns a flat dict of scalars for the caller to log to wandb (mode="eval").
        """
        if not self.config.val_episodes:
            return {}
        from . import eval_utils

        output_dir = Path(output_dir)
        results: dict[str, Any] = {}

        is_val_step = self.config.val_freq > 0 and step % self.config.val_freq == 0
        is_open_loop_step = run_open_loop or (
            self.config.open_loop_freq > 0 and step % self.config.open_loop_freq == 0
        )
        if not is_val_step and not is_open_loop_step:
            return {}

        was_training = self.training
        if is_val_step:
            val_metrics = eval_utils.run_fixed_timestep_validation(
                self,
                dataset_repo_id,
                dataset_root,
                dataset_revision,
                self.config.val_episodes,
                timestep_fractions=tuple(self.config.val_timestep_fractions),
                num_samples=self.config.val_num_samples,
                seed=self.config.val_seed,
            )
            results.update(val_metrics)

            tf_metrics = eval_utils.run_teacher_forced_eval(
                self,
                dataset_repo_id,
                dataset_root,
                dataset_revision,
                episode_id=self.config.teacher_forced_episode,
                offsets=list(self.config.teacher_forced_offsets),
                output_dir=output_dir / "eval_curves",
                step=step,
            )
            if tf_metrics:
                results["teacher_forced/total_mae"] = tf_metrics["teacher_forced/total_mae"]
                results["teacher_forced/direction_consistency"] = tf_metrics[
                    "teacher_forced/direction_consistency"
                ]
                results["teacher_forced/min_std_ratio"] = tf_metrics["teacher_forced/min_std_ratio"]
                for name, mae in tf_metrics["teacher_forced/per_joint_mae"].items():
                    results[f"teacher_forced/mae_{name}"] = mae
                if self._mean_collapse_tracker is None:
                    object.__setattr__(self, "_mean_collapse_tracker", eval_utils.TrainMetricsEMA())
                self._mean_collapse_tracker.check_mean_collapse(tf_metrics["teacher_forced/min_std_ratio"])

        if is_open_loop_step:
            for episode_id in self.config.open_loop_episodes:
                ol_metrics = eval_utils.run_open_loop_episode_eval(
                    self,
                    dataset_repo_id,
                    dataset_root,
                    dataset_revision,
                    episode_id=episode_id,
                    stride=self.config.open_loop_stride,
                    output_dir=output_dir / "eval_curves",
                    step=step,
                )
                if not ol_metrics:
                    continue
                results[f"open_loop/total_mae_ep{episode_id}"] = ol_metrics["open_loop/total_mae"]
                results[f"open_loop/direction_consistency_ep{episode_id}"] = ol_metrics[
                    "open_loop/direction_consistency"
                ]
                for name, mae in ol_metrics["open_loop/per_joint_mae"].items():
                    results[f"open_loop/mae_ep{episode_id}_{name}"] = mae

        if was_training:
            self.train()

        eval_utils.write_markdown_row(
            output_dir / "validation_summary.md",
            {"step": step, **{k: round(v, 6) if isinstance(v, float) else v for k, v in results.items()}},
        )
        return results


def _pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "dtype": str(obj.dtype),
            "shape": obj.shape,
            "data": obj.tobytes(),
        }
    if isinstance(obj, dict):
        return {key: _pack_array(value) for key, value in obj.items()}
    if isinstance(obj, list | tuple):
        return [_pack_array(value) for value in obj]
    return obj


def _unpack_array(obj):
    if "__ndarray__" in obj:
        return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"])
    return obj
