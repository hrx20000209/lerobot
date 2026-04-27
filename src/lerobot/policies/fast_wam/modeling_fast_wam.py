#!/usr/bin/env python

from __future__ import annotations

import json
import os
import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.configs import PreTrainedConfig
from lerobot.utils.constants import ACTION

from ..pretrained import PreTrainedPolicy
from .configuration_fast_wam import FastWAMConfig


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _mixed_precision_to_dtype(mixed_precision: str) -> torch.dtype:
    if mixed_precision == "no":
        return torch.float32
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported mixed_precision: {mixed_precision}")


class FastWAMPolicy(PreTrainedPolicy):
    config_class = FastWAMConfig
    name = "fast_wam"

    def __init__(self, config: FastWAMConfig, **kwargs):
        super().__init__(config)
        dataset_stats = kwargs.pop("dataset_stats", None)
        del kwargs
        config.validate_features()
        self.config = config
        self._action_queue: deque[Tensor] = deque(maxlen=config.n_action_steps)
        self._dataset_stats = dataset_stats
        self._normalization_stats = self._convert_lerobot_stats(dataset_stats)

        # Load the external FastWAM runtime lazily. During LeRobot training the
        # policy explicitly exposes its parameters, but the model stays outside
        # nn.Module registration so Accelerate does not collapse FastWAM's own
        # device placement back onto one GPU.
        object.__setattr__(self, "_fastwam_model", None)
        object.__setattr__(self, "_fastwam_processor", None)
        object.__setattr__(self, "_fastwam_cfg", None)
        object.__setattr__(self, "_fastwam_root", None)
        object.__setattr__(self, "_external_text_encoder", None)
        object.__setattr__(self, "_external_tokenizer", None)
        object.__setattr__(self, "_external_text_encoder_device", None)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: FastWAMConfig | None = None,
        **kwargs,
    ) -> "FastWAMPolicy":
        path = Path(pretrained_name_or_path).expanduser()
        if path.is_file() and path.suffix in {".pt", ".pth", ".ckpt"}:
            if config is None:
                config = FastWAMConfig()
            config.fastwam_checkpoint_path = path
            policy = cls(config, **kwargs)
            policy.eval()
            return policy
        if path.is_dir():
            if config is None:
                loaded = PreTrainedConfig.from_pretrained(path, **kwargs)
                if not isinstance(loaded, FastWAMConfig):
                    raise TypeError(f"Expected FastWAMConfig in {path}, got {type(loaded)}.")
                config = loaded
            checkpoint = path / "fastwam_model.pt"
            if checkpoint.exists():
                config.fastwam_checkpoint_path = checkpoint
            stats = path / "fastwam_dataset_stats.json"
            if stats.exists():
                config.dataset_stats_path = stats
            policy = cls(config, **kwargs)
            policy.eval()
            return policy
        return super().from_pretrained(pretrained_name_or_path, config=config, **kwargs)

    def _save_pretrained(self, save_directory: Path) -> None:
        model = getattr(self, "_fastwam_model", None)
        stats = getattr(self, "_normalization_stats", None)

        old_checkpoint_path = self.config.fastwam_checkpoint_path
        old_stats_path = self.config.dataset_stats_path
        try:
            if model is not None:
                self.config.fastwam_checkpoint_path = Path("fastwam_model.pt")
            if stats is not None:
                self.config.dataset_stats_path = Path("fastwam_dataset_stats.json")
            self.config._save_pretrained(save_directory)
        finally:
            self.config.fastwam_checkpoint_path = old_checkpoint_path
            self.config.dataset_stats_path = old_stats_path

        if model is not None:
            model.save_checkpoint(save_directory / "fastwam_model.pt")
        if stats is not None:
            self._save_fastwam_stats_json(stats, save_directory / "fastwam_dataset_stats.json")

    def train(self, mode: bool = True):
        super().train(mode)
        model = getattr(self, "_fastwam_model", None)
        if model is not None:
            if mode:
                self._apply_training_freeze()
            else:
                model.eval()
        return self

    def parameters(self, recurse: bool = True):
        model = getattr(self, "_fastwam_model", None)
        if model is not None:
            yield from model.parameters(recurse=recurse)
        yield from super().parameters(recurse=recurse)

    def named_parameters(
        self,
        prefix: str = "",
        recurse: bool = True,
        remove_duplicate: bool = True,
    ):
        model = getattr(self, "_fastwam_model", None)
        if model is not None:
            model_prefix = f"{prefix}._fastwam_model" if prefix else "_fastwam_model"
            yield from model.named_parameters(
                prefix=model_prefix,
                recurse=recurse,
                remove_duplicate=remove_duplicate,
            )
        yield from super().named_parameters(
            prefix=prefix,
            recurse=recurse,
            remove_duplicate=remove_duplicate,
        )

    def get_optim_params(self) -> list:
        self._ensure_fastwam_runtime()
        self._apply_training_freeze()
        return [param for param in self._fastwam_model.parameters() if param.requires_grad]

    def reset(self) -> None:
        self._action_queue.clear()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        self._ensure_fastwam_runtime()
        if self.training:
            self._apply_training_freeze()
        sample = self._build_training_sample(batch)
        loss, loss_dict = self._fastwam_model.training_loss(sample, tiled=self.config.tiled)
        return loss, loss_dict

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
        self._ensure_fastwam_runtime()

        image_batch = self._build_image_batch(batch)
        state_batch = self._build_state_batch(batch, batch_size=image_batch.shape[0])
        prompts = self._build_prompts(batch, batch_size=image_batch.shape[0])

        chunks = []
        for idx, prompt in enumerate(prompts):
            chunk = self._infer_one(
                image=image_batch[idx],
                state=state_batch[idx],
                prompt=prompt,
            )
            chunks.append(chunk)
        return torch.stack(chunks, dim=0)

    def _ensure_fastwam_runtime(self) -> None:
        if self._fastwam_model is not None:
            return

        root = self._resolve_fastwam_root()
        self._install_fastwam_import_path(root)
        self._set_diffsynth_model_base_path(root)

        try:
            from hydra import compose, initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra
            from hydra.utils import instantiate
            from omegaconf import OmegaConf
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "FastWAMPolicy needs FastWAM runtime dependencies at train/inference time. "
                "Install/use an environment with FastWAM plus hydra-core and omegaconf."
            ) from exc

        config_dir = root / "configs"
        if not config_dir.exists():
            raise FileNotFoundError(f"FastWAM configs directory not found: {config_dir}")

        overrides = []
        if not _is_none_like(self.config.fastwam_task):
            overrides.append(f"task={self.config.fastwam_task}")

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
            cfg = compose(config_name=self.config.fastwam_config_name, overrides=overrides)

        self._override_fastwam_shape_config(cfg)
        cfg.model.mot_checkpoint_mixed_attn = bool(self.config.gradient_checkpointing)
        use_external_text_encoder = bool(self.config.load_text_encoder) and not _is_none_like(
            self.config.text_encoder_device
        )
        cfg.model.load_text_encoder = bool(self.config.load_text_encoder and not use_external_text_encoder)
        cfg.model.skip_dit_load_from_pretrain = bool(self.config.skip_dit_load_from_pretrain)
        if self.config.action_dit_pretrained_path is not None:
            cfg.model.action_dit_pretrained_path = str(self.config.action_dit_pretrained_path)
        elif bool(self.config.skip_dit_load_from_pretrain):
            cfg.model.action_dit_pretrained_path = None

        checkpoint_path = self._resolve_checkpoint_path(root)
        dataset_stats = self._resolve_fastwam_dataset_stats(root)
        model_dtype = _mixed_precision_to_dtype(self.config.mixed_precision)

        instantiate_kwargs: dict[str, Any] = {
            "model_dtype": model_dtype,
            "device": self.config.device,
        }
        if not _is_none_like(self.config.text_encoder_device) and not use_external_text_encoder:
            instantiate_kwargs["text_encoder_device"] = self.config.text_encoder_device
        if not _is_none_like(self.config.vae_device):
            instantiate_kwargs["vae_device"] = self.config.vae_device
        if not _is_none_like(self.config.mot_layer_devices):
            instantiate_kwargs["mot_layer_devices"] = self.config.mot_layer_devices

        model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.model, resolve=True))
        processor_cfg = OmegaConf.create(OmegaConf.to_container(cfg.data.train.processor, resolve=True))

        model = instantiate(model_cfg, **instantiate_kwargs)
        self._set_runtime_gradient_checkpointing(model)
        if checkpoint_path is not None:
            self._load_fastwam_checkpoint(model, checkpoint_path)
        if self.training:
            model.train()
        else:
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)

        external_text_encoder = None
        external_tokenizer = None
        external_text_encoder_device = None
        if use_external_text_encoder:
            from fastwam.models.wan22.helpers.loader import _load_registered_model
            from fastwam.models.wan22.wan_video_text_encoder import HuggingfaceTokenizer

            text_encoder_path, tokenizer_path = self._resolve_text_assets_paths(root)
            external_text_encoder_device = str(self.config.text_encoder_device)
            external_text_encoder = _load_registered_model(
                str(text_encoder_path),
                "wan_video_text_encoder",
                torch_dtype=model_dtype,
                device=external_text_encoder_device,
            ).eval()
            external_tokenizer = HuggingfaceTokenizer(
                name=str(tokenizer_path),
                seq_len=int(model_cfg.get("tokenizer_max_len", 128)),
                clean="whitespace",
            )

        processor = instantiate(processor_cfg).eval()
        processor.set_normalizer_from_stats(dataset_stats)

        object.__setattr__(self, "_fastwam_model", model)
        object.__setattr__(self, "_fastwam_processor", processor)
        object.__setattr__(self, "_fastwam_cfg", cfg)
        object.__setattr__(self, "_fastwam_root", root)
        object.__setattr__(self, "_external_text_encoder", external_text_encoder)
        object.__setattr__(self, "_external_tokenizer", external_tokenizer)
        object.__setattr__(self, "_external_text_encoder_device", external_text_encoder_device)

    def _override_fastwam_shape_config(self, cfg: Any) -> None:
        processor_cfg = cfg.data.train.processor
        processor_cfg.action_output_dim = int(self.config.action_dim)
        processor_cfg.proprio_output_dim = int(self.config.proprio_dim)
        processor_cfg.norm_default_mode = self.config.norm_default_mode

        shape_meta = cfg.data.train.shape_meta
        for meta in shape_meta.action:
            meta.raw_shape = int(self.config.action_dim)
            meta.shape = int(self.config.action_dim)
        for meta in shape_meta.state:
            meta.raw_shape = int(self.config.proprio_dim)
            meta.shape = int(self.config.proprio_dim)
        processor_cfg.shape_meta = shape_meta

    def _resolve_fastwam_root(self) -> Path:
        candidates = [
            self.config.fastwam_root,
            os.environ.get("FASTWAM_ROOT"),
            Path("/home/rxhuang/Projects/FastWAM"),
        ]
        for candidate in candidates:
            if _is_none_like(candidate):
                continue
            path = Path(candidate).expanduser().resolve()
            if (path / "src" / "fastwam").exists() and (path / "configs").exists():
                return path
        raise FileNotFoundError(
            "Could not find a FastWAM checkout. Set policy.fastwam_root or FASTWAM_ROOT."
        )

    def _install_fastwam_import_path(self, root: Path) -> None:
        for path in (root, root / "src"):
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

    def _set_diffsynth_model_base_path(self, root: Path) -> None:
        if self.config.diffsynth_model_base_path is not None:
            os.environ.setdefault(
                "DIFFSYNTH_MODEL_BASE_PATH",
                str(Path(self.config.diffsynth_model_base_path).expanduser().resolve()),
            )
            return

        nested = root / "checkpoints" / "checkpoints"
        direct = root / "checkpoints"
        if nested.exists():
            os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(nested))
        elif direct.exists():
            os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", str(direct))

    def _resolve_checkpoint_path(self, root: Path) -> Path | None:
        if self.config.fastwam_checkpoint_path is not None:
            path = Path(self.config.fastwam_checkpoint_path).expanduser().resolve()
            if path.exists():
                return path
            raise FileNotFoundError(f"FastWAM checkpoint not found: {path}")

        if self.config.auto_find_fastwam_artifacts:
            path = root / "checkpoints" / "fastwam_release" / f"{self.config.release_checkpoint_name}.pt"
            if path.exists():
                return path

        return None

    def _resolve_dataset_stats_path(self, root: Path) -> Path:
        if self.config.dataset_stats_path is not None:
            path = Path(self.config.dataset_stats_path).expanduser().resolve()
            if path.exists():
                return path
            raise FileNotFoundError(f"FastWAM dataset stats not found: {path}")

        if self.config.auto_find_fastwam_artifacts:
            path = (
                root
                / "checkpoints"
                / "fastwam_release"
                / f"{self.config.release_checkpoint_name}_dataset_stats.json"
            )
            if path.exists():
                return path

        raise FileNotFoundError(
            "FastWAM dataset stats path is required. Set policy.dataset_stats_path."
        )

    def _resolve_fastwam_dataset_stats(self, root: Path) -> dict[str, Any]:
        if self._normalization_stats is not None:
            return self._normalization_stats

        try:
            from fastwam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("FastWAM normalizer utilities are required.") from exc

        dataset_stats_path = self._resolve_dataset_stats_path(root)
        stats = load_dataset_stats_from_json(str(dataset_stats_path))
        object.__setattr__(self, "_normalization_stats", stats)
        return stats

    def _resolve_text_assets_paths(self, root: Path) -> tuple[Path, Path]:
        text_rel = Path(
            "DiffSynth-Studio/Wan-Series-Converted-Safetensors/"
            "models_t5_umt5-xxl-enc-bf16.safetensors"
        )
        legacy_text_rel = Path("Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth")
        tokenizer_rel = Path("Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl")

        bases = [root / "checkpoints" / "checkpoints", root / "checkpoints"]
        text_candidates = [base / text_rel for base in bases] + [base / legacy_text_rel for base in bases]
        tokenizer_candidates = [base / tokenizer_rel for base in bases]

        text_path = next((path for path in text_candidates if path.exists()), None)
        tokenizer_path = next((path for path in tokenizer_candidates if path.exists()), None)
        if text_path is None or tokenizer_path is None:
            looked = "\n".join(
                ["Text encoder candidates:"]
                + [f"  - {path}" for path in text_candidates]
                + ["Tokenizer candidates:"]
                + [f"  - {path}" for path in tokenizer_candidates]
            )
            raise FileNotFoundError(f"Failed to locate FastWAM text assets.\n{looked}")
        return text_path, tokenizer_path

    def _set_runtime_gradient_checkpointing(self, model: Any) -> None:
        enabled = bool(self.config.gradient_checkpointing)
        mot = getattr(model, "mot", None)
        if mot is not None and hasattr(mot, "mot_checkpoint_mixed_attn"):
            mot.mot_checkpoint_mixed_attn = enabled
        for name in ("video_expert", "action_expert"):
            expert = getattr(model, name, None)
            if expert is not None and hasattr(expert, "use_gradient_checkpointing"):
                expert.use_gradient_checkpointing = enabled

    def _load_fastwam_checkpoint(self, model: Any, checkpoint_path: Path) -> None:
        if not self.config.ignore_mismatched_checkpoint_shapes:
            model.load_checkpoint(str(checkpoint_path))
            return

        payload = torch.load(checkpoint_path, map_location="cpu")
        if "mot" in payload:
            self._load_matching_state_dict(model.mot, payload["mot"])
        elif "dit" in payload:
            self._load_matching_state_dict(model.video_expert, payload["dit"])
        else:
            raise ValueError(f"Checkpoint missing both `mot` and `dit` keys: {checkpoint_path}")

        proprio_encoder = getattr(model, "proprio_encoder", None)
        if proprio_encoder is not None and "proprio_encoder" in payload:
            self._load_matching_state_dict(proprio_encoder, payload["proprio_encoder"])

    def _load_matching_state_dict(self, module: torch.nn.Module, state_dict: dict[str, Tensor]) -> None:
        current = module.state_dict()
        matched = {
            key: value
            for key, value in state_dict.items()
            if key in current and tuple(current[key].shape) == tuple(value.shape)
        }
        module.load_state_dict(matched, strict=False)

    def _apply_training_freeze(self) -> None:
        model = getattr(self, "_fastwam_model", None)
        if model is None:
            return

        if self.config.train_expert_only:
            model.eval()
            model.requires_grad_(False)
            action_expert = getattr(model, "action_expert", None)
            if action_expert is not None:
                action_expert.train()
                action_expert.requires_grad_(True)
            proprio_encoder = getattr(model, "proprio_encoder", None)
            if proprio_encoder is not None:
                proprio_encoder.train()
                proprio_encoder.requires_grad_(True)
            return

        if self.config.freeze_vision_encoder:
            model.eval()
            model.requires_grad_(False)
            if hasattr(model, "dit"):
                model.dit.train()
                model.dit.requires_grad_(True)
            video_expert = getattr(model, "video_expert", None)
            if video_expert is not None:
                video_expert.eval()
                video_expert.requires_grad_(False)
            proprio_encoder = getattr(model, "proprio_encoder", None)
            if proprio_encoder is not None:
                proprio_encoder.train()
                proprio_encoder.requires_grad_(True)
            return

        model.train()
        model.requires_grad_(True)
        for name in ("vae", "text_encoder"):
            module = getattr(model, name, None)
            if module is not None:
                module.eval()
                module.requires_grad_(False)

    def _convert_lerobot_stats(
        self, dataset_stats: dict[str, dict[str, Any]] | None
    ) -> dict[str, Any] | None:
        if dataset_stats is None:
            return None

        if "action" in dataset_stats and "state" in dataset_stats:
            return self._tensorize_stats(dataset_stats)

        if ACTION not in dataset_stats or self.config.state_key not in dataset_stats:
            return None

        return {
            "action": {"default": self._convert_one_lerobot_stat(dataset_stats[ACTION])},
            "state": {"default": self._convert_one_lerobot_stat(dataset_stats[self.config.state_key])},
            "num_episodes": self._stats_scalar(dataset_stats.get("episode_index", {}).get("count")),
            "num_transition": self._stats_scalar(dataset_stats.get(ACTION, {}).get("count")),
        }

    def _convert_one_lerobot_stat(self, stats: dict[str, Any]) -> dict[str, Tensor]:
        converted: dict[str, Tensor] = {}
        for key in ("min", "max", "mean", "std", "q01", "q99"):
            if key not in stats:
                continue
            tensor = self._to_float_tensor(stats[key])
            converted[f"global_{key}"] = tensor
            converted[f"stepwise_{key}"] = tensor.unsqueeze(0).expand(self.config.chunk_size, -1).clone()
        return converted

    def _tensorize_stats(self, stats: dict[str, Any]) -> dict[str, Any]:
        def convert(value: Any) -> Any:
            if isinstance(value, Tensor):
                return value.to(dtype=torch.float32)
            if isinstance(value, np.ndarray):
                return torch.from_numpy(value).to(dtype=torch.float32)
            if isinstance(value, dict):
                return {key: convert(val) for key, val in value.items()}
            if isinstance(value, list):
                return [convert(val) for val in value]
            return value

        return convert(stats)

    def _stats_scalar(self, value: Any) -> int | None:
        if value is None:
            return None
        tensor = self._to_float_tensor(value).reshape(-1)
        if tensor.numel() == 0:
            return None
        return int(tensor[0].item())

    def _to_float_tensor(self, value: Any) -> Tensor:
        if isinstance(value, Tensor):
            return value.detach().cpu().to(dtype=torch.float32)
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).to(dtype=torch.float32)
        return torch.as_tensor(value, dtype=torch.float32)

    def _save_fastwam_stats_json(self, stats: dict[str, Any], path: Path) -> None:
        def convert(value: Any) -> Any:
            if isinstance(value, Tensor):
                return value.detach().cpu().tolist()
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, dict):
                return {key: convert(val) for key, val in value.items()}
            if isinstance(value, (list, tuple)):
                return [convert(val) for val in value]
            return value

        path.write_text(json.dumps(convert(stats), indent=2), encoding="utf-8")

    def _build_training_sample(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        video, image_is_pad = self._build_training_video(batch)
        action = self._as_action_sequence(batch[ACTION])
        proprio = self._as_state_sequence(batch[self.config.state_key])[:, : self.config.chunk_size]
        action, proprio = self._normalize_action_and_state(action, proprio)
        prompts = self._build_prompts(batch, batch_size=video.shape[0])
        context, context_mask = self._encode_prompts(prompts)

        sample: dict[str, Tensor] = {
            "video": video,
            "action": action.to(device=self._fastwam_model.device, dtype=self._fastwam_model.torch_dtype),
            "proprio": proprio.to(device=self._fastwam_model.device, dtype=self._fastwam_model.torch_dtype),
            "context": context,
            "context_mask": context_mask,
        }

        if ACTION + "_is_pad" in batch:
            sample["action_is_pad"] = torch.as_tensor(
                batch[ACTION + "_is_pad"], dtype=torch.bool, device=self._fastwam_model.device
            )
        if image_is_pad is not None:
            sample["image_is_pad"] = image_is_pad.to(device=self._fastwam_model.device, dtype=torch.bool)
        return sample

    def _build_training_video(self, batch: dict[str, Any]) -> tuple[Tensor, Tensor | None]:
        camera_keys = self._resolve_training_camera_keys(batch)
        frame_indices = list(range(0, self.config.num_frames, self.config.action_video_freq_ratio))

        videos = []
        masks = []
        for key in camera_keys:
            seq = self._as_batched_image_sequence(batch[key])
            seq = seq[:, frame_indices]
            videos.append(seq)
            mask_key = key + "_is_pad"
            if mask_key in batch:
                masks.append(torch.as_tensor(batch[mask_key], dtype=torch.bool)[:, frame_indices])

        composed = self._compose_camera_sequences(videos)
        video = self._resize_image_sequence(composed, self.config.image_size)
        video = video.permute(0, 2, 1, 3, 4).contiguous()
        video = video.to(device=self._fastwam_model.device, dtype=self._fastwam_model.torch_dtype)

        image_is_pad = None
        if masks:
            image_is_pad = torch.stack(masks, dim=0).any(dim=0)
        return video, image_is_pad

    def _resolve_training_camera_keys(self, batch: dict[str, Any]) -> list[str]:
        if self.config.composite_image_key and self.config.composite_image_key in batch:
            return [self.config.composite_image_key]
        if self.config.camera_keys:
            missing = [key for key in self.config.camera_keys if key not in batch]
            if missing:
                raise KeyError(f"Configured FastWAM camera_keys missing from batch: {missing}")
            return list(self.config.camera_keys)

        robotwin_keys = [
            self.config.high_camera_key,
            self.config.left_camera_key,
            self.config.right_camera_key,
        ]
        if all(key in batch for key in robotwin_keys):
            return robotwin_keys

        visual_keys = [key for key in self.config.image_features if key in batch]
        if not visual_keys:
            raise KeyError("FastWAMPolicy could not find any visual observation keys in the batch.")
        return visual_keys

    def _as_batched_image_sequence(self, value: Any) -> Tensor:
        image = torch.as_tensor(value)
        if image.ndim == 4:
            image = image.unsqueeze(0)
        if image.ndim != 5:
            raise ValueError(
                f"Expected image sequence [B,T,C,H,W] or [B,T,H,W,C], got {tuple(image.shape)}"
            )
        if image.shape[-1] in {1, 3, 4} and image.shape[2] not in {1, 3, 4}:
            image = image.permute(0, 1, 4, 2, 3)
        if image.shape[2] > 3:
            image = image[:, :, :3]
        if image.shape[2] != 3:
            raise ValueError(f"FastWAM expects RGB image sequences, got shape {tuple(image.shape)}")
        image = image.to(dtype=torch.float32)
        if float(image.detach().amax().cpu()) > 2.0:
            image = image / 255.0
        if float(image.detach().amin().cpu()) >= 0.0:
            image = image * 2.0 - 1.0
        if image.shape[1] < self.config.num_frames:
            raise ValueError(
                f"FastWAM needs {self.config.num_frames} observation frames, got {image.shape[1]}."
            )
        return image[:, : self.config.num_frames]

    def _compose_camera_sequences(self, videos: list[Tensor]) -> Tensor:
        if len(videos) == 1:
            return videos[0]

        if self.config.concat_multi_camera == "robotwin" and len(videos) >= 3:
            high = self._resize_image_sequence(videos[0], (256, 320))
            left = self._resize_image_sequence(videos[1], (128, 160))
            right = self._resize_image_sequence(videos[2], (128, 160))
            bottom = torch.cat([left, right], dim=-1)
            return torch.cat([high, bottom], dim=-2)

        resized = [self._resize_image_sequence(video, self.config.image_size) for video in videos]
        if self.config.concat_multi_camera == "vertical":
            return torch.cat(resized, dim=-2)
        return torch.cat(resized, dim=-1)

    def _resize_image_sequence(self, video: Tensor, size_hw: tuple[int, int]) -> Tensor:
        batch_size, num_frames, channels, height, width = video.shape
        flat = video.reshape(batch_size * num_frames, channels, height, width)
        resized = self._resize_image(flat, size_hw)
        return resized.reshape(batch_size, num_frames, channels, *size_hw)

    def _as_action_sequence(self, value: Any) -> Tensor:
        action = torch.as_tensor(value, dtype=torch.float32)
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action sequence [B,T,D], got {tuple(action.shape)}")
        if action.shape[1] < self.config.chunk_size:
            raise ValueError(f"FastWAM needs {self.config.chunk_size} actions, got {action.shape[1]}.")
        if action.shape[-1] != self.config.action_dim:
            raise ValueError(f"Expected action_dim={self.config.action_dim}, got {action.shape[-1]}.")
        return action[:, : self.config.chunk_size]

    def _as_state_sequence(self, value: Any) -> Tensor:
        state = torch.as_tensor(value, dtype=torch.float32)
        if state.ndim == 2:
            state = state.unsqueeze(0)
        if state.ndim != 3:
            raise ValueError(f"Expected state sequence [B,T,D], got {tuple(state.shape)}")
        if state.shape[1] < self.config.chunk_size:
            raise ValueError(f"FastWAM needs at least {self.config.chunk_size} states, got {state.shape[1]}.")
        if state.shape[-1] != self.config.proprio_dim:
            raise ValueError(f"Expected proprio_dim={self.config.proprio_dim}, got {state.shape[-1]}.")
        return state

    def _normalize_action_and_state(self, action: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        stats = self._normalization_stats
        if stats is None:
            raise ValueError("FastWAM training needs dataset statistics for action/state normalization.")

        action_stats = stats["action"]["default"]
        state_stats = stats["state"]["default"]
        return (
            self._linear_normalize(action, action_stats, self.config.norm_default_mode),
            self._linear_normalize(state, state_stats, self.config.norm_default_mode),
        )

    def _linear_normalize(self, value: Tensor, stats: dict[str, Tensor], mode: str) -> Tensor:
        stats = {key: self._to_float_tensor(val).to(value.device) for key, val in stats.items()}

        def pick(name: str) -> Tensor:
            return stats.get(name) if name in stats else stats[f"global_{name}"]

        if mode == "z-score":
            normalized = (value - pick("mean")) / (pick("std") + 1e-8)
        else:
            if mode == "min/max":
                low, high = pick("min"), pick("max")
            elif mode == "q01/q99":
                low, high = pick("q01"), pick("q99")
            else:
                raise ValueError(f"Unsupported normalization mode: {mode}")
            scale = high - low
            ignore = scale < 1e-4
            scale = torch.where(ignore, torch.full_like(scale, 2.0), scale)
            normalized = -1.0 + 2.0 * (value - low) / scale
            normalized = torch.where(ignore, value - low, normalized)
        return torch.clamp(normalized, -5.0, 5.0)

    def _encode_prompts(self, prompts: list[str]) -> tuple[Tensor, Tensor]:
        model = self._fastwam_model
        with torch.no_grad():
            if self._external_text_encoder is not None and self._external_tokenizer is not None:
                ids, mask = self._external_tokenizer(prompts, return_mask=True, add_special_tokens=True)
                ids = ids.to(device=self._external_text_encoder_device)
                mask = mask.to(device=self._external_text_encoder_device, dtype=torch.bool)
                context = self._external_text_encoder(ids, mask)
                for idx, valid in enumerate(mask.sum(dim=1).tolist()):
                    context[idx, int(valid) :] = 0
                context_mask = torch.ones_like(mask, dtype=torch.bool)
                return (
                    context.to(device=model.device, dtype=model.torch_dtype, non_blocking=True),
                    context_mask.to(device=model.device, dtype=torch.bool, non_blocking=True),
                )
            context, context_mask = model.encode_prompt(prompts)
        return context, context_mask

    def _build_image_batch(self, batch: dict[str, Any]) -> Tensor:
        if self.config.composite_image_key and self.config.composite_image_key in batch:
            image = self._as_batched_image(batch[self.config.composite_image_key])
            return self._resize_image(image, self.config.image_size)

        camera_keys = self._resolve_inference_camera_keys(batch)
        images = [self._as_batched_image(batch[key]) for key in camera_keys]
        return self._compose_image_batch(images)

    def _resolve_inference_camera_keys(self, batch: dict[str, Any]) -> list[str]:
        if self.config.camera_keys:
            missing = [key for key in self.config.camera_keys if key not in batch]
            if missing:
                raise KeyError(f"Configured FastWAM camera_keys missing from batch: {missing}")
            return list(self.config.camera_keys)

        robotwin_keys = [
            self.config.high_camera_key,
            self.config.left_camera_key,
            self.config.right_camera_key,
        ]
        if all(key in batch for key in robotwin_keys):
            return robotwin_keys

        visual_keys = [key for key in self.config.image_features if key in batch]
        if visual_keys:
            return visual_keys

        raise KeyError("FastWAMPolicy needs image observations, but no configured visual keys were found.")

    def _compose_image_batch(self, images: list[Tensor]) -> Tensor:
        if len(images) == 1:
            return self._resize_image(images[0], self.config.image_size)
        if self.config.concat_multi_camera == "robotwin" and len(images) >= 3:
            high = self._resize_image(images[0], (256, 320))
            left = self._resize_image(images[1], (128, 160))
            right = self._resize_image(images[2], (128, 160))
            bottom = torch.cat([left, right], dim=-1)
            return torch.cat([high, bottom], dim=-2)

        resized = [self._resize_image(image, self.config.image_size) for image in images]
        if self.config.concat_multi_camera == "vertical":
            composed = torch.cat(resized, dim=-2)
        else:
            composed = torch.cat(resized, dim=-1)
        return self._resize_image(composed, self.config.image_size)

    def _as_batched_image(self, value: Any) -> Tensor:
        image = torch.as_tensor(value)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError(f"Expected image tensor [C,H,W] or [B,C,H,W], got {tuple(image.shape)}")
        if image.shape[-1] in {1, 3, 4} and image.shape[1] not in {1, 3, 4}:
            image = image.permute(0, 3, 1, 2)
        if image.shape[1] > 3:
            image = image[:, :3]
        if image.shape[1] != 3:
            raise ValueError(f"FastWAM expects RGB images, got shape {tuple(image.shape)}")
        image = image.to(dtype=torch.float32)
        if float(image.detach().amax().cpu()) > 2.0:
            image = image / 255.0
        if float(image.detach().amin().cpu()) >= 0.0:
            image = image * 2.0 - 1.0
        return image

    def _resize_image(self, image: Tensor, size_hw: tuple[int, int]) -> Tensor:
        return F.interpolate(image, size=size_hw, mode="bilinear", align_corners=False)

    def _build_state_batch(self, batch: dict[str, Any], batch_size: int) -> Tensor:
        if self.config.state_key not in batch:
            raise KeyError(f"FastWAMPolicy expected state key '{self.config.state_key}'.")
        state = torch.as_tensor(batch[self.config.state_key], dtype=torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim != 2:
            raise ValueError(f"Expected state tensor [D] or [B,D], got {tuple(state.shape)}")
        if state.shape[0] == 1 and batch_size > 1:
            state = state.expand(batch_size, -1)
        if state.shape[0] != batch_size:
            raise ValueError(
                f"State batch size {state.shape[0]} does not match image batch size {batch_size}."
            )
        return state

    def _build_prompts(self, batch: dict[str, Any], batch_size: int) -> list[str]:
        tasks = batch.get(self.config.task_key, self.config.default_task)
        if isinstance(tasks, str):
            task_list = [tasks] * batch_size
        elif isinstance(tasks, (list, tuple)):
            task_list = list(tasks)
            if len(task_list) == 1 and batch_size > 1:
                task_list = task_list * batch_size
        else:
            task_list = [str(tasks)] * batch_size
        if len(task_list) != batch_size:
            raise ValueError(f"Expected {batch_size} task prompts, got {len(task_list)}.")
        return [self.config.prompt_template.format(task=str(task)) for task in task_list]

    def _normalize_state(self, state: Tensor) -> Tensor:
        processor = self._fastwam_processor
        state_meta = processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in FastWAM shape_meta['state'].")
        state_key = state_meta[0]["key"]
        state_batch = {"state": {state_key: state.to(dtype=torch.float32).unsqueeze(0).cpu()}}
        state_batch = processor.action_state_transform(state_batch)
        state_batch = processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key][0]

    def _denormalize_action(self, action: Tensor) -> Tensor:
        processor = self._fastwam_processor
        action_meta = processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in FastWAM shape_meta['action'].")
        action_key = action_meta[0]["key"]
        normalizer = processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm

    def _infer_one(self, image: Tensor, state: Tensor, prompt: str) -> Tensor:
        model = self._fastwam_model
        image = image.unsqueeze(0).to(device=model.device, dtype=model.torch_dtype, non_blocking=True)
        proprio = self._normalize_state(state).to(
            device=model.device,
            dtype=model.torch_dtype,
            non_blocking=True,
        )

        infer_kwargs = {
            "prompt": prompt,
            "input_image": image,
            "action_horizon": self.config.chunk_size,
            "proprio": proprio,
            "negative_prompt": self.config.negative_prompt,
            "text_cfg_scale": self.config.text_cfg_scale,
            "num_inference_steps": self.config.num_inference_steps,
            "sigma_shift": self.config.sigma_shift,
            "seed": self.config.seed,
            "rand_device": self.config.rand_device,
            "tiled": self.config.tiled,
            "collect_timing": self.config.collect_timing,
        }

        if self._external_text_encoder is not None and self._external_tokenizer is not None:
            ids, mask = self._external_tokenizer(prompt, return_mask=True, add_special_tokens=True)
            ids = ids.to(device=self._external_text_encoder_device)
            mask = mask.to(device=self._external_text_encoder_device, dtype=torch.bool)
            context = self._external_text_encoder(ids, mask)
            for idx, valid in enumerate(mask.sum(dim=1).tolist()):
                context[idx, int(valid) :] = 0
            context_mask = torch.ones_like(mask, dtype=torch.bool)
            infer_kwargs["prompt"] = None
            infer_kwargs["context"] = context.to(
                device=model.device,
                dtype=model.torch_dtype,
                non_blocking=True,
            )
            infer_kwargs["context_mask"] = context_mask.to(
                device=model.device,
                dtype=torch.bool,
                non_blocking=True,
            )

        pred = model.infer_action(
            **infer_kwargs,
        )
        return self._denormalize_action(pred["action"])
