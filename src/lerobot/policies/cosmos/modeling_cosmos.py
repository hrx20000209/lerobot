#!/usr/bin/env python

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors
from safetensors.torch import save_file as save_safetensors
from torch import Tensor

from lerobot.configs import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION

from .configuration_cosmos import CosmosConfig

LOG = logging.getLogger(__name__)

SO101_LATENT_INDICES = {
    "current_proprio_latent_idx": 1,
    "current_wrist_image_latent_idx": 2,
    "current_wrist_image2_latent_idx": 3,
    "current_image_latent_idx": 4,
    "action_latent_idx": 5,
    "future_proprio_latent_idx": 6,
    "future_wrist_image_latent_idx": 7,
    "future_wrist_image2_latent_idx": 8,
    "future_image_latent_idx": 9,
    "value_latent_idx": 10,
}


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().lower() in {"", "none", "null"}


def _to_numpy_stat(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _minmax_normalize(array: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    scale = np.maximum(upper - lower, 1e-6)
    return (2.0 * (array - lower) / scale - 1.0).astype(np.float32)


def _minmax_unnormalize(array: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return ((array + 1.0) * 0.5 * np.maximum(upper - lower, 1e-6) + lower).astype(np.float32)


def _ensure_batch_time_image(image: Tensor) -> Tensor:
    image = torch.as_tensor(image).detach().cpu()
    if image.ndim == 3:
        image = image.unsqueeze(0).unsqueeze(0)
    elif image.ndim == 4:
        if image.shape[1] in (1, 3, 4):
            image = image.unsqueeze(1)
        elif image.shape[-1] in (1, 3, 4):
            image = image.permute(0, 3, 1, 2).unsqueeze(1)
        else:
            raise ValueError(f"Cannot identify image channel dimension for shape={tuple(image.shape)}")
    elif image.ndim == 5:
        if image.shape[2] in (1, 3, 4):
            pass
        elif image.shape[-1] in (1, 3, 4):
            image = image.permute(0, 1, 4, 2, 3)
        else:
            raise ValueError(f"Cannot identify image channel dimension for shape={tuple(image.shape)}")
    else:
        raise ValueError(f"Expected image tensor with 3-5 dims, got shape={tuple(image.shape)}")
    return image[:, :, :3]


def _chw_to_hwc_uint8(image: Tensor) -> np.ndarray:
    image = torch.as_tensor(image).detach().cpu()
    if image.ndim != 3:
        raise ValueError(f"Expected CHW image, got shape={tuple(image.shape)}")
    if image.shape[0] not in (1, 3, 4):
        raise ValueError(f"Expected channel-first image, got shape={tuple(image.shape)}")
    image = image[:3].permute(1, 2, 0)
    if image.dtype.is_floating_point:
        if image.numel() and float(image.max()) <= 1.5:
            image = image * 255.0
        image = image.round().clamp(0, 255)
    return image.to(torch.uint8).numpy()


class CosmosPolicy(PreTrainedPolicy):
    config_class = CosmosConfig
    name = "cosmos"

    def __init__(self, config: CosmosConfig, **kwargs: Any) -> None:
        super().__init__(config)
        dataset_stats = kwargs.pop("dataset_stats", None)
        dataset_meta = kwargs.pop("dataset_meta", None)
        del kwargs

        config.validate_features()
        self.config = config
        self._dataset_stats_from_lerobot = dataset_stats
        self._dataset_meta = dataset_meta
        self._action_queue: deque[Tensor] = deque(maxlen=config.n_action_steps)
        self._runtime_loaded = False
        self._debug_logged_train = False
        self._debug_logged_infer = False

        self.cosmos_model = None
        self.cosmos_runtime_config = None
        self.dataset_stats: dict[str, Any] = {}
        self.t5_text_embeddings: dict[str, Tensor] = {}
        self.finetune_report = None
        self._last_forward_metrics: dict[str, float] = {}
        self._last_pred_action_chunk: Tensor | None = None
        self._freeze_applied = False
        self._trainable_logged = False

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config: CosmosConfig | None = None,
        **kwargs: Any,
    ) -> CosmosPolicy:
        path = Path(pretrained_name_or_path).expanduser()
        if path.is_dir():
            if config is None:
                loaded = PreTrainedConfig.from_pretrained(path, **kwargs)
                if not isinstance(loaded, CosmosConfig):
                    raise TypeError(f"Expected CosmosConfig in {path}, got {type(loaded)}")
                config = loaded
            config.pretrained_path = path
            if (path / "cosmos_policy_trainable.safetensors").is_file():
                config.fine_tuned_weights_path = "cosmos_policy_trainable.safetensors"
            policy = cls(config, **kwargs)
            policy.eval()
            return policy
        return super().from_pretrained(pretrained_name_or_path, config=config, **kwargs)

    def _save_pretrained(self, save_directory: Path) -> None:
        save_directory.mkdir(parents=True, exist_ok=True)
        old_weights_path = self.config.fine_tuned_weights_path
        try:
            if self.cosmos_model is not None:
                self.config.fine_tuned_weights_path = "cosmos_policy_trainable.safetensors"
            self.config._save_pretrained(save_directory)
            self._ensure_config_type(save_directory)
        finally:
            self.config.fine_tuned_weights_path = old_weights_path

        if self.cosmos_model is not None:
            state = {
                name: param.detach().cpu()
                for name, param in self.cosmos_model.named_parameters()
                if param.requires_grad
            }
            if not state:
                state = {name: tensor.detach().cpu() for name, tensor in self.cosmos_model.state_dict().items()}
            save_safetensors(state, save_directory / "cosmos_policy_trainable.safetensors")
        if self.dataset_stats:
            serializable = {
                key: value.tolist() if isinstance(value, np.ndarray) else value
                for key, value in self.dataset_stats.items()
            }
            (save_directory / "so101_dataset_statistics.json").write_text(
                json.dumps(serializable, indent=2) + "\n", encoding="utf-8"
            )

    def _ensure_config_type(self, save_directory: Path) -> None:
        config_path = save_directory / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["type"] = self.name
        config_path.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")

    def train(self, mode: bool = True):
        super().train(mode)
        if self.cosmos_model is not None:
            self.cosmos_model.train(mode)
            self._apply_freeze()
        return self

    def get_optim_params(self) -> list[Tensor]:
        self._ensure_runtime(for_training=True)
        self._apply_freeze()
        params = [p for p in self.cosmos_model.parameters() if p.requires_grad]
        if not params:
            raise RuntimeError("CosmosPolicy has no trainable parameters after applying train_mode.")
        return params

    def reset(self) -> None:
        self._action_queue.clear()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, float]]:
        self._ensure_runtime(for_training=True)
        self.cosmos_model.train()
        self._apply_freeze()
        cosmos_batch, debug = self._build_training_batch(batch)
        output_batch, loss = self.cosmos_model.training_step(cosmos_batch, iteration=0)
        if self.config.action_loss_weight != 1.0:
            loss = loss * float(self.config.action_loss_weight)

        action_loss = output_batch.get("demo_sample_action_l1_loss", loss.detach())
        visual_loss = output_batch.get("demo_sample_future_image_l1_loss", torch.zeros_like(loss.detach()))
        metrics = {
            "action_loss": float(torch.as_tensor(action_loss).detach().float().mean().item()),
            "visual_loss": float(torch.nan_to_num(torch.as_tensor(visual_loss).detach().float()).mean().item()),
            "raw_action_min": debug["raw_action_min"],
            "raw_action_max": debug["raw_action_max"],
            "normalized_action_min": debug["normalized_action_min"],
            "normalized_action_max": debug["normalized_action_max"],
        }
        if "model_pred" in output_batch:
            pred = output_batch["model_pred"].x0
            metrics["pred_action_min"] = float(pred.detach().float().amin().item())
            metrics["pred_action_max"] = float(pred.detach().float().amax().item())
        self._last_forward_metrics = metrics
        if not self._debug_logged_train:
            LOG.warning("Cosmos train debug: %s", json.dumps(debug, sort_keys=True))
            self._debug_logged_train = True
        return loss, metrics

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
        generator = kwargs.pop("generator", None)
        del generator
        if kwargs:
            raise TypeError(f"Unexpected Cosmos inference kwargs: {sorted(kwargs)}")
        self._ensure_runtime(for_training=False)
        self.cosmos_model.eval()

        state = self._current_state(batch)
        observations = self._build_inference_observations(batch, state)
        chunks = []
        for index, obs in enumerate(observations):
            task = self._task_for_batch_index(batch, index)
            result = self._cosmos_get_action(
                self._deploy_cfg(),
                self.cosmos_model,
                self.dataset_stats,
                obs,
                task,
                seed=self.config.seed + index,
                randomize_seed=False,
                num_denoising_steps_action=self.config.num_denoising_steps_action,
                generate_future_state_and_value_in_parallel=False,
            )
            chunk = torch.as_tensor(np.asarray(result["actions"], dtype=np.float32), dtype=torch.float32)
            chunks.append(chunk)
        actions = torch.stack(chunks, dim=0).to(device=state.device)
        unclipped = actions.clone()
        if self.config.safety_clip_actions:
            actions = self._clip_action_chunk(actions, state)
        self._last_pred_action_chunk = actions.detach().cpu()
        if not self._debug_logged_infer:
            LOG.warning(
                "Cosmos inference debug: state=%s raw_pred_range=[%.4f, %.4f] clipped_range=[%.4f, %.4f] shape=%s",
                state.detach().cpu().tolist(),
                float(unclipped.amin().item()),
                float(unclipped.amax().item()),
                float(actions.amin().item()),
                float(actions.amax().item()),
                tuple(actions.shape),
            )
            self._debug_logged_infer = True
        return actions

    @property
    def device(self) -> torch.device:
        return torch.device(self.config.device)

    def _install_cosmos_import_path(self) -> Path:
        root = Path(self.config.cosmos_repo).expanduser().resolve()
        if not (root / "cosmos_policy").exists():
            raise FileNotFoundError(
                f"Cosmos repo not found at {root}. Set policy.cosmos_repo=/home/rxhuang/Projects/cosmos-policy "
                "and PYTHONPATH correctly."
            )
        root_str = str(root)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)
        return root

    def _resolve_path(self, value: str | Path | None) -> Path | None:
        if _is_none_like(value):
            return None
        path = Path(value).expanduser()
        if not path.is_absolute() and self.config.pretrained_path is not None:
            path = Path(self.config.pretrained_path).expanduser() / path
        return path

    def _resolve_checkpoint(self) -> str:
        candidates: list[Path] = []
        for value in (self.config.ckpt_path, self.config.model_path):
            resolved = self._resolve_path(value)
            if resolved is not None:
                candidates.append(resolved)
        candidates.extend(
            [
                Path("/home/rxhuang/Projects/models/cosmos"),
                Path(
                    "/data/rxhuang/cosmos_action_focused_runs/cosmos_policy/so101_lerobot/"
                    "action_focused_B_full_dit_2gpu_smoke_20260707/checkpoints/iter_000010000"
                ),
                Path(
                    "/data/rxhuang/cosmos_action_focused_runs/cosmos_policy/so101_lerobot/"
                    "action_focused_B_full_dit_2gpu_smoke_20260707/checkpoints/iter_000000020"
                ),
            ]
        )
        for path in candidates:
            if path.exists():
                return str(path)

        available = []
        for root in (Path("/data/rxhuang/cosmos_action_focused_runs"), Path("/data/rxhuang/cosmos_three_cubes_runs")):
            if root.exists():
                available.extend(str(p) for p in root.glob("**/checkpoints/iter_*") if p.is_dir())
        raise FileNotFoundError(
            "Could not resolve Cosmos checkpoint. Checked policy.ckpt_path/model_path and defaults. "
            f"Available checkpoint paths include: {available[:20]}"
        )

    def _ensure_runtime(self, for_training: bool) -> None:
        if self._runtime_loaded:
            return
        self._install_cosmos_import_path()
        try:
            from cosmos_policy.experiments.robot.cosmos_utils import (
                get_action,
                get_model,
                init_t5_text_embeddings_cache,
                load_dataset_stats,
            )
            from cosmos_policy.models.finetune import apply_finetune_mode
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Set policy.cosmos_repo=/home/rxhuang/Projects/cosmos-policy and PYTHONPATH correctly."
            ) from exc

        self._cosmos_get_action = get_action
        self._cosmos_load_dataset_stats = load_dataset_stats
        init_t5_text_embeddings_cache(self.config.t5_text_embeddings_path)
        self.dataset_stats = load_dataset_stats(self.config.dataset_stats_path)
        for key in ("actions_min", "actions_max", "proprio_min", "proprio_max"):
            self.dataset_stats[key] = _to_numpy_stat(self.dataset_stats[key])
        self.t5_text_embeddings = self._load_t5_embeddings()

        model, runtime_config = get_model(self._deploy_cfg(ckpt_path=self._resolve_checkpoint()))
        self.cosmos_model = model
        self.cosmos_runtime_config = runtime_config
        self._apply_freeze(apply_finetune_mode=apply_finetune_mode, force=True)
        self._load_fine_tuned_overlay()
        self._runtime_loaded = True
        self.to(self.device)
        if not for_training:
            self.eval()

    def _apply_freeze(self, apply_finetune_mode=None, force: bool = False) -> None:
        if self.cosmos_model is None:
            return
        if self._freeze_applied and not force:
            return
        if apply_finetune_mode is None:
            try:
                from cosmos_policy.models.finetune import apply_finetune_mode
            except ModuleNotFoundError:
                return
        self.finetune_report = apply_finetune_mode(
            self.cosmos_model,
            self.config.train_mode,
            self.config.train_last_n_dit_blocks,
            action_head_fallback=None,
        )
        self._freeze_applied = True
        total = sum(p.numel() for p in self.cosmos_model.parameters())
        trainable = sum(p.numel() for p in self.cosmos_model.parameters() if p.requires_grad)
        if not self._trainable_logged:
            LOG.warning(
                "Cosmos trainable params: total=%s trainable=%s ratio=%.4f%% modules=%s",
                f"{total:,}",
                f"{trainable:,}",
                100.0 * trainable / max(total, 1),
                getattr(self.finetune_report, "trainable_module_names", ()),
            )
            self._trainable_logged = True

    def _is_trainable_state_name(self, name: str) -> bool:
        if self.cosmos_model is None:
            return False
        param = dict(self.cosmos_model.named_parameters()).get(name)
        return bool(param is not None and param.requires_grad)

    def _load_fine_tuned_overlay(self) -> None:
        weights_path = self._resolve_path(self.config.fine_tuned_weights_path)
        if weights_path is None or not weights_path.is_file():
            return
        state = load_safetensors(weights_path)
        params = dict(self.cosmos_model.named_parameters())
        loaded = 0
        missing = []
        unexpected = []
        shape_mismatch = []
        with torch.no_grad():
            for name, value in state.items():
                param = params.get(name)
                if param is None:
                    unexpected.append(name)
                    continue
                if tuple(param.shape) != tuple(value.shape):
                    shape_mismatch.append((name, tuple(value.shape), tuple(param.shape)))
                    continue
                param.copy_(value.to(device=param.device, dtype=param.dtype))
                loaded += 1
        for name, param in params.items():
            if param.requires_grad and name not in state:
                missing.append(name)
        LOG.warning(
            "Loaded Cosmos LeRobot overlay %s | loaded=%d missing_trainable=%d unexpected=%d shape_mismatch=%d",
            weights_path,
            loaded,
            len(missing),
            len(unexpected),
            len(shape_mismatch),
        )
        if shape_mismatch:
            LOG.warning("Cosmos overlay shape mismatch examples: %s", shape_mismatch[:5])

    def _load_t5_embeddings(self) -> dict[str, Tensor]:
        path = Path(self.config.t5_text_embeddings_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Missing SO101 T5 embeddings: {path}")
        with path.open("rb") as file:
            return pickle.load(file)

    def _deploy_cfg(self, ckpt_path: str | None = None) -> SimpleNamespace:
        return SimpleNamespace(
            suite="aloha",
            config=self.config.config_name,
            config_file=self.config.config_file,
            ckpt_path=ckpt_path or self._resolve_checkpoint(),
            action_dim=self.config.action_dim,
            chunk_size=self.config.chunk_size,
            num_wrist_images=2,
            num_third_person_images=1,
            use_proprio=True,
            use_wrist_image=True,
            use_third_person_image=True,
            normalize_proprio=True,
            unnormalize_actions=True,
            use_variance_scale=False,
            use_jpeg_compression=False,
            trained_with_image_aug=True,
            ar_future_prediction=False,
            ar_value_prediction=False,
            ar_qvalue_prediction=False,
            num_denoising_steps_action=self.config.num_denoising_steps_action,
        )

    def _task_for_batch_index(self, batch: dict[str, Any], index: int) -> str:
        for key in ("task", "tasks", "language_instruction"):
            if key not in batch:
                continue
            value = batch[key]
            if isinstance(value, list | tuple):
                return str(value[index])
            if isinstance(value, str):
                return value
        return self.config.default_task

    def _t5_for_tasks(self, batch: dict[str, Any], batch_size: int) -> Tensor:
        values = []
        for index in range(batch_size):
            task = self._task_for_batch_index(batch, index)
            if task not in self.t5_text_embeddings:
                if self.config.default_task in self.t5_text_embeddings:
                    task = self.config.default_task
                else:
                    task = next(iter(self.t5_text_embeddings))
            values.append(torch.squeeze(self.t5_text_embeddings[task]).to(dtype=torch.bfloat16))
        return torch.stack(values, dim=0)

    def _build_training_batch(self, batch: dict[str, Tensor]) -> tuple[dict[str, Tensor], dict[str, float]]:
        try:
            from cosmos_policy.datasets.dataset_utils import preprocess_image
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Cosmos preprocessing import failed; check policy.cosmos_repo.") from exc

        raw_actions = torch.as_tensor(batch[self.config.action_key], dtype=torch.float32)
        if raw_actions.ndim == 2:
            raw_actions = raw_actions.unsqueeze(1)
        raw_actions = raw_actions[:, : self.config.chunk_size, : self.config.action_dim]
        if raw_actions.shape[1] < self.config.chunk_size:
            pad = raw_actions[:, -1:].expand(-1, self.config.chunk_size - raw_actions.shape[1], -1)
            raw_actions = torch.cat([raw_actions, pad], dim=1)

        states = torch.as_tensor(batch[self.config.state_key], dtype=torch.float32)
        if states.ndim == 2:
            current_state = states
            future_state = states
        else:
            current_state = states[:, 0, : self.config.state_dim]
            future_state = states[:, -1, : self.config.state_dim]

        actions_np = raw_actions.detach().cpu().numpy().astype(np.float32)
        proprio_np = current_state.detach().cpu().numpy().astype(np.float32)
        future_proprio_np = future_state.detach().cpu().numpy().astype(np.float32)
        norm_actions = _minmax_normalize(actions_np, self.dataset_stats["actions_min"], self.dataset_stats["actions_max"])
        norm_proprio = _minmax_normalize(proprio_np, self.dataset_stats["proprio_min"], self.dataset_stats["proprio_max"])
        norm_future = _minmax_normalize(
            future_proprio_np, self.dataset_stats["proprio_min"], self.dataset_stats["proprio_max"]
        )

        primary = _ensure_batch_time_image(batch[self.config.primary_camera_key])
        wrist_right = _ensure_batch_time_image(batch[self.config.wrist_camera_key])
        wrist_left_key = self.config.wrist_left_camera_key
        wrist_left = _ensure_batch_time_image(batch[wrist_left_key]) if wrist_left_key in batch else wrist_right
        batch_size = raw_actions.shape[0]
        videos = []
        for bidx in range(batch_size):
            current = {
                "primary": _chw_to_hwc_uint8(primary[bidx, 0]),
                "wrist_left": _chw_to_hwc_uint8(wrist_left[bidx, 0]),
                "wrist_right": _chw_to_hwc_uint8(wrist_right[bidx, 0]),
            }
            future = {
                "primary": _chw_to_hwc_uint8(primary[bidx, -1]),
                "wrist_left": _chw_to_hwc_uint8(wrist_left[bidx, -1]),
                "wrist_right": _chw_to_hwc_uint8(wrist_right[bidx, -1]),
            }
            blank = np.zeros_like(current["primary"])
            frames = [
                blank,
                blank,
                current["wrist_left"],
                current["wrist_right"],
                current["primary"],
                blank,
                blank,
                future["wrist_left"],
                future["wrist_right"],
                future["primary"],
                blank,
            ]
            unique_video = preprocess_image(
                np.stack(frames),
                final_image_size=self.config.final_image_size,
                normalize_images=False,
                use_image_aug=self.training,
                stronger_image_aug=self.training,
            )
            repeats = torch.tensor([1] + [4] * 10)
            video = torch.repeat_interleave(unique_video, repeats, dim=1)
            videos.append(video)
        video_tensor = torch.stack(videos, dim=0).to(device=self.device)

        result: dict[str, Tensor] = {
            "dataset_name": "video_data",
            "video": video_tensor,
            "actions": torch.from_numpy(norm_actions).to(device=self.device, dtype=torch.bfloat16),
            "t5_text_embeddings": self._t5_for_tasks(batch, batch_size).to(device=self.device),
            "t5_text_mask": torch.ones((batch_size, 512), dtype=torch.int64, device=self.device),
            "fps": torch.full((batch_size,), 16, dtype=torch.bfloat16, device=self.device),
            "padding_mask": torch.zeros(
                (batch_size, 1, self.config.final_image_size, self.config.final_image_size),
                dtype=torch.bfloat16,
                device=self.device,
            ),
            "image_size": self.config.final_image_size * torch.ones((batch_size, 4), device=self.device),
            "proprio": torch.from_numpy(norm_proprio).to(device=self.device, dtype=torch.bfloat16),
            "future_proprio": torch.from_numpy(norm_future).to(device=self.device, dtype=torch.bfloat16),
            "value_function_return": torch.zeros((batch_size,), dtype=torch.bfloat16, device=self.device),
            "rollout_data_mask": torch.zeros((batch_size,), dtype=torch.long, device=self.device),
            "rollout_data_success_mask": torch.zeros((batch_size,), dtype=torch.long, device=self.device),
            "world_model_sample_mask": torch.zeros((batch_size,), dtype=torch.long, device=self.device),
            "value_function_sample_mask": torch.zeros((batch_size,), dtype=torch.long, device=self.device),
        }
        for key, value in SO101_LATENT_INDICES.items():
            result[key] = torch.full((batch_size,), value, dtype=torch.int64, device=self.device)

        debug = {
            "raw_image_min": float(video_tensor.float().amin().item()),
            "raw_image_max": float(video_tensor.float().amax().item()),
            "raw_action_min": float(raw_actions.amin().item()),
            "raw_action_max": float(raw_actions.amax().item()),
            "raw_state_min": float(current_state.amin().item()),
            "raw_state_max": float(current_state.amax().item()),
            "normalized_action_min": float(np.min(norm_actions)),
            "normalized_action_max": float(np.max(norm_actions)),
        }
        return result, debug

    def _current_state(self, batch: dict[str, Any]) -> Tensor:
        state = torch.as_tensor(batch[self.config.state_key], dtype=torch.float32, device=self.device)
        if state.ndim == 3:
            state = state[:, 0]
        if state.ndim == 1:
            state = state.unsqueeze(0)
        return state[:, : self.config.state_dim]

    def _build_inference_observations(self, batch: dict[str, Any], state: Tensor) -> list[dict[str, np.ndarray]]:
        primary = _ensure_batch_time_image(torch.as_tensor(batch[self.config.primary_camera_key]))
        wrist_right = _ensure_batch_time_image(torch.as_tensor(batch[self.config.wrist_camera_key]))
        wrist_left = (
            _ensure_batch_time_image(torch.as_tensor(batch[self.config.wrist_left_camera_key]))
            if self.config.wrist_left_camera_key in batch
            else wrist_right
        )
        observations = []
        for bidx in range(state.shape[0]):
            observations.append(
                {
                    "primary_image": _chw_to_hwc_uint8(primary[bidx, 0]),
                    "left_wrist_image": _chw_to_hwc_uint8(wrist_left[bidx, 0]),
                    "right_wrist_image": _chw_to_hwc_uint8(wrist_right[bidx, 0]),
                    "proprio": state[bidx].detach().cpu().numpy().astype(np.float32),
                }
            )
        return observations

    def _clip_action_chunk(self, actions: Tensor, state: Tensor) -> Tensor:
        lower = torch.as_tensor(self.dataset_stats["actions_min"], dtype=actions.dtype, device=actions.device)
        upper = torch.as_tensor(self.dataset_stats["actions_max"], dtype=actions.dtype, device=actions.device)
        actions = torch.minimum(torch.maximum(actions, lower), upper)
        obs_delta = torch.full_like(actions, float(self.config.max_delta_from_observation))
        obs_delta[..., 5] = float(self.config.max_gripper_delta_from_observation)
        obs_lower = state[:, None, :] - obs_delta
        obs_upper = state[:, None, :] + obs_delta
        actions = torch.minimum(torch.maximum(actions, obs_lower), obs_upper)

        max_step = torch.full((self.config.action_dim,), float(self.config.max_step_delta), device=actions.device)
        max_step[5] = float(self.config.max_gripper_step_delta)
        clipped = actions.clone()
        prev = state
        for idx in range(actions.shape[1]):
            clipped[:, idx] = torch.minimum(torch.maximum(actions[:, idx], prev - max_step), prev + max_step)
            prev = clipped[:, idx]
        return clipped
