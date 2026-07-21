#!/usr/bin/env python
"""Shared validation/evaluation primitives for LingBoVA training and offline eval.

This module is imported by two call sites so eval logic is written exactly once:
  - The training-time validation hook wired into `lerobot_train.py` (see
    `LingBoVAPolicy.run_lingbo_va_validation`, gated so it is a no-op for every other policy).
  - `scripts/eval_lingbo_va_offline.py`, a standalone CLI that runs the same primitives against
    any saved checkpoint + any episode, as a pre-deployment gate.

Nothing in here duplicates the KV-cache / attention-mode protocol implemented in the vendored
`wan_va.wan_va_server.VA_Server` — `build_weight_sharing_server` subclasses it and reuses its
`_reset` / `_infer` / `_compute_kv_cache` unmodified.
"""

from __future__ import annotations

import csv
import gc
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor

from lerobot.utils.constants import ACTION

if TYPE_CHECKING:
    from .modeling_lingbo_va import LingBoVAPolicy

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------------------------
# Per-step EMA + CSV/markdown logging (Task 2.1, 2.5)
# -----------------------------------------------------------------------------------------------


class TrainMetricsEMA:
    """Stateful EMA(decay) tracker for per-step train metrics. One instance per training run."""

    def __init__(self, decay: float = 0.99):
        self.decay = decay
        self._state: dict[str, float] = {}
        # Rolling window of the last N "min std-ratio across joints" values, used to detect
        # sustained mean-collapse (Task 2.3 item 6: ratio < 0.2 for 3 consecutive rounds).
        self._std_ratio_history: list[float] = []

    def _update_one(self, key: str, value: float) -> float:
        if key not in self._state:
            self._state[key] = value
        else:
            self._state[key] = self.decay * self._state[key] + (1 - self.decay) * value
        return self._state[key]

    def update(
        self, step: int, loss: float, loss_video: float, loss_action: float, grad_norm: float, lr: float
    ) -> dict[str, float]:
        return {
            "step": step,
            "loss": loss,
            "loss_video": loss_video,
            "loss_action": loss_action,
            "grad_norm": grad_norm,
            "lr": lr,
            "loss_ema": self._update_one("loss", loss),
            "loss_video_ema": self._update_one("loss_video", loss_video),
            "loss_action_ema": self._update_one("loss_action", loss_action),
            "grad_norm_ema": self._update_one("grad_norm", grad_norm),
        }

    def check_mean_collapse(self, min_std_ratio_this_round: float, window: int = 3) -> bool:
        """Returns True (and logs a warning) if std(pred)/std(GT) stayed < 0.2 for `window` rounds."""
        self._std_ratio_history.append(min_std_ratio_this_round)
        self._std_ratio_history = self._std_ratio_history[-window:]
        collapsed = len(self._std_ratio_history) == window and all(
            v < 0.2 for v in self._std_ratio_history
        )
        if collapsed:
            logger.warning(
                "[MEAN COLLAPSE WARNING] std(pred)/std(GT) has been below 0.2 for the last "
                "%d validation rounds (values=%s). The action head is likely predicting a "
                "near-constant chunk regardless of input.",
                window,
                self._std_ratio_history,
            )
        return collapsed


class CsvMetricsWriter:
    """Small append-only CSV writer. No existing repo precedent for a generic metrics CSV."""

    def __init__(self, path: str | Path, columns: list[str]):
        self.path = Path(path)
        self.columns = columns
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=columns).writeheader()

    def write_row(self, row: dict[str, Any]) -> None:
        with self.path.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.columns).writerow({k: row.get(k, "") for k in self.columns})


def write_markdown_row(path: str | Path, row: dict[str, Any]) -> None:
    """Append one row to a markdown table at `path`, writing the header on first use."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    columns = list(row.keys())
    with path.open("a", encoding="utf-8") as f:
        if is_new:
            f.write("| " + " | ".join(columns) + " |\n")
            f.write("|" + "|".join(["---"] * len(columns)) + "|\n")
        f.write("| " + " | ".join(str(row[c]) for c in columns) + " |\n")


# -----------------------------------------------------------------------------------------------
# Train-only-scoped normalization stats (Task 1.6)
# -----------------------------------------------------------------------------------------------


def compute_train_only_action_stats(
    repo_id: str,
    root: str | Path | None,
    revision: str | None,
    train_episodes: list[int],
) -> dict[str, list[float]]:
    """Compute q01/q99/min/max/mean/std for the ACTION feature over `train_episodes` only.

    `LeRobotDatasetMetadata.stats` is always global (loaded straight from `meta/stats.json`,
    independent of any `episodes=` filter passed to `LeRobotDataset.__init__` — see
    `lerobot/datasets/dataset_metadata.py`), so it cannot be reused here; this recomputes
    quantiles from raw per-frame actions restricted to `train_episodes`, using the exact same
    quantile set as the generic pipeline (`lerobot.datasets.compute_stats.DEFAULT_QUANTILES`).

    Uses `LeRobotDataset.select_columns([ACTION])` (raw parquet column access) rather than
    `dataset[i]`/`__getitem__`, which decodes every camera's video frames per index — for a
    ~51k-frame dataset that would take a very long time. `select_columns` never touches video.
    """
    from lerobot.datasets.compute_stats import DEFAULT_QUANTILES, get_feature_stats
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id, root=root, revision=revision, episodes=train_episodes)
    action_column = dataset.select_columns([ACTION])[ACTION]
    actions = np.asarray(action_column, dtype=np.float32)
    stats = get_feature_stats(actions, axis=0, keepdims=True, quantile_list=DEFAULT_QUANTILES)
    return {
        "q01": np.asarray(stats["q01"]).reshape(-1).tolist(),
        "q99": np.asarray(stats["q99"]).reshape(-1).tolist(),
        "min": np.asarray(stats["min"]).reshape(-1).tolist(),
        "max": np.asarray(stats["max"]).reshape(-1).tolist(),
        "mean": np.asarray(stats["mean"]).reshape(-1).tolist(),
        "std": np.asarray(stats["std"]).reshape(-1).tolist(),
    }


# -----------------------------------------------------------------------------------------------
# Weight-sharing in-process VA_Server shim (Task 1.5b, reused by 2.3/2.4/3.3)
# -----------------------------------------------------------------------------------------------


def _set_attn_op(module: Any, value: Any) -> None:
    """Set `module.attn_op` bypassing `nn.Module.__setattr__`'s Parameter/Module type guard.

    `WanAttention.__init__` sets `self.attn_op = FlexAttnFunc(...)` for `attn_mode="flex"` —
    `FlexAttnFunc` is itself an `nn.Module`, so plain assignment auto-registers it as a PyTorch
    submodule (`module._modules["attn_op"]`). Reassigning that attribute name to a plain function
    afterward (`custom_sdpa`/`flash_attn_func`, used for "torch"/"flashattn") then raises
    `TypeError: cannot assign ... as child module 'attn_op' (torch.nn.Module or None expected)`,
    since PyTorch guards against silently overwriting a registered submodule with a non-Module
    value. This helper removes any stale `_modules` entry first, then sets the attribute via
    `object.__setattr__` to skip that guard entirely. Safe: `FlexAttnFunc` holds no trainable
    parameters of its own (only class-level compiled/mask state), so nothing the optimizer or
    `.state_dict()` needs is ever silently dropped by unregistering it.
    """
    if "attn_op" in getattr(module, "_modules", {}):
        del module._modules["attn_op"]
    object.__setattr__(module, "attn_op", value)


def _override_attn_mode(transformer: Any, attn_mode: str) -> None:
    """In-place reassign `WanAttention.attn_op` without reloading the model.

    Only "torch"/"flashattn" are supported here (inference-only attn ops that operate per-call
    with no persistent mask state); "flex" requires the training-only block mask machinery and
    cannot be swapped in this way.
    """
    if attn_mode not in {"torch", "flashattn"}:
        raise ValueError(
            f"_override_attn_mode only supports 'torch'/'flashattn' (got '{attn_mode}'); "
            "'flex' requires re-instantiation, not in-place reassignment."
        )
    from wan_va.modules.model import custom_sdpa

    if attn_mode == "torch":
        new_op = custom_sdpa
    else:
        from wan_va.modules.model import flash_attn_func

        if flash_attn_func is None:
            raise ImportError("attn_mode='flashattn' requires flash-attn to be installed.")
        new_op = flash_attn_func

    count = 0
    for module in transformer.modules():
        if hasattr(module, "attn_op"):
            _set_attn_op(module, new_op)
            count += 1
    if count == 0:
        raise RuntimeError(
            "_override_attn_mode found no submodules with an `attn_op` attribute; the vendored "
            "WanAttention implementation may have changed."
        )
    logger.info("Overrode attn_op to '%s' on %d submodules.", attn_mode, count)


def _write_prompt_embedding_cache(policy: "LingBoVAPolicy", prompt: str) -> tuple[str, str]:
    """Precompute the real-prompt and empty-prompt (CFG negative) text embeddings and write them
    to disk, so the eval server can load them via `VA_Server._reset`'s `prompt_emb_path`/
    `negative_prompt_emb_path` mechanism instead of ever calling `encode_prompt` (which needs the
    T5 text encoder resident on GPU).

    Both embeddings are almost always already cached on CPU in `policy._prompt_cache` — every
    normal training forward pass encodes the (constant, single-task) prompt, and `cfg_prob>0`
    means the empty-string embedding gets computed within the first few hundred steps too — so
    this call is typically a cache hit with zero GPU work. Returns (prompt_emb_path,
    negative_prompt_emb_path) as strings, matching what `VA_Server._reset` expects: files
    containing a `[L, D]` tensor (no batch dim — `_reset` adds it via `.unsqueeze(0)`).
    """
    cache_dir = Path(policy.config.save_root).expanduser() / "eval_prompt_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    real_emb = policy._encode_prompts([prompt])[0].detach().cpu()
    empty_emb = policy._empty_prompt_embedding().detach().cpu()
    real_path = cache_dir / "prompt_emb.pt"
    empty_path = cache_dir / "empty_prompt_emb.pt"
    torch.save(real_emb, real_path)
    torch.save(empty_emb, empty_path)
    return str(real_path), str(empty_path)


@contextmanager
def build_weight_sharing_server(
    policy: "LingBoVAPolicy", attn_mode_override: str | None = None, prompt: str | None = None
):
    """Context manager yielding a VA_Server-compatible object sharing `policy`'s already-loaded
    weights in place, instead of loading an independent copy from disk (what `_ensure_local_server()`
    does today).

    This is the single reusable inference primitive for training-time teacher-forced/open-loop
    eval (Task 2.3, 2.4) and for making `_ensure_local_runtime(for_training=False)` meaningful
    (Task 1.5b): reusing `self._transformer`/`_vae`/`_text_encoder` avoids a disk round-trip and
    avoids holding two full model copies in GPU memory simultaneously, which would not fit
    alongside an active training run on a single 24GB GPU.

    `_reset`/`_infer`/`_compute_kv_cache`/`infer` are inherited UNMODIFIED from `VA_Server` (only
    `__init__` is overridden here) — this reuses the existing, verified-correct KV-cache/
    `frame_st_id` protocol rather than reimplementing any of it. Every attribute `VA_Server._reset`
    needs beyond what's set below is derived from `self.job_config`, which
    `policy._make_va_job_config()` already builds correctly.

    Requires `policy._transformer` etc. to already be loaded (call `policy.forward(...)` or
    `policy._ensure_local_runtime(for_training=True)` at least once first).

    MUST be used as a context manager (`with build_weight_sharing_server(...) as server:`), not
    called directly — this is what `policy._transformer` shared with training is the SAME live
    object, not a copy. `attn_mode_override` reassigns every `WanAttention.attn_op` in place
    (needed because a `train_attn_mode="flex"` transformer cannot run the streaming/KV-cache
    `forward()` path at all); leaving that override in place after eval would silently corrupt
    every subsequent training step's attention masking (train_attn_mode='flex' would still be set
    on the config, but the live model object would keep using unmasked attention). On exit this
    restores the original attn_op on every submodule and clears the eval-only KV cache the
    server allocated, so training resumes exactly as if no eval had happened.

    `prompt`: if given, precomputes the prompt/empty-prompt text embeddings (see
    `_write_prompt_embedding_cache`) and points the server at them via `prompt_emb_path`/
    `negative_prompt_emb_path`, so `_reset` never needs the T5 text encoder resident on GPU at
    all. On a near-full 24GB training footprint (resident LoRA transformer + VAE), simultaneously
    loading the text encoder too can OOM even after clearing caches — this sidesteps the problem
    entirely rather than trying to survive it. Pass `prompt=None` only if you intend to call
    `server.infer({"reset": True, ...})` without ever encoding a prompt (e.g. `reset=None`).
    """
    if getattr(policy, "_transformer", None) is None:
        raise RuntimeError(
            "build_weight_sharing_server requires policy._transformer to already be loaded; "
            "call policy._ensure_local_runtime(for_training=True) (or policy.forward(...)) first."
        )

    policy._install_lingbo_import_path()
    from wan_va.utils.scheduler import FlowMatchScheduler
    from wan_va.wan_va_server import VA_Server

    job_config = policy._make_va_job_config()
    if prompt is not None:
        prompt_emb_path, negative_prompt_emb_path = _write_prompt_embedding_cache(policy, prompt)
        job_config.prompt_emb_path = prompt_emb_path
        job_config.negative_prompt_emb_path = negative_prompt_emb_path
    transformer = policy._transformer
    device = next(transformer.parameters()).device

    class _WeightSharingVAServer(VA_Server):
        def __init__(self):  # noqa: super().__init__ intentionally not called (skip disk loads)
            self.cache_name = "pos"
            self.job_config = job_config
            self.save_root = job_config.save_root
            self.dtype = job_config.param_dtype
            self.device = device
            self.enable_offload = bool(getattr(job_config, "enable_offload", False))

            self.scheduler = FlowMatchScheduler(shift=job_config.snr_shift, sigma_min=0.0, extra_one_step=True)
            self.action_scheduler = FlowMatchScheduler(
                shift=job_config.action_snr_shift, sigma_min=0.0, extra_one_step=True
            )
            self.scheduler.set_timesteps(1000, training=True)
            self.action_scheduler.set_timesteps(1000, training=True)

            # Move the shared VAE onto the transformer's device: it may currently be offloaded to
            # CPU (offload_vae_after_encode) from the last training step's encode call. Restored
            # to its configured device on exit below, matching what the next normal training step
            # would do anyway.
            #
            # The text encoder is deliberately NOT moved here. On an already near-full 24GB
            # training footprint (resident LoRA transformer + VAE), moving the T5 text encoder to
            # GPU too can OOM outright — not just fragmentation, genuinely not enough capacity
            # (observed: "23.48 GiB memory in use" of a 23.55 GiB card, 45 MiB free). When
            # `prompt_emb_path`/`negative_prompt_emb_path` are set (see the `prompt` argument
            # above), `VA_Server._reset` loads precomputed embeddings from disk and never touches
            # `self.text_encoder` at all. If no `prompt` was given, `_reset_and_encode_prompt`
            # falls back to moving the text encoder to GPU just for that call.
            policy._vae.to(self.device)

            self.vae = policy._vae
            self.streaming_vae = policy._streaming_vae
            self.tokenizer = policy._tokenizer
            self.text_encoder = policy._text_encoder
            self.transformer = transformer
            self.prompt_emb_path = getattr(job_config, "prompt_emb_path", None)
            self.negative_prompt_emb_path = getattr(job_config, "negative_prompt_emb_path", None)
            self.env_type = job_config.env_type
            self.streaming_vae_half = None

    original_attn_ops: list[tuple[Any, Any]] = []
    if attn_mode_override is not None:
        original_attn_ops = [(m, m.attn_op) for m in transformer.modules() if hasattr(m, "attn_op")]
        _override_attn_mode(transformer, attn_mode_override)

    server = _WeightSharingVAServer()
    try:
        yield server
    finally:
        for module, original_op in original_attn_ops:
            module.attn_op = original_op
        try:
            transformer.clear_cache(server.cache_name)
        except Exception:
            logger.warning("build_weight_sharing_server: failed to clear KV cache on exit.", exc_info=True)
        if policy.config.offload_vae_after_encode and device.type == "cuda":
            policy._vae.to("cpu")
        if policy.config.offload_text_encoder_after_encode and device.type == "cuda":
            policy._text_encoder.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()


def _reset_and_encode_prompt(policy: "LingBoVAPolicy", server: Any, prompt: str) -> None:
    """Reset `server` and encode `prompt`.

    If `server.prompt_emb_path` is already set (the normal case — `build_weight_sharing_server`
    was given `prompt=...` and precomputed it), `VA_Server._reset` loads the embedding from disk
    and never touches `self.text_encoder`, so this is just `server.infer({"reset": True, ...})`.
    Otherwise, falls back to moving the (possibly large) T5 text encoder to GPU for this one call
    and offloading it again immediately afterward — this fallback path is what OOMed on an
    already near-full 24GB training footprint, so prefer passing `prompt=` to
    `build_weight_sharing_server` over relying on it."""
    device = server.device
    needs_live_encode = not getattr(server, "prompt_emb_path", None)
    if needs_live_encode and device.type == "cuda":
        # Best-effort: reclaim any fragmented/cached memory before this device move, which is
        # right at the edge of the training memory budget (see build_weight_sharing_server).
        gc.collect()
        torch.cuda.empty_cache()
        policy._text_encoder.to(device)
    try:
        server.infer({"reset": True, "prompt": prompt})
    finally:
        if not needs_live_encode:
            return
        if policy.config.offload_text_encoder_after_encode and device.type == "cuda":
            policy._text_encoder.to("cpu")
            torch.cuda.empty_cache()


# -----------------------------------------------------------------------------------------------
# Dataset helpers shared by the eval routines below
# -----------------------------------------------------------------------------------------------


def _resolve_camera_keys_for_eval(policy: "LingBoVAPolicy") -> list[str]:
    """Camera keys to use when building synthetic eval batches, resolved from config alone.

    `policy._resolve_camera_keys(batch)` (used by the training/inference forward paths) always
    validates its configured camera_keys are present IN a given batch — it isn't meant to be
    called before a batch exists (calling it with `{}` raises `KeyError`, since every configured
    key is trivially "missing" from an empty dict). The eval routines in this module need the key
    list *before* they can build the batch (`_batch_at` needs `camera_keys` as an input), so this
    reads `config.camera_keys` directly instead.
    """
    if policy.config.camera_keys:
        return list(policy.config.camera_keys)
    raise KeyError(
        "LingBoVA eval routines require policy.config.camera_keys to be set explicitly; "
        "the batch-driven auto-detection fallback in _resolve_camera_keys needs a real batch, "
        "which isn't available yet at this point."
    )


def _make_single_episode_dataset(policy: "LingBoVAPolicy", repo_id: str, root, revision, episode_id: int):
    """A `LeRobotDataset` restricted to exactly one episode, with delta_timestamps matching the
    policy's training-shaped windows (`observation_delta_indices`/`action_delta_indices`), so
    local index `i` == frame `i` of that episode (no global/local index bookkeeping needed)."""
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    ds_meta = LeRobotDatasetMetadata(repo_id, root=root, revision=revision)
    delta_timestamps = resolve_delta_timestamps(policy.config, ds_meta)
    return LeRobotDataset(
        repo_id, root=root, revision=revision, episodes=[episode_id], delta_timestamps=delta_timestamps
    )


def _batch_at(dataset, index: int, camera_keys: list[str], task_key: str) -> dict[str, Any]:
    """Build a batch-size-1 dict from `dataset[index]`, matching the shape `policy.forward`/
    `_make_server_payload` expect (leading batch dim added to every tensor)."""
    sample = dataset[index]
    batch: dict[str, Any] = {}
    for key in camera_keys:
        batch[key] = sample[key].unsqueeze(0)
    batch[ACTION] = sample[ACTION].unsqueeze(0)
    batch[task_key] = sample.get(task_key, "")
    return batch


# -----------------------------------------------------------------------------------------------
# Fixed-timestep validation loss (Task 2.2)
# -----------------------------------------------------------------------------------------------


def run_fixed_timestep_validation(
    policy: "LingBoVAPolicy",
    repo_id: str,
    root,
    revision,
    val_episodes: list[int],
    timestep_fractions: tuple[float, ...] = (0.25, 0.5, 0.75),
    num_samples: int = 32,
    seed: int = 12345,
) -> dict[str, float]:
    """Low-variance validation loss: fixed 32 samples from `val_episodes`, fixed flow-matching
    timesteps, fixed RNG seed — mirrors upstream's `Trainer.validate()` pinned-RNG pattern.
    Reuses `policy.forward()`'s existing loss computation directly (no reimplementation)."""
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

    ds_meta = LeRobotDatasetMetadata(repo_id, root=root, revision=revision)
    delta_timestamps = resolve_delta_timestamps(policy.config, ds_meta)
    dataset = LeRobotDataset(
        repo_id, root=root, revision=revision, episodes=val_episodes, delta_timestamps=delta_timestamps
    )
    generator = torch.Generator().manual_seed(seed)
    num_samples = min(num_samples, len(dataset))
    indices = torch.randperm(len(dataset), generator=generator)[:num_samples].tolist()
    camera_keys = _resolve_camera_keys_for_eval(policy)

    was_training = policy.training
    policy.eval()
    totals = {f"val/loss_video@{frac}": 0.0 for frac in timestep_fractions}
    totals.update({f"val/loss_action@{frac}": 0.0 for frac in timestep_fractions})
    count = 0
    device_type = next(policy._transformer.parameters()).device.type
    with torch.no_grad():
        for idx in indices:
            batch = _batch_at(dataset, idx, camera_keys, policy.config.task_key)
            for frac in timestep_fractions:
                _, out = policy.forward(batch, fixed_timesteps=torch.tensor([frac]))
                totals[f"val/loss_video@{frac}"] += out["loss_video"]
                totals[f"val/loss_action@{frac}"] += out["loss_action"]
            count += 1
            # ~100 forward passes through VAE-encode + transformer here, each allocating and
            # freeing differently-shaped intermediate tensors. PyTorch's caching allocator holds
            # freed blocks as "reserved" rather than returning them to the OS, which fragments
            # over many iterations and was observed to leave too little *contiguous* free memory
            # for the subsequent build_weight_sharing_server's VAE/text-encoder device moves, even
            # though this loop's own peak usage looked fine in isolation. Periodic empty_cache()
            # keeps that reserved pool from ballooning across the full 96-iteration loop.
            if device_type == "cuda" and count % 8 == 0:
                gc.collect()
                torch.cuda.empty_cache()
    if was_training:
        policy.train()
    if device_type == "cuda":
        gc.collect()
        torch.cuda.empty_cache()
    if count == 0:
        return {}
    result = {k: v / count for k, v in totals.items()}
    result["val/loss_video"] = sum(result[f"val/loss_video@{frac}"] for frac in timestep_fractions) / len(
        timestep_fractions
    )
    result["val/loss_action"] = sum(
        result[f"val/loss_action@{frac}"] for frac in timestep_fractions
    ) / len(timestep_fractions)
    return result


# -----------------------------------------------------------------------------------------------
# Teacher-forced action-curve eval (Task 2.3) and full-episode open-loop eval (Task 2.4)
# -----------------------------------------------------------------------------------------------

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")


def _action_curve_metrics(pred: np.ndarray, gt: np.ndarray) -> dict[str, Any]:
    """pred, gt: (T, action_dim). Returns per-joint MAE, direction-consistency, std ratio, etc."""
    abs_err = np.abs(pred - gt)
    per_joint_mae = abs_err.mean(axis=0)
    total_mae = float(abs_err.mean())
    pred_diff_sign = np.sign(np.diff(pred, axis=0))
    gt_diff_sign = np.sign(np.diff(gt, axis=0))
    direction_consistency = float((pred_diff_sign == gt_diff_sign).mean()) if pred.shape[0] > 1 else float("nan")
    std_pred = pred.std(axis=0)
    std_gt = gt.std(axis=0)
    std_ratio = std_pred / np.clip(std_gt, 1e-6, None)
    return {
        "per_joint_mae": per_joint_mae.tolist(),
        "total_mae": total_mae,
        "direction_consistency": direction_consistency,
        "std_ratio": std_ratio.tolist(),
        "min_std_ratio": float(std_ratio.min()),
    }


def _plot_action_curves(pred: np.ndarray, gt: np.ndarray, title: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(JOINT_NAMES), 1, figsize=(10, 2.2 * len(JOINT_NAMES)), sharex=True)
    t = np.arange(pred.shape[0])
    for i, name in enumerate(JOINT_NAMES):
        ax = axes[i]
        ax.plot(t, gt[:, i], label="GT", color="tab:blue")
        ax.plot(t, pred[:, i], label="pred", color="tab:orange")
        ax.plot(t, np.abs(pred[:, i] - gt[:, i]), label="abs error", color="tab:red", linestyle="--", alpha=0.6)
        ax.set_ylabel(name)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("step")
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def run_teacher_forced_eval(
    policy: "LingBoVAPolicy",
    repo_id: str,
    root,
    revision,
    episode_id: int,
    offsets: list[int],
    output_dir: str | Path,
    step: int,
    prompt: str | None = None,
) -> dict[str, Any]:
    """Single-shot conditional generation at each of `offsets`, conditioned on real history from
    `episode_id`, compared to GT — NOT a chained autoregressive rollout (see `run_open_loop_episode_eval`
    for that). Each offset independently: reset -> infer one chunk -> compare -> discard (no
    compute_kv_cache is ever sent), matching upstream's single-chunk-conditional-generation
    semantics for this diagnostic."""
    dataset = _make_single_episode_dataset(policy, repo_id, root, revision, episode_id)
    camera_keys = _resolve_camera_keys_for_eval(policy)
    task_text = prompt or dataset[0].get(policy.config.task_key, policy.config.default_task)

    policy._ensure_local_runtime(for_training=True)

    per_offset: list[dict[str, Any]] = []
    output_dir = Path(output_dir) / f"step_{step:06d}"
    with build_weight_sharing_server(
        policy, attn_mode_override=policy.config.inference_attn_mode, prompt=task_text
    ) as server:
        for offset in offsets:
            if offset >= len(dataset):
                logger.warning(
                    "Teacher-forced eval offset %d >= episode length %d, skipping.", offset, len(dataset)
                )
                continue
            batch = _batch_at(dataset, offset, camera_keys, policy.config.task_key)
            gt_action = batch[ACTION][0].numpy()  # (chunk_size, action_dim)

            _reset_and_encode_prompt(policy, server, task_text)
            payload = policy._make_server_payload(batch, reset=False)
            response = server.infer(payload)
            pred_action = policy._server_action_to_tensor(response["action"])[0].numpy()  # (chunk_size, action_dim)
            server.transformer.clear_cache(server.cache_name)

            n = min(pred_action.shape[0], gt_action.shape[0])
            metrics = _action_curve_metrics(pred_action[:n], gt_action[:n])
            metrics["offset"] = offset
            per_offset.append(metrics)
            _plot_action_curves(
                pred_action[:n],
                gt_action[:n],
                title=f"teacher-forced ep{episode_id} offset={offset} step={step}",
                out_path=output_dir / f"teacher_forced_ep{episode_id}_offset{offset}.png",
            )

    if not per_offset:
        return {}
    agg_per_joint_mae = np.mean([m["per_joint_mae"] for m in per_offset], axis=0)
    agg = {
        "teacher_forced/total_mae": float(np.mean([m["total_mae"] for m in per_offset])),
        "teacher_forced/direction_consistency": float(
            np.nanmean([m["direction_consistency"] for m in per_offset])
        ),
        "teacher_forced/min_std_ratio": float(np.min([m["min_std_ratio"] for m in per_offset])),
        "teacher_forced/per_joint_mae": {
            name: float(agg_per_joint_mae[i]) for i, name in enumerate(JOINT_NAMES[: len(agg_per_joint_mae)])
        },
        "teacher_forced/per_offset": per_offset,
    }
    return agg


def run_open_loop_episode_eval(
    policy: "LingBoVAPolicy",
    repo_id: str,
    root,
    revision,
    episode_id: int,
    stride: int,
    output_dir: str | Path,
    step: int,
    prompt: str | None = None,
) -> dict[str, Any]:
    """Full-episode open-loop replay: reset once, then repeatedly predict a chunk, advance
    `stride` steps, and re-query with fresh GT-conditioned camera frames from `episode_id`'s
    recorded video. The KV cache is committed with the MODEL'S OWN predicted actions (true
    open-loop rollout — measures compounding drift, matching real deployment semantics), not GT
    actions. There is no physical robot in this offline replay: only the camera frames come from
    the real (ground-truth) episode at each new offset; the action history conditioning the model
    is entirely its own prior predictions."""
    dataset = _make_single_episode_dataset(policy, repo_id, root, revision, episode_id)
    camera_keys = _resolve_camera_keys_for_eval(policy)
    task_text = prompt or dataset[0].get(policy.config.task_key, policy.config.default_task)
    action_per_frame = policy.config.action_per_frame

    policy._ensure_local_runtime(for_training=True)

    all_pred: list[np.ndarray] = []
    all_gt: list[np.ndarray] = []
    episode_len = len(dataset)
    with build_weight_sharing_server(
        policy, attn_mode_override=policy.config.inference_attn_mode, prompt=task_text
    ) as server:
        _reset_and_encode_prompt(policy, server, task_text)

        offset = 0
        while offset < episode_len:
            batch = _batch_at(dataset, offset, camera_keys, policy.config.task_key)
            gt_action = batch[ACTION][0].numpy()

            payload = policy._make_server_payload(batch, reset=False)
            response = server.infer(payload)
            pred_action = policy._server_action_to_tensor(response["action"])[0].numpy()

            take = min(stride, pred_action.shape[0], gt_action.shape[0], episode_len - offset)
            all_pred.append(pred_action[:take])
            all_gt.append(gt_action[:take])

            # Commit the model's own predicted actions (not GT) to the KV cache: true open-loop.
            # Sent directly to this in-process `server` — NOT via policy.commit_executed_action,
            # which would dispatch through the policy's configured server/local backend instead
            # of this eval-only weight-sharing server (see _build_kv_cache_payload's docstring).
            executed = torch.from_numpy(pred_action[:take]).to(torch.float32)
            pad = (-take) % action_per_frame
            if pad:
                executed = torch.cat([executed, executed[-1:].repeat(pad, 1)], dim=0)
            kv_payload = policy._build_kv_cache_payload(executed, batch)
            server.infer(kv_payload)

            offset += take
            if take == 0:
                break

    if not all_pred:
        return {}
    pred = np.concatenate(all_pred, axis=0)
    gt = np.concatenate(all_gt, axis=0)
    metrics = _action_curve_metrics(pred, gt)
    output_dir = Path(output_dir) / f"step_{step:06d}"
    _plot_action_curves(
        pred, gt, title=f"open-loop ep{episode_id} step={step}", out_path=output_dir / f"open_loop_ep{episode_id}.png"
    )
    return {
        "open_loop/total_mae": metrics["total_mae"],
        "open_loop/direction_consistency": metrics["direction_consistency"],
        "open_loop/min_std_ratio": metrics["min_std_ratio"],
        "open_loop/per_joint_mae": {
            name: metrics["per_joint_mae"][i] for i, name in enumerate(JOINT_NAMES[: len(metrics["per_joint_mae"])])
        },
    }
