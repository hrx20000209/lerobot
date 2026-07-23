"""Self-conditioning training for LingBot-VA's action history (fixes autoregressive drift).

Background (verified this session on our own step_2000 checkpoint before writing any of
this): teacher-forced open-loop amplitude = 93.0% of GT, 0/6 joints worse than a
constant-midpoint baseline. Swap only the *action history* fed back into the KV cache from
the episode's real actions to the model's own previous prediction (observations stay real)
-> amplitude collapses to 38.2%, 4/6 joints worse than the midpoint baseline. Diagnosed with
scripts/inference/diagnose_lingbot_va_actions.py (--self-feedback). The model has learned to
extrapolate its action-history token stream rather than read the (always noiseless) video
stream, and an out-of-distribution history collapses it toward the per-joint mean.

Where this hooks in (verified by reading utils.py's FlexAttnFunc / forward_train, not
guessed): a training example's "action history" is *not* a separate bookkeeping structure
the way _obs_buffer is at inference time. Every action frame is embedded *twice* per forward
pass -- once as a "noisy" query token (from `action_dict["noisy_latents"]`, denoised and
supervised by `targets`) and once as a "clean" key/value token (from
`action_dict["latent"]`, attended to by strictly-later noisy queries through
`noise2clean_mask + block_causal_mask_exclude_self`). `LingBotVAPolicy._add_noise_stream`
computes `noisy_latents`/`targets` from its `latent` argument *before* the
noisy_cond_prob branch that would otherwise perturb the returned `"latent"` field -- and
actions are always called with noisy_cond_prob=0.0, so `action_dict["latent"]` comes back
as the untouched, masked GT action tensor. That field is exactly the "history" the model
attends to, and it's disjoint from the tensors that produce the loss. Self-conditioning
therefore only ever needs to overwrite `action_dict["latent"]` *after* the normal
`_add_noise_stream` call returns it -- `targets`/`noisy_latents` (and hence the supervised
objective) are computed beforehand from real GT and are never touched.

Two things checked before implementing (per the task's own instructions), not assumed:
  - `observation.state` is never referenced anywhere in modeling_lingbot_va.py -- the model
    has no proprioception channel other than the action-history tokens themselves. So
    "keep the most recent chunk's history clean some of the time" is not optional here.
  - Our own training driver (common.py) builds one `LeRobotDataset` window per example via
    `episodes=[...]` + `delta_timestamps` from a single episode index -- there is no
    multi-episode sequence packing in this pipeline, so no episode-boundary mask exists to
    respect in the first place.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from einops import rearrange


@dataclass
class SelfCondConfig:
    p_max: float = 0.5
    ramp_frac: float = 0.4
    x0_t: float = 0.5
    keep_recent_clean_prob: float = 0.5
    action_history_noise: bool = False  # mutually exclusive ablation arm
    action_history_noise_prob: float = 0.4


def selfcond_schedule(step: int, num_steps: int, cfg: SelfCondConfig) -> float:
    """Linear ramp 0 -> p_max over the first `ramp_frac` fraction of training."""
    if cfg.p_max <= 0:
        return 0.0
    ramp_steps = max(1, int(cfg.ramp_frac * num_steps))
    return cfg.p_max * min(1.0, step / ramp_steps)


def _low_pass_noise(shape, device, dtype, kernel: int = 3) -> torch.Tensor:
    """Gaussian noise smoothed along the action_per_frame axis (dim=-2 of [..., f, n, 1])."""
    noise = torch.randn(shape, device=device, dtype=dtype)
    # shape: [B, action_dim, F, action_per_frame, 1] -- smooth along action_per_frame (dim 3)
    weight = torch.ones(1, 1, kernel, device=device, dtype=dtype) / kernel
    b, c, f, n, one = shape
    flat = noise.permute(0, 1, 2, 4, 3).reshape(-1, 1, n)  # [B*C*F*1, 1, n]
    padded = torch.nn.functional.pad(flat, (kernel // 2, kernel // 2), mode="replicate")
    smoothed = torch.nn.functional.conv1d(padded, weight)
    return smoothed.reshape(b, c, f, one, n).permute(0, 1, 2, 4, 3)


def _chunk_bounds(n_frames: int, chunk_size: int) -> list[tuple[int, int]]:
    return [(s, min(s + chunk_size, n_frames)) for s in range(0, n_frames, chunk_size)]


@torch.no_grad()
def _selfcond_x0_estimate(policy, latent_dict, action_gt, actions_mask, text_emb, chunk_size, window_size, x0_t):
    """Pass 1: one no-grad teacher-forced forward at a fixed timestep, single-step x0 estimate.

    a_t = (1 - t) * clean + t * noise  (this codebase's add_noise convention)
    v   = noise - clean                (training_target convention)
    =>  clean = a_t - t * v  -- matches the spec's a0_hat = a_t - t * v_hat exactly.
    """
    device = action_gt.device
    dtype = action_gt.dtype
    b, _, f, n, _ = action_gt.shape
    sched = policy._train_sched_action
    timestep_value = x0_t * sched.num_train_timesteps
    timesteps = torch.full((f,), timestep_value, device=device, dtype=torch.float32)

    noise = torch.zeros_like(action_gt).normal_()
    noisy = sched.add_noise(action_gt, noise, timesteps, t_dim=2)
    noisy = noisy * actions_mask

    action_dict_p1 = {
        "timesteps": timesteps[None].repeat(b, 1),
        "noisy_latents": noisy,
        "latent": action_gt,  # Pass 1 bootstraps off the real (GT) history, only Pass 2 self-conditions
        "cond_timesteps": torch.zeros_like(timesteps)[None].repeat(b, 1),
        "text_emb": text_emb,
    }
    # grid_id depends only on shape/frame indices, not on noise -- reuse the same construction
    # _add_noise_stream uses (import locally to avoid a module-level dependency cycle).
    from lerobot.policies.lingbot_va.utils import get_mesh_id

    action_dict_p1["grid_id"] = (
        get_mesh_id(f, n, 1, t=1, f_w=1, f_shift=0, action=True).to(device)[None].repeat(b, 1, 1)
    )

    input_dict_p1 = {
        "latent_dict": latent_dict,
        "action_dict": action_dict_p1,
        "chunk_size": chunk_size,
        "window_size": window_size,
    }
    _, action_pred_p1 = policy.transformer(input_dict_p1, train_mode=True)
    v_hat = rearrange(action_pred_p1, "b (f n) c -> b c f n 1", f=f)
    a0_hat = noisy - x0_t * v_hat
    return (a0_hat * actions_mask).detach()


def training_loss_selfcond(policy, batch, step: int, num_steps: int, cfg: SelfCondConfig):
    """Drop-in replacement for `policy.forward(batch)` / `training_loss_from_streams`.

    Identical output to the unmodified path whenever no replacement happens this step
    (guaranteed when `cfg.p_max == 0` and `cfg.action_history_noise is False`: the random
    draws consumed, in order, are exactly `_add_noise_stream(video)`, `_add_noise_stream(action)`,
    then chunk_size/window_size -- the same sequence `training_loss_from_streams` consumes,
    so no extra randomness is introduced and the two paths are bit-identical for the same seed).
    """
    if policy.config.attn_mode != "flex":
        raise ValueError("self-conditioning training requires attn_mode='flex'")
    policy._ensure_frozen_modules()  # lazy VAE/text-encoder load; forward() does this too
    policy._ensure_train_schedulers()

    latents, actions, actions_mask, text_emb = policy._build_training_streams(batch)
    latent_dict = policy._add_noise_stream(
        latents, policy._train_sched_latent, action_mask=None, action_mode=False, noisy_cond_prob=0.5
    )
    action_dict = policy._add_noise_stream(
        actions, policy._train_sched_action, action_mask=actions_mask, action_mode=True, noisy_cond_prob=0.0
    )
    latent_dict["text_emb"] = text_emb
    action_dict["text_emb"] = text_emb
    action_dict["actions_mask"] = actions_mask

    chunk_size = int(torch.randint(1, 5, (1,)).item())
    window_size = int(torch.randint(4, 65, (1,)).item())

    # Reference scale for the a0-vs-GT L2 sanity check: mean step-to-step GT action
    # difference (normalized space), so "a0_gt_l2 stays > 2x this" has a concrete meaning.
    gt_step_diff = (action_dict["latent"][:, :, 1:] - action_dict["latent"][:, :, :-1]).abs().mean().item()
    metrics_extra = {
        "selfcond_p": 0.0, "selfcond_replace_rate": 0.0, "selfcond_a0_gt_l2": None,
        "selfcond_gt_step_diff": gt_step_diff,
    }

    n_frames = action_dict["latent"].shape[2]
    bounds = _chunk_bounds(n_frames, chunk_size)
    n_chunks = len(bounds)

    if cfg.action_history_noise:
        # Ablation arm: structured perturbation instead of a learned self-prediction.
        n_replaced = 0
        for ci, (s, e) in enumerate(bounds):
            if torch.rand(()).item() >= cfg.action_history_noise_prob:
                continue
            gt_chunk = action_dict["latent"][:, :, s:e]
            scale = torch.empty((), device=gt_chunk.device).uniform_(0.4, 1.0)
            noise = _low_pass_noise(gt_chunk.shape, gt_chunk.device, gt_chunk.dtype) * gt_chunk.std().clamp_min(1e-4) * 0.1
            action_dict["latent"][:, :, s:e] = gt_chunk * scale + noise
            n_replaced += 1
        action_dict["latent"] = action_dict["latent"] * actions_mask
        metrics_extra["selfcond_replace_rate"] = n_replaced / max(n_chunks, 1)

    elif cfg.p_max > 0:
        p = selfcond_schedule(step, num_steps, cfg)
        metrics_extra["selfcond_p"] = p
        if p > 0:
            a0_hat = _selfcond_x0_estimate(
                policy, latent_dict, action_dict["latent"], actions_mask, text_emb,
                chunk_size, window_size, cfg.x0_t,
            )
            l2s = []
            n_replaced = 0
            for ci, (s, e) in enumerate(bounds):
                is_last = ci == n_chunks - 1
                if is_last and torch.rand(()).item() < cfg.keep_recent_clean_prob:
                    continue  # action history is this model's only proprioception proxy
                if torch.rand(()).item() >= p:
                    continue
                gt_chunk = action_dict["latent"][:, :, s:e]
                a0_chunk = a0_hat[:, :, s:e]
                l2s.append((gt_chunk - a0_chunk).pow(2).mean().sqrt().item())
                action_dict["latent"][:, :, s:e] = a0_chunk
                n_replaced += 1
            action_dict["latent"] = action_dict["latent"] * actions_mask
            metrics_extra["selfcond_replace_rate"] = n_replaced / max(n_chunks, 1)
            metrics_extra["selfcond_a0_gt_l2"] = sum(l2s) / len(l2s) if l2s else None

    input_dict = {
        "latent_dict": latent_dict,
        "action_dict": action_dict,
        "chunk_size": chunk_size,
        "window_size": window_size,
    }
    pred = policy.transformer(input_dict, train_mode=True)
    latent_loss, action_loss = policy._flow_matching_loss(input_dict, pred)
    loss = latent_loss + action_loss
    metrics = {"latent_loss": latent_loss.item(), "action_loss": action_loss.item(), **metrics_extra}
    return loss, metrics
