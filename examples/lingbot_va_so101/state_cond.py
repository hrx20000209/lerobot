"""State-conditioned training for LingBot-VA: fix the gripper train/deploy mismatch.

Root cause (docs/closed_loop_verification.md UPDATE #2): training conditions the action-history
tokens on the GROUND-TRUTH ACTION (gripper ~4 during grasp), but deployment (state-feedback, the
only drift-free option) can only supply MEASURED STATE (gripper ~15, stalled on the cube). Arm
channels are unaffected (action~=state, r 0.93-0.997); the gripper is off by ~10 during the grasp
and the model -- which leans on the action-history for the barely-visible gripper -- outputs the
stalled ~15 and never grips.

Fix: condition the action-history on measured STATE while keeping the ACTION as the flow-matching
target. Then the model learns "given gripper-state-history ~15 (stalling), keep commanding ~4",
which is exactly the state-feedback deployment distribution.

Where it hooks in (from selfcond.py's verified analysis of utils.py forward_train): the history
the model attends to is precisely `action_dict["latent"]` (the clean action key/value tokens),
DISJOINT from `noisy_latents`/`targets` (which produce the loss). `_add_noise_stream` computes
noisy_latents/targets from GT action BEFORE returning `latent`, and actions use noisy_cond_prob=0.0,
so overwriting `action_dict["latent"]` after the call swaps only the history -- the supervised
objective stays GT action. This mirrors training_loss_from_streams exactly except for that one line
(and the gripper-weight hook, kept in sync with the core edit).
"""
from __future__ import annotations

import torch


def build_state_conditioning(policy, state_raw, q01, q99, actions_mask):
    """Measured state [B, T, n_used] -> normalized, scattered history tensor [B, action_dim, F, apf, 1].

    Reshaped IDENTICALLY to how `_build_training_streams` reshapes the GT action, so the state token
    at (frame f, substep n) aligns with the action target at (f, n).
    """
    cfg = policy.config
    device = cfg.device
    used = cfg.used_action_channel_ids
    apf, fc = cfg.action_per_frame, cfg.frame_chunk_size

    st = state_raw.to(device, torch.float32)  # [B, T>=fc*apf, n_used]
    q01 = q01.to(device, torch.float32)
    q99 = q99.to(device, torch.float32)
    st_norm = 2 * (st - q01) / (q99 - q01).clamp_min(1e-6) - 1  # same map as common.normalize_action
    b = st_norm.shape[0]
    st_norm = st_norm[:, : fc * apf].reshape(b, fc, apf, len(used)).permute(0, 3, 1, 2)  # [B,n_used,F,apf]
    full = st_norm.new_zeros(b, cfg.action_dim, fc, apf)
    full[:, torch.as_tensor(used, device=device)] = st_norm
    state_cond = full.unsqueeze(-1).to(policy.dtype)  # [B, action_dim, F, apf, 1]
    return state_cond * actions_mask


def training_loss_state_cond(policy, batch, q01, q99, return_state_stats=False):
    """Drop-in for policy.forward(batch), but with STATE-conditioned action history.

    batch["action"] must already be normalized (target); batch["observation.state"] is raw
    (normalized here) and must carry the same chunk_size timesteps as the action.
    """
    if policy.config.attn_mode != "flex":
        raise ValueError("state-conditioned training requires attn_mode='flex'")
    policy._ensure_frozen_modules()
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

    # Gripper (per-channel) loss weighting -- kept in sync with training_loss_from_streams.
    cw = policy.config.used_action_channel_weights
    if cw is not None:
        wt = torch.ones(policy.config.action_dim, device=actions.device, dtype=actions.dtype)
        idx = torch.as_tensor(policy.config.used_action_channel_ids, device=actions.device)
        wt[idx] = torch.as_tensor(cw, device=actions.device, dtype=actions.dtype)
        action_dict["actions_loss_weight"] = (actions_mask > 0).to(actions.dtype) * wt.view(1, -1, 1, 1, 1)

    # *** THE FIX: swap the action-history conditioning to measured STATE (target stays GT action). ***
    gt_latent = action_dict["latent"]
    state_cond = build_state_conditioning(policy, batch["observation.state"], q01, q99, actions_mask)
    assert state_cond.shape == gt_latent.shape, f"state {tuple(state_cond.shape)} vs action {tuple(gt_latent.shape)}"
    action_dict["latent"] = state_cond

    input_dict = {
        "latent_dict": latent_dict,
        "action_dict": action_dict,
        "chunk_size": int(torch.randint(1, 5, (1,)).item()),
        "window_size": int(torch.randint(4, 65, (1,)).item()),
    }
    pred = policy.transformer(input_dict, train_mode=True)
    latent_loss, action_loss = policy._flow_matching_loss(input_dict, pred)
    loss = latent_loss + action_loss
    metrics = {"latent_loss": latent_loss.item(), "action_loss": action_loss.item()}
    if return_state_stats:
        # How far the swapped history is from the GT-action history (per channel), for sanity.
        with torch.no_grad():
            diff = (state_cond - gt_latent).abs().mean(dim=(0, 2, 3, 4))  # [action_dim]
            metrics["state_vs_action_hist_l1"] = diff.detach().float().cpu()
    return loss, metrics
