#!/usr/bin/env python
"""Unit test: --selfcond-p-max 0 must give bit-identical loss to the unmodified training path.

Also runs a 50-step smoke test with self-conditioning genuinely active (p_max>0) checking
for NaN and peak GPU memory. The spec's "<100GB" budget was written for a different (Thor,
122GB unified memory) machine; this session's actual hardware is 24GB-per-GPU, so the smoke
test instead asserts peak memory stays within a single 24GB GPU with headroom.
"""
import torch

from common import attach_text_embed_cache, build_config, make_dataset, make_ds_meta, normalize_action, action_quantiles
from lerobot.policies.factory import make_policy
from selfcond import SelfCondConfig, training_loss_selfcond


def build_policy_and_batch():
    config = build_config()
    ds_meta = make_ds_meta()
    q01, q99 = action_quantiles(ds_meta)
    from common import TRAIN_EPISODES

    ds = make_dataset(config, episodes=TRAIN_EPISODES[:5])
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
    batch = next(iter(loader))
    batch = {k: (v.to(config.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    batch["action"] = normalize_action(batch["action"], q01, q99)

    policy = make_policy(config, ds_meta=ds_meta)
    attach_text_embed_cache(policy)
    policy = policy.wrap_with_peft(peft_cli_overrides={"target_modules": "all-linear", "r": 16})
    policy.train()
    return policy, batch


def test_identity():
    policy, batch = build_policy_and_batch()
    cfg_off = SelfCondConfig(p_max=0.0)

    # Values only, not gradients -- both calls under no_grad so their activation graphs
    # don't both stay resident at once (that OOM'd on the first attempt at this test).
    with torch.no_grad():
        torch.manual_seed(0)
        loss_a, metrics_a = policy.forward(batch)

        torch.manual_seed(0)
        loss_b, metrics_b = training_loss_selfcond(policy, batch, step=1, num_steps=100, cfg=cfg_off)

    diff = (loss_a - loss_b).abs().item()
    print(f"loss_original={loss_a.item():.10f} loss_wrapper_p0={loss_b.item():.10f} diff={diff:.2e}")
    print(f"latent_loss diff={abs(metrics_a['latent_loss'] - metrics_b['latent_loss']):.2e}")
    print(f"action_loss diff={abs(metrics_a['action_loss'] - metrics_b['action_loss']):.2e}")
    assert diff < 1e-6, f"IDENTITY TEST FAILED: loss diff {diff} >= 1e-6"
    print("IDENTITY TEST PASSED (selfcond_p_max=0 matches unmodified path within 1e-6)")


def test_smoke_active():
    policy, batch = build_policy_and_batch()
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-5)
    cfg_on = SelfCondConfig(p_max=0.5, ramp_frac=0.4, x0_t=0.5, keep_recent_clean_prob=0.5)

    torch.cuda.reset_peak_memory_stats()
    for step in range(1, 51):
        loss, metrics = training_loss_selfcond(policy, batch, step=step, num_steps=50, cfg=cfg_on)
        assert torch.isfinite(loss).all(), f"NaN/Inf loss at step {step}"
        optimizer.zero_grad()
        loss.backward()
        for p in trainable_params:
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), f"NaN/Inf grad at step {step}"
        optimizer.step()
        if step % 10 == 0 or step == 1:
            print(
                f"step={step} loss={loss.item():.4f} selfcond_p={metrics['selfcond_p']:.3f} "
                f"replace_rate={metrics['selfcond_replace_rate']:.3f} a0_gt_l2={metrics['selfcond_a0_gt_l2']}"
            )
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"50-step smoke test PASSED. Peak GPU memory allocated: {peak_gb:.2f} GB")
    # Measured ~23.7GB on a 24GB GPU: Pass 1's extra forward (even under no_grad) roughly
    # doubles peak memory pressure vs. the unmodified single-pass path. That means real
    # training runs need a genuinely idle GPU on this shared cluster, not just "some room" --
    # unlike a few of this session's earlier (single-pass) runs that tolerated a little
    # contention. 23.9GB is the actual ceiling to watch for, not a bug to fix.
    assert peak_gb < 23.9, f"Peak memory {peak_gb:.2f}GB exceeds a 24GB GPU's real ceiling"


if __name__ == "__main__":
    test_identity()
    test_smoke_active()
