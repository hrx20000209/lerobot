#!/usr/bin/env python
"""Post-train #3: STATE-conditioned action history (the gripper fix).

Same recipe as train_gripper_weighted.py (gripper loss weight 5.0, LoRA r=64 all-linear +
action_embedder/action_proj_out fully trained) BUT the action-history conditioning is swapped to
measured proprioception state, matching the state-feedback deployment (see state_cond.py). This is
the principled fix for the gripper train/deploy mismatch; the arm is unaffected (action~=state).
"""
import argparse
import csv
import time
from collections import deque
from pathlib import Path

import torch

from common import (
    ACTION_NAMES, DATASET_ROOT, FPS, RUNS_ROOT, action_quantiles, attach_text_embed_cache,
    build_config, make_ds_meta, normalize_action, TRAIN_EPISODES,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from state_cond import training_loss_state_cond

GRIPPER_IDX_IN_USED = 5


def make_dataset_with_state(config, episodes):
    """Like common.make_dataset but also loads observation.state on the action timesteps."""
    dt = {k: [i / FPS for i in config.observation_delta_indices] for k in config.obs_cam_keys}
    dt["action"] = [i / FPS for i in config.action_delta_indices]
    dt["observation.state"] = [i / FPS for i in config.action_delta_indices]  # same 16 steps as action
    return LeRobotDataset("three_cubes_1", root=DATASET_ROOT, delta_timestamps=dt, episodes=episodes)


def cycle(loader):
    while True:
        yield from loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", default="state_cond_w5_r64_heads")
    p.add_argument("--num_steps", type=int, default=15000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--gripper_weight", type=float, default=5.0)
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lr_lora", type=float, default=1e-4)
    p.add_argument("--lr_full", type=float, default=1e-5)
    args = p.parse_args()

    run_dir = RUNS_ROOT / args.run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    weights = [1.0, 1.0, 1.0, 1.0, 1.0, args.gripper_weight]
    config = build_config(used_action_channel_weights=weights)
    print(f"used_action_channel_ids={config.used_action_channel_ids} weights={weights}", flush=True)

    ds_meta = make_ds_meta()
    q01, q99 = action_quantiles(ds_meta)

    train_ds = make_dataset_with_state(config, episodes=TRAIN_EPISODES)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True
    )

    policy = make_policy(config, ds_meta=ds_meta)
    attach_text_embed_cache(policy)
    policy = policy.wrap_with_peft(
        peft_cli_overrides={
            "target_modules": "all-linear",
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "modules_to_save": ["action_embedder", "action_proj_out"],
        }
    )
    policy.train()

    lora_params, full_params, other_params = [], [], []
    saw_embedder = saw_projout = False
    for n, prm in policy.named_parameters():
        if not prm.requires_grad:
            continue
        if "lora_" in n:
            lora_params.append(prm)
        elif "modules_to_save" in n:
            full_params.append(prm)
            saw_embedder |= "action_embedder" in n
            saw_projout |= "action_proj_out" in n
        else:
            other_params.append(prm)
    print(f"trainable: lora={sum(p.numel() for p in lora_params):,} "
          f"full={sum(p.numel() for p in full_params):,} other={sum(p.numel() for p in other_params):,}",
          flush=True)
    assert saw_embedder and saw_projout, "action heads not fully trainable!"
    assert not other_params, "unexpected trainable params outside lora/modules_to_save"

    optimizer = torch.optim.AdamW(
        [{"params": lora_params, "lr": args.lr_lora}, {"params": full_params, "lr": args.lr_full}],
        betas=config.optimizer_betas, eps=config.optimizer_eps, weight_decay=config.optimizer_weight_decay,
    )
    scheduler = config.get_scheduler_preset().build(optimizer, args.num_steps)
    grad_clip_norm = config.optimizer_grad_clip_norm
    trainable = lora_params + full_params

    csv_file = open(run_dir / "train_log.csv", "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["step", "latent_loss", "action_loss", "total_loss", "lr_lora", "lr_full", "grad_norm", "step_time_s"]
        + [f"aloss_{nm}" for nm in ACTION_NAMES]
    )

    data_iter = cycle(loader)
    used_ids = config.used_action_channel_ids
    for step in range(1, args.num_steps + 1):
        t0 = time.time()
        batch = next(data_iter)
        batch = {k: (v.to(config.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        batch["action"] = normalize_action(batch["action"], q01, q99)  # target (state stays raw for state_cond)

        loss, metrics = training_loss_state_cond(
            policy, batch, q01, q99, return_state_stats=(step == 1)
        )
        if step == 1:
            svh = metrics.get("state_vs_action_hist_l1")
            if svh is not None:
                print("state-vs-action history |L1| per used channel (normalized): "
                      + ", ".join(f"{nm}={float(svh[c]):.3f}" for nm, c in zip(ACTION_NAMES, used_ids)), flush=True)
                print("  (arm ~0 => action~=state; GRIPPER large => the mismatch we are fixing)", flush=True)
        if not torch.isfinite(loss):
            print(f"[step {step}] NON-FINITE loss; stopping.", flush=True)
            break
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, grad_clip_norm)
        optimizer.step()
        scheduler.step()
        step_time = time.time() - t0

        per_ch = getattr(policy, "_last_per_ch_action_loss", None)
        per_used = [float(per_ch[c]) for c in used_ids] if per_ch is not None else [float("nan")] * len(used_ids)
        csv_writer.writerow(
            [step, metrics["latent_loss"], metrics["action_loss"], loss.item(),
             optimizer.param_groups[0]["lr"], optimizer.param_groups[1]["lr"], grad_norm.item(), step_time]
            + per_used
        )
        csv_file.flush()

        if step % args.log_every == 0 or step == 1:
            grip = per_used[GRIPPER_IDX_IN_USED]
            arm_mean = sum(per_used[:5]) / 5
            print(f"step={step} latent={metrics['latent_loss']:.4f} action={metrics['action_loss']:.4f} "
                  f"total={loss.item():.4f} | aloss arm_mean={arm_mean:.4f} GRIPPER={grip:.4f} "
                  f"| gnorm={grad_norm.item():.2f} {step_time:.2f}s", flush=True)

        if step % args.save_every == 0 or step == args.num_steps:
            ckpt_dir = run_dir / "checkpoints" / f"step_{step}"
            policy.save_pretrained(ckpt_dir)
            print(f"[step {step}] checkpoint saved to {ckpt_dir}", flush=True)

    csv_file.close()
    print(f"Training complete. Run dir: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
