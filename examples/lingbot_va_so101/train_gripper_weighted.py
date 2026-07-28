#!/usr/bin/env python
"""Post-train #2: gripper-weighted loss + stronger LoRA (r=64) + fully-trained action heads.

Motivation (docs/closed_loop_verification.md): with the state-feedback inference fix the arm is
essentially solved (mse 85, amp 91% through select_action), but the GRIPPER (channel 28) still
fails to fully close during grasp -> likely the real "can't complete the task" cause. This run:
  - up-weights the gripper channel in the action flow-matching loss
    (config.used_action_channel_weights), and
  - trains action_embedder / action_proj_out at FULL rank (PEFT modules_to_save) on top of LoRA
    r=64 all-linear, since the action input/output layers carry embodiment-specific action
    semantics that low-rank adaptation struggles to move.
Logs per-dim action loss so we can actually watch the gripper learn. Reuses the existing
(correct) manual quantile normalization. Eval is done separately with verify_closed_loop.py
(state-feedback closed loop) on saved checkpoints.
"""
import argparse
import csv
import time
from collections import deque
from pathlib import Path

import torch

from common import (
    ACTION_NAMES, RUNS_ROOT, action_quantiles, attach_text_embed_cache, build_config,
    make_dataset, make_ds_meta, normalize_action, TRAIN_EPISODES,
)
from lerobot.policies.factory import make_policy

GRIPPER_IDX_IN_USED = 5  # used_action_channel_ids = [14,15,16,17,18,28]; gripper is the 6th (index 5)


def cycle(loader):
    while True:
        yield from loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", default="gripper_w5_r64_heads")
    p.add_argument("--num_steps", type=int, default=15000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=1000)
    p.add_argument("--gripper_weight", type=float, default=5.0)
    p.add_argument("--lora_r", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    p.add_argument("--lr_lora", type=float, default=1e-4)
    p.add_argument("--lr_full", type=float, default=1e-5)
    args = p.parse_args()

    run_dir = RUNS_ROOT / args.run_id
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    weights = [1.0, 1.0, 1.0, 1.0, 1.0, args.gripper_weight]  # 5 arm joints + gripper
    config = build_config(used_action_channel_weights=weights)
    assert len(weights) == len(config.used_action_channel_ids)
    print(f"used_action_channel_ids={config.used_action_channel_ids} weights={weights}", flush=True)

    ds_meta = make_ds_meta()
    q01, q99 = action_quantiles(ds_meta)

    train_ds = make_dataset(config, episodes=TRAIN_EPISODES)
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

    # --- sanity: which params are trainable, and are the action heads among them? ---
    lora_params, full_params, other_params = [], [], []
    full_names, saw_embedder, saw_projout = [], False, False
    for n, prm in policy.named_parameters():
        if not prm.requires_grad:
            continue
        if "lora_" in n:
            lora_params.append(prm)
        elif "modules_to_save" in n:
            full_params.append(prm)
            full_names.append(n)
            saw_embedder |= "action_embedder" in n
            saw_projout |= "action_proj_out" in n
        else:
            other_params.append(prm)
    n_lora = sum(p.numel() for p in lora_params)
    n_full = sum(p.numel() for p in full_params)
    n_other = sum(p.numel() for p in other_params)
    n_total = sum(p.numel() for p in policy.parameters())
    print(f"trainable: lora={n_lora:,} full={n_full:,} other={n_other:,} "
          f"/ total={n_total:,} ({100*(n_lora+n_full+n_other)/n_total:.3f}%)", flush=True)
    print(f"full-trained modules ({len(full_names)} tensors): action_embedder={saw_embedder} "
          f"action_proj_out={saw_projout}", flush=True)
    for nm in full_names:
        print(f"    full: {nm}", flush=True)
    assert saw_embedder and saw_projout, "action_embedder/action_proj_out NOT in trainable modules_to_save!"
    assert not other_params, f"unexpected trainable params outside lora/modules_to_save: {n_other}"

    param_groups = [
        {"params": lora_params, "lr": args.lr_lora},
        {"params": full_params, "lr": args.lr_full},
    ]
    optimizer = torch.optim.AdamW(
        param_groups, betas=config.optimizer_betas, eps=config.optimizer_eps,
        weight_decay=config.optimizer_weight_decay,
    )
    scheduler = config.get_scheduler_preset().build(optimizer, args.num_steps)
    grad_clip_norm = config.optimizer_grad_clip_norm

    csv_path = run_dir / "train_log.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["step", "latent_loss", "action_loss", "total_loss", "lr_lora", "lr_full", "grad_norm", "step_time_s"]
        + [f"aloss_{nm}" for nm in ACTION_NAMES]
    )

    data_iter = cycle(loader)
    recent = deque(maxlen=200)
    used_ids = config.used_action_channel_ids
    for step in range(1, args.num_steps + 1):
        t0 = time.time()
        batch = next(data_iter)
        batch = {k: (v.to(config.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        batch["action"] = normalize_action(batch["action"], q01, q99)

        loss, metrics = policy.forward(batch)
        if not torch.isfinite(loss):
            print(f"[step {step}] NON-FINITE loss {loss.item()}; stopping.", flush=True)
            break
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for g in param_groups for p in g["params"]], grad_clip_norm
        )
        optimizer.step()
        scheduler.step()
        step_time = time.time() - t0
        recent.append(metrics["action_loss"])

        # per-dim action loss (unweighted) stashed by _flow_matching_loss, indexed to used channels
        per_ch = getattr(policy, "_last_per_ch_action_loss", None)
        per_used = [float(per_ch[c]) for c in used_ids] if per_ch is not None else [float("nan")] * len(used_ids)

        lr_lora = optimizer.param_groups[0]["lr"]
        lr_full = optimizer.param_groups[1]["lr"]
        csv_writer.writerow(
            [step, metrics["latent_loss"], metrics["action_loss"], loss.item(), lr_lora, lr_full,
             grad_norm.item(), step_time] + per_used
        )
        csv_file.flush()

        if step % args.log_every == 0 or step == 1:
            grip = per_used[GRIPPER_IDX_IN_USED]
            arm_mean = sum(per_used[:5]) / 5
            print(
                f"step={step} latent={metrics['latent_loss']:.4f} action={metrics['action_loss']:.4f} "
                f"total={loss.item():.4f} | aloss arm_mean={arm_mean:.4f} GRIPPER={grip:.4f} "
                f"| gnorm={grad_norm.item():.2f} {step_time:.2f}s",
                flush=True,
            )

        if step % args.save_every == 0 or step == args.num_steps:
            ckpt_dir = run_dir / "checkpoints" / f"step_{step}"
            policy.save_pretrained(ckpt_dir)
            print(f"[step {step}] checkpoint saved to {ckpt_dir}", flush=True)

    csv_file.close()
    print(f"Training complete. Run dir: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
