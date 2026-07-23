#!/usr/bin/env python
"""Step 2: post-train lingbot_va on three_cubes_1 (SO-101), joint-space channels.

Logging: CSV always (source of truth); wandb if available (this env has it + a valid
~/.netrc token). Periodic teacher-forced open-loop eval + per-dim action-compare plots.
See NOTES.md for the Step 0/1 findings this builds on.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import torch

from common import RUNS_ROOT, action_quantiles, attach_text_embed_cache, build_config, make_dataset, make_ds_meta, normalize_action
from lerobot.policies.factory import make_policy
from selfcond import SelfCondConfig, training_loss_selfcond

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def cycle(loader):
    while True:
        yield from loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", required=True)
    p.add_argument("--num_steps", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=100)
    p.add_argument("--eval_episodes", type=int, nargs="+", default=[95])
    p.add_argument("--no_wandb", action="store_true")
    p.add_argument(
        "--peft_target_modules",
        default="all-linear",
        help="lingbot_va has no policy-specific PEFT defaults (checked: no "
        "_get_default_peft_targets override), so this must be passed explicitly or "
        "wrap_with_peft raises. 'all-linear' is PEFT's built-in catch-all.",
    )
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument(
        "--early_stop_loss",
        type=float,
        default=None,
        help="Stop early once the rolling mean of the last --early_stop_window total_loss "
        "values drops below this. Uses a rolling mean (not a single step) because "
        "batch_size=1 single-episode batches are noisy -- e.g. the 500-step smoke run saw "
        "per-step total_loss swing from 0.03 to 9+ between adjacent steps, so a raw "
        "per-step threshold would trigger (or fail to trigger) on noise rather than "
        "actual convergence.",
    )
    p.add_argument("--early_stop_window", type=int, default=50)
    p.add_argument(
        "--eval_gpu",
        type=int,
        default=None,
        help="CUDA_VISIBLE_DEVICES for the eval subprocess. Must differ from the training "
        "GPU: eval needs attn_mode='torch' (predict_action_chunk is incompatible with "
        "the training-only attn_mode='flex' class-level block mask -- confirmed by a "
        "'block_mask was created for ... but got q_len=192' crash) and a second full "
        "base-model copy alongside training would not fit one 24GB GPU. Required unless "
        "--eval_every > --num_steps (i.e. eval never runs).",
    )
    p.add_argument(
        "--selfcond-p-max", type=float, default=0.0,
        help="Max probability (after ramp) of replacing a chunk's action-history "
        "conditioning with the model's own no-grad x0 self-prediction. 0 (default) "
        "reproduces the original training path exactly -- see selfcond.py's docstring "
        "for why that's guaranteed bit-identical, not just 'should be similar'.",
    )
    p.add_argument("--selfcond-ramp-frac", type=float, default=0.4)
    p.add_argument("--selfcond-x0-t", type=float, default=0.5)
    p.add_argument("--keep-recent-clean-prob", type=float, default=0.5)
    p.add_argument(
        "--action-history-noise", action="store_true",
        help="Ablation arm, mutually exclusive with --selfcond-p-max: perturb action "
        "history with structured noise instead of a learned self-prediction.",
    )
    p.add_argument("--action-history-noise-prob", type=float, default=0.4)
    args = p.parse_args()
    if args.eval_gpu is None and args.eval_every <= args.num_steps:
        p.error("--eval_gpu is required whenever eval will run (see --help for why)")
    if args.action_history_noise and args.selfcond_p_max > 0:
        p.error("--action-history-noise and --selfcond-p-max are mutually exclusive")
    selfcond_cfg = SelfCondConfig(
        p_max=args.selfcond_p_max,
        ramp_frac=args.selfcond_ramp_frac,
        x0_t=args.selfcond_x0_t,
        keep_recent_clean_prob=args.keep_recent_clean_prob,
        action_history_noise=args.action_history_noise,
        action_history_noise_prob=args.action_history_noise_prob,
    )
    selfcond_active = args.selfcond_p_max > 0 or args.action_history_noise

    run_dir = RUNS_ROOT / args.run_id
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)

    config = build_config()
    ds_meta = make_ds_meta()
    q01, q99 = action_quantiles(ds_meta)

    from common import TRAIN_EPISODES

    train_ds = make_dataset(config, episodes=TRAIN_EPISODES)
    loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True
    )

    policy = make_policy(config, ds_meta=ds_meta)
    # Must patch the *pre-PEFT* object: PeftModel.__getattr__ delegates *attribute reads*
    # for missing names to the wrapped base model, but `self` inside the base model's own
    # methods (e.g. _build_training_streams calling self._get_t5_prompt_embeds) is always
    # the inner object, not the PeftModel wrapper -- so assigning the cache onto the
    # wrapper after wrapping would silently never be hit. Attach before wrapping instead;
    # PEFT holds a reference to (not a copy of) this same object, so the patch persists.
    attach_text_embed_cache(policy)
    # Full-parameter AdamW on the ~5B transformer does not fit a single 24GB GPU (README's
    # own warning, and matches what we saw OOM even across 6 GPUs in the other lingbot-va
    # codebase this session) -- LoRA via PEFT per the documented `wrap_with_peft` mechanism.
    policy = policy.wrap_with_peft(
        peft_cli_overrides={"target_modules": args.peft_target_modules, "r": args.lora_r}
    )
    policy.train()

    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in policy.parameters())
    print(f"PEFT-wrapped: {n_trainable:,}/{n_total:,} trainable ({100 * n_trainable / n_total:.2f}%)", flush=True)

    optimizer = config.get_optimizer_preset().build(trainable_params)
    scheduler_cfg = config.get_scheduler_preset()
    scheduler = scheduler_cfg.build(optimizer, args.num_steps) if scheduler_cfg is not None else None
    grad_clip_norm = config.optimizer_grad_clip_norm

    use_wandb = HAS_WANDB and not args.no_wandb
    if use_wandb:
        wandb.init(project="lingbot_va_so101", name=args.run_id, config=vars(args))

    csv_path = run_dir / "train_log.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["step", "latent_loss", "action_loss", "total_loss", "lr", "grad_norm", "step_time_s",
         "selfcond_p", "selfcond_replace_rate", "selfcond_a0_gt_l2"]
    )

    data_iter = cycle(loader)
    recent_losses = deque(maxlen=args.early_stop_window)
    recent_a0_l2_ratios = deque(maxlen=5)  # last 5 logged points = last ~500 steps
    for step in range(1, args.num_steps + 1):
        t0 = time.time()
        batch = next(data_iter)
        batch = {k: (v.to(config.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        batch["action"] = normalize_action(batch["action"], q01, q99)

        if selfcond_active:
            loss, metrics = training_loss_selfcond(policy, batch, step, args.num_steps, selfcond_cfg)
        else:
            loss, metrics = policy.forward(batch)
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip_norm)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        step_time = time.time() - t0

        row = [
            step, metrics["latent_loss"], metrics["action_loss"], loss.item(), lr, grad_norm.item(), step_time,
            metrics.get("selfcond_p", 0.0), metrics.get("selfcond_replace_rate", 0.0),
            metrics.get("selfcond_a0_gt_l2"),
        ]
        csv_writer.writerow(row)
        csv_file.flush()
        recent_losses.append(loss.item())

        if step % args.log_every == 0 or step == 1:
            print(
                f"step={step} latent_loss={metrics['latent_loss']:.4f} action_loss={metrics['action_loss']:.4f} "
                f"total_loss={loss.item():.4f} lr={lr:.2e} grad_norm={grad_norm.item():.3f} "
                f"step_time={step_time:.1f}s",
                flush=True,
            )
            if use_wandb:
                wandb.log(
                    {
                        "latent_loss": metrics["latent_loss"],
                        "action_loss": metrics["action_loss"],
                        "total_loss": loss.item(),
                        "lr": lr,
                        "grad_norm": grad_norm.item(),
                        "step_time_s": step_time,
                    },
                    step=step,
                )

        if selfcond_active and step % 100 == 0:
            l2 = metrics.get("selfcond_a0_gt_l2")
            print(
                f"[step {step}] selfcond: p={metrics.get('selfcond_p', 0.0):.3f} "
                f"replace_rate={metrics.get('selfcond_replace_rate', 0.0):.3f} "
                f"a0_vs_gt_l2={l2 if l2 is None else f'{l2:.3f}'}",
                flush=True,
            )
            gt_step_diff = metrics.get("selfcond_gt_step_diff")
            if l2 is not None and gt_step_diff:
                recent_a0_l2_ratios.append(l2 / gt_step_diff)
                if len(recent_a0_l2_ratios) == recent_a0_l2_ratios.maxlen and min(recent_a0_l2_ratios) > 2.0:
                    print(
                        f"[step {step}] WARNING: self-cond a0 predictions have stayed >2x the "
                        f"GT step-to-step action difference for the last {recent_a0_l2_ratios.maxlen} "
                        f"logged points (ratios={[f'{r:.2f}' for r in recent_a0_l2_ratios]}) -- Pass 1 "
                        f"quality looks poor; consider a longer --selfcond-ramp-frac.",
                        flush=True,
                    )

        if step % args.save_every == 0 or step == args.num_steps:
            ckpt_dir = run_dir / "checkpoints" / f"step_{step}"
            policy.save_pretrained(ckpt_dir)
            print(f"[step {step}] checkpoint saved to {ckpt_dir}", flush=True)

        if step % args.eval_every == 0 or step == args.num_steps:
            # Eval needs its own process/GPU (attn_mode="torch" vs training's "flex", and a
            # second base-model copy wouldn't fit alongside training on one GPU -- see
            # --eval_gpu's help text). Force a checkpoint at this step if save_every didn't
            # already land here, then launch eval_checkpoint.py non-blocking so training
            # doesn't stall waiting on it.
            ckpt_dir = run_dir / "checkpoints" / f"step_{step}"
            if not ckpt_dir.exists():
                policy.save_pretrained(ckpt_dir)
                print(f"[step {step}] checkpoint saved to {ckpt_dir} (forced for eval)", flush=True)
            eval_log = open(run_dir / "logs" / f"eval_step{step}.log", "w")
            subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).parent / "eval_checkpoint.py"),
                    "--run_id", args.run_id,
                    "--step", str(step),
                    "--checkpoint_dir", str(ckpt_dir),
                    "--episodes", *map(str, args.eval_episodes),
                ],
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(args.eval_gpu)},
                stdout=eval_log,
                stderr=subprocess.STDOUT,
            )
            print(f"[step {step}] launched async eval on GPU {args.eval_gpu} (log: {eval_log.name})", flush=True)

        if (
            args.early_stop_loss is not None
            and len(recent_losses) == args.early_stop_window
            and (rolling_mean := sum(recent_losses) / len(recent_losses)) < args.early_stop_loss
        ):
            print(
                f"[step {step}] EARLY STOP: rolling mean total_loss over last "
                f"{args.early_stop_window} steps = {rolling_mean:.4f} < {args.early_stop_loss}",
                flush=True,
            )
            ckpt_dir = run_dir / "checkpoints" / f"step_{step}"
            if not ckpt_dir.exists():
                policy.save_pretrained(ckpt_dir)
                print(f"[step {step}] checkpoint saved to {ckpt_dir} (final, early stop)", flush=True)
            if args.eval_gpu is not None:
                eval_log = open(run_dir / "logs" / f"eval_step{step}.log", "w")
                subprocess.Popen(
                    [
                        sys.executable,
                        str(Path(__file__).parent / "eval_checkpoint.py"),
                        "--run_id", args.run_id,
                        "--step", str(step),
                        "--checkpoint_dir", str(ckpt_dir),
                        "--episodes", *map(str, args.eval_episodes),
                    ],
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(args.eval_gpu)},
                    stdout=eval_log,
                    stderr=subprocess.STDOUT,
                )
                print(f"[step {step}] launched final async eval on GPU {args.eval_gpu}", flush=True)
            break

    csv_file.close()
    if use_wandb:
        wandb.finish()
    print(f"Training complete. Run dir: {run_dir}")


if __name__ == "__main__":
    main()
