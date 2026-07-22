#!/usr/bin/env python
"""Standalone eval: teacher-forced open-loop action-compare plot for one checkpoint.

Deliberately a separate process/GPU from training. Two independent reasons force this
split (not just convenience):

1. attn_mode: LingBot-VA's FlexAttnFunc (attn_mode="flex", required for policy.forward()
   training) keeps its block-causal mask in *class-level* attributes sized for the
   training call's chunk_size/window_size. predict_action_chunk()/_infer() (used here)
   need attn_mode="torch" -- mixing the two on one model instance throws
   "block_mask was created for ... but got q_len=192 and kv_len=192" (confirmed by
   running the dry run in-process before splitting this out).
2. Memory: a second full ~5B-param base model resident alongside the training model
   would likely not fit a single 24GB GPU together with training activations/optimizer
   state (also confirmed empirically -- GPU4's OOM during the first dry-run attempt).

Reads a LoRA adapter checkpoint saved by train_lingbot_va_so101.py (adapter_config.json
points back at the same lerobot/lingbot_va_base this loads fresh with attn_mode="torch").
"""

import argparse
import csv
from pathlib import Path

from peft import PeftModel

from common import RUNS_ROOT, action_quantiles, attach_text_embed_cache, build_config, make_ds_meta
from lerobot.policies.factory import make_policy
from plotting import plot_action_compare
from rollout_eval import per_dim_mse, teacher_forced_rollout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--episodes", type=int, nargs="+", default=[95])
    args = p.parse_args()

    run_dir = RUNS_ROOT / args.run_id
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)

    config = build_config(attn_mode="torch")
    ds_meta = make_ds_meta()
    q01, q99 = action_quantiles(ds_meta)

    base_policy = make_policy(config, ds_meta=ds_meta)
    attach_text_embed_cache(base_policy)  # must patch before PEFT wrap -- see common.py docstring
    policy = PeftModel.from_pretrained(base_policy, args.checkpoint_dir, is_trainable=False)
    policy.eval()
    inner = policy.get_base_model()

    for ep in args.episodes:
        pred_raw, gt_raw = teacher_forced_rollout(inner, config, ds_meta, ep, q01, q99, device=config.device)
        mse = per_dim_mse(pred_raw, gt_raw)
        mean_mse = sum(mse.values()) / len(mse)
        plot_path = run_dir / "plots" / f"action_compare_step{args.step}_ep{ep}.png"
        plot_action_compare(
            pred_raw, gt_raw, plot_path, title=f"step={args.step} episode={ep} mean_mse={mean_mse:.3f}"
        )
        print(f"[step {args.step}] eval ep={ep} per-dim MSE={mse}", flush=True)
        with open(run_dir / "eval_log.csv", "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([args.step, ep, mean_mse] + list(mse.values()))
    print("EVAL DONE", flush=True)


if __name__ == "__main__":
    main()
