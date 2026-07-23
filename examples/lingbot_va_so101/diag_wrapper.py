#!/usr/bin/env python
"""Run scripts/inference/diagnose_lingbot_va_actions.py's exact rollout/diagnosis logic
against our own step_2000 checkpoint, which has no config.json (only adapter_config.json --
our training loop never called PreTrainedConfig.save_pretrained separately), so that
script's PreTrainedConfig.from_pretrained(ckpt) loading assumption doesn't fit. Builds the
config the way our own eval_checkpoint.py does instead, then reuses its
teacher_forced_rollout() + diagnostic printouts unmodified.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/home/rxhuang/Projects/lerobot/scripts/inference")

import argparse

import torch
from diagnose_lingbot_va_actions import ACTION_NAMES, denormalize_action, teacher_forced_rollout

from common import build_config, make_ds_meta
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", default="/data/rxhuang/lingbot_va_runs/full_20k/checkpoints/step_2000")
p.add_argument("--episode", type=int, default=95)
p.add_argument("--self-feedback", action="store_true")
p.add_argument("--json", default=None)
p.add_argument("--plot", default=None, help="If set, save a predicted-vs-GT action plot to this path.")
p.add_argument("--plot_title", default=None)
args = p.parse_args()

config = build_config(attn_mode="torch")
ds_meta = make_ds_meta()
q01 = torch.tensor(ds_meta.stats["action"]["q01"], dtype=torch.float32)
q99 = torch.tensor(ds_meta.stats["action"]["q99"], dtype=torch.float32)

base_policy = make_policy(config, ds_meta=ds_meta)
from peft import PeftModel

policy = PeftModel.from_pretrained(base_policy, args.checkpoint, is_trainable=False).get_base_model()
policy.eval()

ds = LeRobotDataset("three_cubes_1", root="/data/rxhuang/three_cubes_1", episodes=[args.episode])
pred_raw, gt_raw, pred_norm, bounds = teacher_forced_rollout(
    policy, config, ds, q01, q99, "cuda", self_feedback=args.self_feedback
)

mode = "self-feedback (online-like)" if args.self_feedback else "teacher-forced (ground-truth actions)"
print(f"\nAction feedback mode: {mode}")
mid = ((q01 + q99) / 2).cpu()
baseline = mid.expand_as(gt_raw)
model_mse = ((pred_raw - gt_raw) ** 2).mean(dim=0)
base_mse = ((baseline - gt_raw) ** 2).mean(dim=0)

print(f"\nEpisode {args.episode}: {pred_raw.shape[0]} predicted steps\n")
rows = []
for i, name in enumerate(ACTION_NAMES):
    r = {
        "joint": name, "pred_std": pred_raw[:, i].std().item(), "gt_std": gt_raw[:, i].std().item(),
        "model_mse": model_mse[i].item(), "midpoint_mse": base_mse[i].item(),
    }
    rows.append(r)
    verdict = "beats base" if r["model_mse"] < r["midpoint_mse"] else "WORSE"
    print(f"{name:<15} pred_std={r['pred_std']:>7.2f} gt_std={r['gt_std']:>7.2f} "
          f"model_mse={r['model_mse']:>9.2f} midpoint_mse={r['midpoint_mse']:>9.2f} {verdict}")

offs, per_chunk = 0, []
for ci, n in enumerate(bounds):
    pc, gc = pred_raw[offs : offs + n], gt_raw[offs : offs + n]
    per_chunk.append({
        "pred_travel": float((pc.max(dim=0).values - pc.min(dim=0).values).sum()),
        "gt_travel": float((gc.max(dim=0).values - gc.min(dim=0).values).sum()),
        "rmse": float(((pc - gc) ** 2).mean().sqrt()),
    })
    offs += n
q = max(1, len(per_chunk) // 4)
print("\nDrift over the rollout (quarters):")
for qi in range(0, len(per_chunk), q):
    seg = per_chunk[qi : qi + q]
    if not seg:
        continue
    tr = sum(s["pred_travel"] for s in seg) / max(sum(s["gt_travel"] for s in seg), 1e-6)
    rm = sum(s["rmse"] for s in seg) / len(seg)
    print(f"chunks {qi}-{qi+len(seg)-1}: travel_ratio={tr*100:.1f}% rmse={rm:.2f}deg")

amp = float(torch.tensor([r["pred_std"] for r in rows]).sum() / torch.tensor([r["gt_std"] for r in rows]).sum())
n_worse = sum(1 for r in rows if r["model_mse"] >= r["midpoint_mse"])
print(f"\nPredicted motion amplitude vs ground truth: {amp * 100:.1f}%")
print(f"Joints where the model does not beat 'always predict the midpoint': {n_worse}/{len(rows)}")

if args.plot:
    from plotting import plot_action_compare

    title = args.plot_title or f"{mode} | checkpoint={Path(args.checkpoint).name} | amplitude={amp * 100:.1f}%"
    plot_action_compare(pred_raw, gt_raw, Path(args.plot), title=title)
    print(f"\nPlot saved to {args.plot}")
