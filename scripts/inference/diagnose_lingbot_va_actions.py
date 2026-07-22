#!/usr/bin/env python
"""Diagnose LingBot-VA action amplitude: is the policy actually predicting motion?

Symptom this exists to explain: on the real arm the first chunk snaps the robot hard to
some pose, and after that it barely moves.

The hypothesis that predicts exactly that symptom is "the model outputs ~0 in normalized
space". Actions are unnormalized with QUANTILES,

    raw = (norm + 1) / 2 * (q99 - q01) + q01

so norm == 0 maps to the *midpoint* of each joint's demonstrated range. A policy stuck at
0 therefore commands the mid-range pose forever: one big jump from wherever the arm
started, then nothing.

This runs a teacher-forced open-loop rollout on a held-out episode (real frames and real
executed actions fed back into the KV cache, so closed-loop drift is excluded) and
compares, per joint:

  * predicted vs ground-truth spread (std, p2p) -- is the model moving at all?
  * predicted mean vs the q01/q99 midpoint    -- is it parked at normalized 0?
  * the MSE of the model against the MSE of a constant "always predict the midpoint"
    baseline -- does the adapter beat the trivial answer?

Usage:
    python scripts/inference/diagnose_lingbot_va_actions.py
    python scripts/inference/diagnose_lingbot_va_actions.py --episode 96 --json out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class

ACTION_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
TEMPORAL_DOWNSAMPLE = 4  # Wan2.2 VAE


def normalize_action(a, q01, q99):
    q01, q99 = q01.to(a.device, a.dtype), q99.to(a.device, a.dtype)
    return 2 * (a - q01) / (q99 - q01).clamp_min(1e-6) - 1


def denormalize_action(a, q01, q99):
    q01, q99 = q01.to(a.device, a.dtype), q99.to(a.device, a.dtype)
    return (a + 1) / 2 * (q99 - q01) + q01


def scatter_to_action_dim(used_norm, action_dim, used_ids):
    full = used_norm.new_zeros(list(used_norm.shape[:-1]) + [action_dim])
    full[..., torch.as_tensor(used_ids, device=used_norm.device)] = used_norm
    return full


@torch.no_grad()
def teacher_forced_rollout(policy, config, ds, q01, q99, device, self_feedback=False, state_feedback=False):
    """Returns (pred_raw, gt_raw, pred_norm, chunk_bounds).

    ``self_feedback`` swaps the *actions* fed back into the KV cache from the episode's real
    executed actions to the model's own previous prediction -- what actually happens online.
    Observations stay real either way, so this isolates action-feedback drift from the
    visual drift that only a live robot can produce.
    """
    cam_keys = config.obs_cam_keys
    used_ids = config.used_action_channel_ids
    action_per_frame = config.action_per_frame
    raw_frames_per_chunk = config.frame_chunk_size * TEMPORAL_DOWNSAMPLE
    task = ds[0]["task"]
    n_frames = len(ds)

    def frame_batch(i):
        return {k: ds[i][k].unsqueeze(0).to(device) for k in cam_keys} | {"task": [task]}

    def gt_window(start, length):
        return torch.stack([ds[start + j]["action"] for j in range(length)]).to(device)

    def state_window(start, length):
        """Measured proprioception standing in for the executed action.

        On a position-controlled arm the commanded target at t is very nearly the position
        reached at t+1 (measured on this dataset: r = 0.93-0.997, mean error 0.85-3.75 deg
        against per-joint sigmas of 10-52 deg). A robot always has this; it never has the
        ground-truth action, and its own prediction is what drifts.
        """
        idx = [min(start + j + 1, len(ds) - 1) for j in range(length)]
        return torch.stack([ds[i]["observation.state"] for i in idx]).to(device)

    policy.reset()
    pred_chunks, gt_chunks, bounds = [], [], []

    actions = policy.predict_action_chunk(frame_batch(0))
    pred_chunks.append(actions[0])
    gt_chunks.append(gt_window(1, actions.shape[1]))
    bounds.append(actions.shape[1])
    consumed = 1 + actions.shape[1]

    while consumed + raw_frames_per_chunk <= n_frames:
        start, length = consumed, raw_frames_per_chunk
        obs_buffer = [
            {k: ds[start + j][k].unsqueeze(0).to(device) for k in cam_keys} for j in range(length)
        ]
        policy._obs_buffer = obs_buffer
        if not self_feedback:
            source = state_window(start, length) if state_feedback else gt_window(start, length)
            gt_used_norm = normalize_action(source, q01, q99)
            f = length // action_per_frame
            gt_full = scatter_to_action_dim(gt_used_norm, config.action_dim, used_ids)
            gt_full = (
                gt_full.view(f, action_per_frame, config.action_dim)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .unsqueeze(-1)
            )
            policy._executed_actions = gt_full.to(policy.dtype)
        # else: leave policy._executed_actions as the model's own last chunk (online behaviour)
        actions = policy.predict_action_chunk(None)
        pred_chunks.append(actions[0])
        gt_chunks.append(gt_window(start, actions.shape[1]))
        bounds.append(actions.shape[1])
        consumed = start + actions.shape[1]

    pred_norm = torch.cat(pred_chunks, dim=0)
    gt_raw = torch.cat(gt_chunks, dim=0)
    return denormalize_action(pred_norm, q01, q99).cpu(), gt_raw.cpu(), pred_norm.cpu(), bounds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/home/hrx/Projects/models/three_cubes_1/lingbot_va_async")
    p.add_argument("--dataset-root", default="/home/hrx/Datasets/three_cubes_1")
    p.add_argument("--repo-id", default="three_cubes_1")
    p.add_argument("--episode", type=int, default=95, help="Held-out (training used 0-94).")
    p.add_argument("--video-backend", default="pyav")
    p.add_argument("--json", default=None)
    p.add_argument("--save-traj", default=None, help="Write pred/gt trajectories to this .npz for plotting.")
    p.add_argument(
        "--self-feedback",
        action="store_true",
        help="Feed the model's own actions back instead of the episode's real ones (online behaviour).",
    )
    p.add_argument(
        "--state-feedback",
        action="store_true",
        help="Feed measured proprioception (state[t+1]) back as the executed action -- what a real "
        "robot can actually supply, unlike the ground-truth action.",
    )
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    config = PreTrainedConfig.from_pretrained(ckpt)
    config.device = "cuda"
    meta = LeRobotDatasetMetadata(args.repo_id, root=args.dataset_root)
    q01 = torch.tensor(meta.stats["action"]["q01"], dtype=torch.float32)
    q99 = torch.tensor(meta.stats["action"]["q99"], dtype=torch.float32)

    from peft import PeftConfig, PeftModel

    pc = PeftConfig.from_pretrained(ckpt)
    policy = get_policy_class("lingbot_va").from_pretrained(pc.base_model_name_or_path, config=config)
    policy = PeftModel.from_pretrained(policy, str(ckpt), config=pc, is_trainable=False)
    policy = policy.to("cuda").eval().get_base_model()

    ds = LeRobotDataset(
        args.repo_id, root=args.dataset_root, episodes=[args.episode], video_backend=args.video_backend
    )
    pred_raw, gt_raw, pred_norm, bounds = teacher_forced_rollout(
        policy, config, ds, q01, q99, "cuda",
        self_feedback=args.self_feedback, state_feedback=args.state_feedback,
    )
    if args.save_traj:
        import numpy as np
        np.savez(args.save_traj, pred=pred_raw.numpy(), gt=gt_raw.numpy(),
                 bounds=np.array(bounds), self_feedback=args.self_feedback,
                 state_feedback=args.state_feedback)
        print(f"Wrote {args.save_traj}")
    mode = (
        "self-feedback (online-like)" if args.self_feedback
        else "state-feedback (measured proprioception)" if args.state_feedback
        else "teacher-forced (ground-truth actions)"
    )
    print(f"\nAction feedback mode: {mode}")

    mid = ((q01 + q99) / 2).cpu()
    baseline = mid.expand_as(gt_raw)  # "always predict the midpoint"
    model_mse = ((pred_raw - gt_raw) ** 2).mean(dim=0)
    base_mse = ((baseline - gt_raw) ** 2).mean(dim=0)

    print(f"\nEpisode {args.episode}: {pred_raw.shape[0]} predicted steps\n")
    hdr = f"{'joint':<15}{'pred std':>9}{'gt std':>9}{'pred p2p':>10}{'gt p2p':>9}{'pred mean':>11}{'midpoint':>10}"
    print(hdr)
    print("-" * len(hdr))
    rows = []
    for i, name in enumerate(ACTION_NAMES):
        r = {
            "joint": name,
            "pred_std": pred_raw[:, i].std().item(),
            "gt_std": gt_raw[:, i].std().item(),
            "pred_p2p": (pred_raw[:, i].max() - pred_raw[:, i].min()).item(),
            "gt_p2p": (gt_raw[:, i].max() - gt_raw[:, i].min()).item(),
            "pred_mean": pred_raw[:, i].mean().item(),
            "midpoint": mid[i].item(),
            "pred_norm_mean": pred_norm[:, i].mean().item(),
            "pred_norm_std": pred_norm[:, i].std().item(),
            "model_mse": model_mse[i].item(),
            "midpoint_mse": base_mse[i].item(),
        }
        rows.append(r)
        print(
            f"{name:<15}{r['pred_std']:>9.2f}{r['gt_std']:>9.2f}{r['pred_p2p']:>10.2f}"
            f"{r['gt_p2p']:>9.2f}{r['pred_mean']:>11.2f}{r['midpoint']:>10.2f}"
        )

    print(f"\n{'joint':<15}{'norm mean':>11}{'norm std':>10}{'model MSE':>12}{'midpoint MSE':>14}{'verdict':>12}")
    print("-" * 74)
    for r in rows:
        verdict = "beats base" if r["model_mse"] < r["midpoint_mse"] else "WORSE"
        print(
            f"{r['joint']:<15}{r['pred_norm_mean']:>11.3f}{r['pred_norm_std']:>10.3f}"
            f"{r['model_mse']:>12.2f}{r['midpoint_mse']:>14.2f}{verdict:>12}"
        )

    # How much motion is inside ONE chunk? This is what the arm performs between replans, and
    # it is what "the arm barely moves" is really measuring on a robot whose replan period is
    # ~10x the chunk's own duration.
    offs, pred_disp, gt_disp = 0, [], []
    for n in bounds:
        pc, gc = pred_raw[offs : offs + n], gt_raw[offs : offs + n]
        pred_disp.append((pc.max(dim=0).values - pc.min(dim=0).values))
        gt_disp.append((gc.max(dim=0).values - gc.min(dim=0).values))
        offs += n
    pred_disp = torch.stack(pred_disp).mean(dim=0)
    gt_disp = torch.stack(gt_disp).mean(dim=0)
    print(f"\nMean per-chunk travel over {len(bounds)} chunks ({config.chunk_size} steps = "
          f"{config.chunk_size / 30:.2f}s of 30 Hz demonstration), degrees:")
    print(f"{'joint':<15}{'predicted':>11}{'ground truth':>14}")
    print("-" * 40)
    for i, name in enumerate(ACTION_NAMES):
        print(f"{name:<15}{pred_disp[i]:>11.2f}{gt_disp[i]:>14.2f}")
        rows[i]["pred_chunk_travel"] = pred_disp[i].item()
        rows[i]["gt_chunk_travel"] = gt_disp[i].item()

    # Does the error accumulate? Autoregressive drift shows up as a trend across the rollout,
    # not as a uniformly bad prediction -- and that distinction decides whether periodic
    # re-grounding would help or whether the policy is simply wrong from the start.
    offs, per_chunk = 0, []
    for ci, n in enumerate(bounds):
        pc, gc = pred_raw[offs : offs + n], gt_raw[offs : offs + n]
        # Store the travel sums, not their per-chunk ratio: the demonstration is briefly
        # stationary in places, and dividing chunk-by-chunk lets a near-zero denominator
        # dominate any average. Ratios are formed from aggregated sums below instead.
        per_chunk.append(
            {
                "chunk": ci,
                "pred_travel": float((pc.max(dim=0).values - pc.min(dim=0).values).sum()),
                "gt_travel": float((gc.max(dim=0).values - gc.min(dim=0).values).sum()),
                "rmse": float(((pc - gc) ** 2).mean().sqrt()),
            }
        )
        offs += n
    q = max(1, len(per_chunk) // 4)
    print("\nDrift over the rollout (quarters):")
    print(f"{'chunks':<14}{'travel vs GT':>14}{'RMSE (deg)':>13}")
    print("-" * 41)
    quarters = []
    for qi in range(0, len(per_chunk), q):
        seg = per_chunk[qi : qi + q]
        if not seg:
            continue
        tr = sum(s["pred_travel"] for s in seg) / max(sum(s["gt_travel"] for s in seg), 1e-6)
        rm = sum(s["rmse"] for s in seg) / len(seg)
        quarters.append({"first_chunk": seg[0]["chunk"], "travel_ratio": tr, "rmse": rm})
        print(f"{f'{seg[0]['chunk']}-{seg[-1]['chunk']}':<14}{tr * 100:>13.1f}%{rm:>13.2f}")

    amp = float(torch.tensor([r["pred_std"] for r in rows]).sum() / torch.tensor([r["gt_std"] for r in rows]).sum())
    n_worse = sum(1 for r in rows if r["model_mse"] >= r["midpoint_mse"])
    print(f"\nPredicted motion amplitude vs ground truth: {amp * 100:.1f}%")
    print(f"Joints where the model does not beat 'always predict the midpoint': {n_worse}/{len(rows)}")
    if amp < 0.5 and args.self_feedback:
        print(
            "\nDIAGNOSIS: amplitude collapses once the model's own actions feed its KV cache.\n"
            "Re-run without --self-feedback: if that recovers full amplitude, the adapter is fine\n"
            "and this is autoregressive drift, not an undertrained action head."
        )
    elif amp < 0.5:
        print(
            "\nDIAGNOSIS: the policy predicts far less motion than the demonstrations even with\n"
            "real actions fed back. With a mean near the q01/q99 midpoint that is the 'stuck at\n"
            "normalized 0' failure -- one jump to mid-range at episode start, then stillness."
        )

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "episode": args.episode,
                    "self_feedback": args.self_feedback,
                    "steps": pred_raw.shape[0],
                    "amplitude_ratio": amp,
                    "joints": rows,
                    "per_chunk": per_chunk,
                    "quarters": quarters,
                },
                indent=2,
            )
        )
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
