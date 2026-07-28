#!/usr/bin/env python
"""Stage-1 verification: does the closed-loop collapse persist at step_20000?

Apples-to-apples on ONE held-out episode at the SAME checkpoint (step_20000), comparing:
  - teacher_forced : real obs + real GT past actions in the KV cache (upper bound)
  - state_feedback : real obs + measured state[t+1] as past action (what a real robot has)
  - self_feedback  : real obs + model's own previous prediction (true online action history)
  - select_action  : the actual public deployment API, one real frame per control tick

Prints per-dim MSE + motion-amplitude ratio for each, and saves plots under
outputs/eval/action_plots/step_20000/. Physical (degree) units throughout.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/home/rxhuang/Projects/lerobot/scripts/inference")

from common import (  # noqa: E402
    ACTION_NAMES, action_quantiles, attach_text_embed_cache, build_config,
    denormalize_action, make_ds_meta,
)
from diagnose_lingbot_va_actions import teacher_forced_rollout  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.policies.factory import make_policy  # noqa: E402
from plotting import plot_action_compare  # noqa: E402

OUT_ROOT = Path("/home/rxhuang/Projects/lerobot/outputs/eval/action_plots")


def out_dir_for(checkpoint: str) -> Path:
    ck = Path(checkpoint)
    return OUT_ROOT / f"{ck.parent.parent.name}_{ck.name}"  # <run>_<step>, e.g. gripper_w5_r64_heads_step_8000


class InMemoryEpisode:
    """Decode a held-out episode ONCE and serve frames from RAM.

    teacher_forced_rollout / select_action index the dataset once per frame per mode; with
    av1 random-access video decode that is the (CPU-bound) bottleneck and 4 modes re-decode
    the same episode 4x. Materializing the frames once makes every mode GPU-bound instead.
    """

    def __init__(self, repo_id, root, episode, cam_keys):
        ds = LeRobotDataset(repo_id, root=root, episodes=[episode])
        keep = list(cam_keys) + ["action", "observation.state"]
        self._frames = []
        for i in range(len(ds)):
            s = ds[i]
            self._frames.append({k: s[k].clone() for k in keep})
        self._task = ds[0]["task"]

    def __len__(self):
        return len(self._frames)

    def __getitem__(self, i):
        f = self._frames[i]
        return {**f, "task": self._task}


def summarize(name, pred_raw, gt_raw):
    mse = ((pred_raw - gt_raw) ** 2).mean(dim=0)
    amp = float(pred_raw.std(0).sum() / gt_raw.std(0).sum())
    print(f"\n[{name}] mean_mse={mse.mean().item():.2f}  amplitude={amp*100:.1f}%")
    for i, nm in enumerate(ACTION_NAMES):
        print(f"    {nm:<14} mse={mse[i].item():>9.2f}  pred_std={pred_raw[:,i].std():>7.2f} gt_std={gt_raw[:,i].std():>7.2f}")
    return {"mean_mse": mse.mean().item(), "amplitude_pct": amp * 100,
            "per_dim_mse": {nm: mse[i].item() for i, nm in enumerate(ACTION_NAMES)}}


def run_select_action(policy, config, ds, q01, q99, device):
    """Drive the real public deployment API, one real frame per control tick.

    observation.state is included in every batch so the state-feedback patch (if attached) can
    read it; plain select_action ignores it.
    """
    cam_keys = config.obs_cam_keys
    task = ds[0]["task"]
    policy.reset()
    preds = []
    n = len(ds)
    for t in range(n):
        batch = {k: ds[t][k].unsqueeze(0).to(device) for k in cam_keys}
        batch["observation.state"] = ds[t]["observation.state"].unsqueeze(0).to(device)
        batch["task"] = [task]
        with torch.no_grad():
            a = policy.select_action(batch)  # [1, n_used] normalized
        preds.append(a.detach().cpu().reshape(-1))
    pred_norm = torch.stack(preds)
    gt_raw = torch.stack([ds[t]["action"] for t in range(len(pred_norm))])
    return denormalize_action(pred_norm, q01, q99), gt_raw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/data/rxhuang/lingbot_va_runs/full_20k/checkpoints/step_20000")
    p.add_argument("--episode", type=int, default=95)
    p.add_argument("--modes", nargs="+",
                   default=["teacher_forced", "state_feedback", "self_feedback", "select_action"])
    p.add_argument("--max-frames", type=int, default=320,
                   help="Truncate the episode to bound runtime (enough chunks to see collapse-or-not).")
    args = p.parse_args()
    OUT = out_dir_for(args.checkpoint)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {OUT}", flush=True)

    config = build_config(attn_mode="torch")
    ds_meta = make_ds_meta()
    q01, q99 = action_quantiles(ds_meta)

    print("decoding episode into RAM (once)...", flush=True)
    ds = InMemoryEpisode("three_cubes_1", "/data/rxhuang/three_cubes_1", args.episode, config.obs_cam_keys)
    if args.max_frames and len(ds) > args.max_frames:
        ds._frames = ds._frames[: args.max_frames]
    print(f"episode {args.episode}: {len(ds)} frames; checkpoint={Path(args.checkpoint).name}", flush=True)

    policy = make_policy(config, ds_meta=ds_meta)
    attach_text_embed_cache(policy)
    from peft import PeftModel
    policy = PeftModel.from_pretrained(policy, args.checkpoint, is_trainable=False).get_base_model()
    policy.eval()

    results = {}
    for mode in args.modes:
        t0 = time.time()
        if mode == "select_action_statefb":
            from state_feedback import attach_state_feedback
            attach_state_feedback(policy, q01, q99, verbose=True)
            pred_raw, gt_raw = run_select_action(policy, config, ds, q01, q99, "cuda")
        elif mode == "select_action_hybrid":
            # arm from measured state, gripper (channel 28) from the model's own commanded action
            from state_feedback import attach_state_feedback
            attach_state_feedback(policy, q01, q99, verbose=True, hybrid_gripper_channel=28)
            pred_raw, gt_raw = run_select_action(policy, config, ds, q01, q99, "cuda")
        elif mode == "select_action":
            pred_raw, gt_raw = run_select_action(policy, config, ds, q01, q99, "cuda")
        else:
            kw = dict(self_feedback=(mode == "self_feedback"), state_feedback=(mode == "state_feedback"))
            pred_raw, gt_raw, _, _ = teacher_forced_rollout(policy, config, ds, q01, q99, "cuda", **kw)
        results[mode] = summarize(mode, pred_raw, gt_raw)
        results[mode]["seconds"] = round(time.time() - t0, 1)
        ck_label = Path(args.checkpoint).name
        plot_action_compare(pred_raw, gt_raw, OUT / f"ep{args.episode}_{mode}.png",
                            title=f"{mode} | {ck_label} | ep{args.episode} | "
                                  f"mse={results[mode]['mean_mse']:.1f} amp={results[mode]['amplitude_pct']:.0f}%")
        print(f"    ({mode} took {results[mode]['seconds']}s)")

    json.dump(results, open(OUT / f"ep{args.episode}_metrics.json", "w"), indent=2)
    print("\n==== SUMMARY (mean_mse / amplitude) ====")
    for m, r in results.items():
        print(f"  {m:<16} mse={r['mean_mse']:>8.1f}  amp={r['amplitude_pct']:>5.0f}%")
    print(f"\nsaved plots + metrics to {OUT}")


if __name__ == "__main__":
    main()
