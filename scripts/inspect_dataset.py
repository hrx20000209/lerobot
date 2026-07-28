#!/usr/bin/env python
"""Stage-1 read-only inspection of the three_cubes_1 SO-101 LeRobot dataset.

Reads the raw parquet action/state columns (no video decode), prints per-dim
statistics, and writes:
  - outputs/diagnosis/action_stats.png  (per-dim histogram + min/max/mean/std)
  - outputs/diagnosis/action_curves_ep<e>.png  (action vs time for sample episodes)

Usage:
  python scripts/inspect_dataset.py \
      --root /data/rxhuang/three_cubes_1 \
      --episodes 0 50 --out outputs/diagnosis
"""
import argparse
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ACTION_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def load_frames(root: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(root) / "data" / "**" / "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"no parquet under {root}/data")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("index").reset_index(drop=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/data/rxhuang/three_cubes_1")
    p.add_argument("--episodes", type=int, nargs="+", default=[0, 50])
    p.add_argument("--out", default="outputs/diagnosis")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_frames(args.root)
    a = np.stack(df["action"].values).astype(np.float64)  # [N, 6]
    s = np.stack(df["observation.state"].values).astype(np.float64)
    n_dim = a.shape[1]

    lens = df.groupby("episode_index").size()
    print(f"frames={len(df)} episodes={df['episode_index'].nunique()} "
          f"ep_len min/median/max={int(lens.min())}/{int(lens.median())}/{int(lens.max())}")

    stats = {}
    print("\nper-dim action stats (physical units = degrees):")
    print(f"{'dim':<15}{'min':>10}{'max':>10}{'mean':>10}{'std':>10}{'q01':>10}{'q99':>10}")
    for i, name in enumerate(ACTION_NAMES[:n_dim]):
        col = a[:, i]
        q01, q99 = np.quantile(col, 0.01), np.quantile(col, 0.99)
        stats[name] = dict(min=col.min(), max=col.max(), mean=col.mean(), std=col.std(), q01=q01, q99=q99)
        print(f"{name:<15}{col.min():>10.2f}{col.max():>10.2f}{col.mean():>10.2f}"
              f"{col.std():>10.2f}{q01:>10.2f}{q99:>10.2f}")

    # action == state next-step? check control convention (is action an absolute target ~ next state?)
    # SO-101 teleop records action = commanded absolute joint target; state = measured. Compare.
    diff = np.abs(a - s).mean(0)
    print(f"\nmean|action - state| per dim: {np.round(diff, 3).tolist()}")
    print("(small => action is ~absolute joint target near current state; large => delta or different frame)")

    # --- histograms ---
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for i, name in enumerate(ACTION_NAMES[:n_dim]):
        ax = axes.flat[i]
        ax.hist(a[:, i], bins=80, color="steelblue", alpha=0.85)
        st = stats[name]
        ax.set_title(f"{name}\nmin={st['min']:.1f} max={st['max']:.1f} "
                     f"mean={st['mean']:.1f} std={st['std']:.1f}", fontsize=9)
        ax.axvline(st["q01"], color="red", ls="--", lw=1)
        ax.axvline(st["q99"], color="red", ls="--", lw=1)
    fig.suptitle("three_cubes_1 action distribution (per joint, degrees); red = q01/q99", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "action_stats.png", dpi=110)
    plt.close(fig)
    print(f"\nsaved {out / 'action_stats.png'}")

    # --- per-episode action curves ---
    for ep in args.episodes:
        sub = df[df["episode_index"] == ep]
        if len(sub) == 0:
            print(f"episode {ep} not found, skipping")
            continue
        ea = np.stack(sub["action"].values)
        es = np.stack(sub["observation.state"].values)
        fig, axes = plt.subplots(n_dim, 1, figsize=(11, 1.7 * n_dim), sharex=True)
        for i, name in enumerate(ACTION_NAMES[:n_dim]):
            axes[i].plot(ea[:, i], color="C1", label="action")
            axes[i].plot(es[:, i], color="C0", lw=0.8, alpha=0.7, label="state")
            axes[i].set_ylabel(name, fontsize=8)
        axes[0].legend(fontsize=8, loc="upper right")
        axes[-1].set_xlabel("frame")
        fig.suptitle(f"episode {ep}: action (orange) vs state (blue), {len(sub)} frames")
        fig.tight_layout()
        fig.savefig(out / f"action_curves_ep{ep}.png", dpi=110)
        plt.close(fig)
        print(f"saved {out / f'action_curves_ep{ep}.png'}")

    json.dump({k: {kk: float(vv) for kk, vv in v.items()} for k, v in stats.items()},
              open(out / "action_stats.json", "w"), indent=2)
    print(f"saved {out / 'action_stats.json'}")


if __name__ == "__main__":
    main()
