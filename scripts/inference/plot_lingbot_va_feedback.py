#!/usr/bin/env python
"""Plot predicted vs ground-truth joint trajectories under two KV-cache feedback modes.

Shows what changes when the action history fed back into LingBot-VA's KV cache stops
being the model's own prediction and becomes the arm's measured proprioception.

Inputs are the .npz files written by
``diagnose_lingbot_va_actions.py --save-traj`` under ``--self-feedback`` and
``--state-feedback``.

Usage:
    python scripts/inference/plot_lingbot_va_feedback.py self.npz state.npz -o out.png
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Slots 1 and 2 of the data-viz reference palette (documented as validated; the first
# three slots clear the all-pairs floors that small multiples require). Ground truth is
# deliberately NOT a categorical hue -- it is the reference, so it wears neutral ink.
C_GT = "#52514e"
C_STATE = "#2a78d6"
C_SELF = "#eb6834"
GRID = "#e3e2df"
INK = "#0b0b0b"
INK2 = "#52514e"


def amplitude(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(pred.std(axis=0).sum() / gt.std(axis=0).sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("self_npz")
    p.add_argument("state_npz")
    p.add_argument("-o", "--out", default="lingbot_va_feedback.png")
    args = p.parse_args()

    a = np.load(args.self_npz)
    b = np.load(args.state_npz)
    gt = a["gt"]
    n = min(len(gt), len(b["gt"]), len(a["pred"]), len(b["pred"]))
    gt, pred_self, pred_state = gt[:n], a["pred"][:n], b["pred"][:n]
    t = np.arange(n) / 30.0  # demonstrations are 30 Hz

    amp_self = amplitude(pred_self, gt) * 100
    amp_state = amplitude(pred_state, gt) * 100

    fig, axes = plt.subplots(2, 3, figsize=(15, 7.2), dpi=160, sharex=True)
    fig.patch.set_facecolor("#fcfcfb")

    for j, (ax, name) in enumerate(zip(axes.ravel(), JOINTS, strict=True)):
        ax.set_facecolor("#fcfcfb")
        ax.plot(t, gt[:, j], color=C_GT, lw=2.2, label="Ground truth", zorder=3)
        ax.plot(t, pred_state[:, j], color=C_STATE, lw=1.8, label="Fed measured state", zorder=2)
        ax.plot(t, pred_self[:, j], color=C_SELF, lw=1.8, ls=(0, (5, 2)),
                label="Fed own prediction", zorder=1)

        ax.set_title(name, fontsize=11.5, color=INK, pad=7, loc="left", fontweight="600")
        # RMSE rides in the title row, not inside the axes: the traces reach both the top and
        # the bottom of most facets, so any in-plot annotation collides with data or the axis.
        rs = np.sqrt(((pred_state[:, j] - gt[:, j]) ** 2).mean())
        rf = np.sqrt(((pred_self[:, j] - gt[:, j]) ** 2).mean())
        ax.set_title(f"RMSE  {rs:.1f}°  vs  {rf:.1f}°", fontsize=8.8, color=INK2,
                     pad=7, loc="right", family="monospace")

        ax.grid(True, color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9, length=3)
        if j % 3 == 0:
            ax.set_ylabel("degrees", fontsize=9.5, color=INK2)
        if j >= 3:
            ax.set_xlabel("seconds of demonstration (30 Hz)", fontsize=9.5, color=INK2)

    fig.suptitle(
        "What the KV cache is told the arm did — and what the policy predicts next",
        x=0.008, y=0.985, ha="left", fontsize=14.5, color=INK, fontweight="700",
    )
    fig.text(
        0.008, 0.945,
        "Held-out episode 95, open loop. Camera history is real in both runs; only the action history differs.",
        ha="left", va="top", fontsize=10.5, color=INK2,
    )
    fig.text(
        0.008, 0.912,
        f"Motion amplitude vs ground truth:  measured state {amp_state:.0f}%   ·   own prediction {amp_self:.0f}%",
        ha="left", va="top", fontsize=10.5, color=INK, fontweight="600",
    )
    # Legend gets its own row under the header rather than sharing the top-right corner
    # with the subtitle, which overlapped it.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.995, 0.955),
               ncol=1, frameon=False, fontsize=10.5, labelcolor=INK2,
               handlelength=2.6, borderaxespad=0)

    fig.tight_layout(rect=(0, 0, 1, 0.885))
    fig.savefig(args.out, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"Wrote {args.out}  ({n} steps)")
    print(f"  amplitude: state-feedback {amp_state:.1f}%  self-feedback {amp_self:.1f}%")


if __name__ == "__main__":
    main()
