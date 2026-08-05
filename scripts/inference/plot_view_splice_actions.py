#!/usr/bin/env python
"""Plot the action chunks produced with and without a spliced-in camera slot."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

for _f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",):
    if Path(_f).exists():
        try:
            fm.fontManager.addfont(_f)
        except Exception:  # noqa: BLE001
            pass
_have = {f.name for f in fm.fontManager.ttflist}
for _c in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback"):
    if _c in _have:
        plt.rcParams["font.sans-serif"] = [_c, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
LABEL = {"baseline": "3 视角（基准）", "zero": "丢 right + 补零",
         "copy_front": "丢 right + 复制 front", "copy_wrist": "丢 right + 复制 wrist"}
STYLE = {"baseline": ("k", "-", 2.0), "zero": ("tab:red", "--", 1.3),
         "copy_front": ("tab:blue", "--", 1.3), "copy_wrist": ("tab:green", "--", 1.3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--obs", type=int, default=0, help="Which observation's chunk to draw in detail")
    args = ap.parse_args()

    d = np.load(args.npz)
    modes = [m for m in ("baseline", "zero", "copy_front", "copy_wrist") if m in d]
    n_obs, chunk, n_j = d["baseline"].shape
    print(f"{n_obs} 观测 x {chunk} 步 x {n_j} 关节")

    # (a) per-joint action curve for one observation
    fig, axes = plt.subplots(2, 3, figsize=(16, 7), sharex=True)
    for j in range(n_j):
        ax = axes[j // 3][j % 3]
        for m in modes:
            c, ls, lw = STYLE[m]
            ax.plot(d[m][args.obs, :, j], color=c, ls=ls, lw=lw, label=LABEL[m])
        ax.set_title(JOINTS[j], fontsize=10)
        ax.grid(alpha=0.3)
        if j >= 3:
            ax.set_xlabel("chunk 内步序")
        if j % 3 == 0:
            ax.set_ylabel("角度 (deg)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"减视角后的动作曲线（观测 #{args.obs}）：虚线能否贴合黑色基准", y=1.0)
    fig.tight_layout()
    fig.savefig(Path(args.out) / "view_splice_action_curves.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # (b) deviation from baseline across every observation
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
    base = d["baseline"]
    for m in modes:
        if m == "baseline":
            continue
        dev = np.abs(d[m] - base)          # (obs, step, joint)
        c, ls, lw = STYLE[m]
        axes[0].plot(dev.mean(axis=(0, 2)), color=c, ls=ls, lw=1.6, label=LABEL[m])
        axes[1].plot(dev.max(axis=(0, 2)), color=c, ls=ls, lw=1.6, label=LABEL[m])
    for ax, t in ((axes[0], "平均偏差"), (axes[1], "最大偏差")):
        ax.set_xlabel("chunk 内步序")
        ax.set_ylabel("|Δaction| (deg)")
        ax.set_title(f"{t}（对全部观测聚合）")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[1].axhline(1.4, color="gray", ls=":", lw=1)
    axes[1].text(0.4, 1.5, "舵机跟踪误差 p50 ≈ 1.4°", fontsize=8, color="gray")
    fig.tight_layout()
    fig.savefig(Path(args.out) / "view_splice_deviation.png", dpi=140)
    plt.close(fig)

    print(f"{'方案':<24}{'mean|Δ|':>10}{'p95|Δ|':>10}{'max|Δ|':>10}")
    for m in modes:
        if m == "baseline":
            continue
        dev = np.abs(d[m] - base)
        print(f"{LABEL[m]:<24}{dev.mean():>10.3f}{np.percentile(dev, 95):>10.3f}{dev.max():>10.3f}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
