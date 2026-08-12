#!/usr/bin/env python
"""Full-episode action curves for each dropped view, not a single chunk.

A one-chunk comparison cannot distinguish a transient wobble from a bias that
holds for the whole task; over a full episode the difference is obvious by eye.
Each curve is the first action of the chunk produced at that observation, so the
series is what the arm would actually have been commanded toward.
"""

import json
import sys
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
LABEL = {"-1": "3 视角基准", "2": "删槽位2 (right 相机)", "3": "删槽位3 (wrist 相机)",
         "4": "删槽位4 (front 主视角)"}
COL = {"-1": "k", "2": "tab:red", "3": "tab:orange", "4": "tab:green"}
WORST = {"2": 54.5, "3": 45.7, "4": 0.2}

src = Path(sys.argv[1] if len(sys.argv) > 1
           else "/tmp/claude-1000/-home-hrx/8e18d9e2-26fc-4d7f-9b9d-27dfa1940253/scratchpad/view_drop_full.json")
d = json.loads(src.read_text())
t = np.array(d["rel_t"])
acts = {k: np.array(v) for k, v in d["actions"].items()}
base = acts["-1"]
drops = [k for k in acts if k != "-1"]

fig = plt.figure(figsize=(17, 10))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.95], hspace=0.42, wspace=0.24)

# six joints: first action of each chunk over the whole episode
for j in range(6):
    ax = fig.add_subplot(gs[j // 3, j % 3])
    ax.plot(t, base[:, 0, j], "-", color="k", lw=2.0, label=LABEL["-1"], zorder=5)
    for k in drops:
        ax.plot(t, acts[k][:, 0, j], "--", color=COL[k], lw=1.3, alpha=0.9, label=LABEL[k])
    ax.set_title(JOINTS[j], fontsize=10)
    ax.set_xlabel("时间 (s)", fontsize=8)
    ax.set_ylabel("指令 (deg)", fontsize=8)
    ax.grid(alpha=0.3)
    if j == 0:
        ax.legend(fontsize=7.5, loc="best")

# deviation over the episode
ax = fig.add_subplot(gs[2, 0])
for k in drops:
    dev = np.abs(acts[k][:, 0, :] - base[:, 0, :]).mean(axis=1)
    ax.plot(t, dev, "-", color=COL[k], lw=1.4, label=f"{LABEL[k]}  中位 {np.median(dev):.2f}°")
ax.set_xlabel("时间 (s)"); ax.set_ylabel("平均 |Δ动作| (deg)")
ax.set_title("(a) 偏差随任务进程（整段，非单个 chunk）", fontsize=10)
ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

# deviation across the whole chunk, not just its first step
ax = fig.add_subplot(gs[2, 1])
for k in drops:
    per_step = np.abs(acts[k] - base).mean(axis=(0, 2))
    ax.plot(np.arange(len(per_step)), per_step, "-o", ms=3, color=COL[k], label=LABEL[k])
ax.set_xlabel("chunk 内步序"); ax.set_ylabel("平均 |Δ动作| (deg)")
ax.set_title("(b) 偏差沿 chunk 内部的分布", fontsize=10)
ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

# the thing that decides attributability
ax = fig.add_subplot(gs[2, 2])
ks = list(WORST)
x = np.arange(len(ks))
w = 0.38
devs = [float(np.median(np.abs(acts[k][:, 0, :] - base[:, 0, :]).mean(axis=1))) for k in ks]
ax.bar(x - w / 2, devs, w, color="tab:blue", label="动作偏差中位 (deg)")
ax2 = ax.twinx()
ax2.bar(x + w / 2, [WORST[k] for k in ks], w, color="tab:red", alpha=0.85,
        label="其余视角 latent 最大改变 (%)")
for xi, k in zip(x, ks):
    ax.text(xi - w / 2, devs[ks.index(k)] + 0.05, f"{devs[ks.index(k)]:.2f}", ha="center", fontsize=8)
    ax2.text(xi + w / 2, WORST[k] + 1.5, f"{WORST[k]:.1f}%", ha="center", fontsize=8, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([LABEL[k].replace(" ", "\n", 1) for k in ks], fontsize=7.5)
ax.set_ylabel("动作偏差 (deg)", color="tab:blue")
ax2.set_ylabel("其余视角被改变 (%)", color="tab:red")
ax.set_title("(c) 动作偏差看着都不大，但只有删槽位4\n没有连带破坏其余视角 → 唯一可归因", fontsize=10)
ax.grid(alpha=0.3, axis="y")

fig.suptitle("减视角消融：整段任务的动作轨迹（base24 观测序列，denoise=1，同种子）", fontsize=13, y=0.98)
out = Path("/home/hrx/Projects/models/three_cubes_1/cosmos_policy_step20000/profiling/view_drop_full.png")
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"-> {out}")
for k in drops:
    dev = np.abs(acts[k][:, 0, :] - base[:, 0, :]).mean(axis=1)
    print(f"  {LABEL[k]:24s} 偏差 中位 {np.median(dev):5.2f}°  p95 {np.percentile(dev,95):5.2f}°  最大 {dev.max():5.2f}°")
