#!/usr/bin/env python
"""The 2026-08-06 result in one figure: what latent feedback did and did not do.

Three panels, because the story needs all three. (a) is the mechanism -- where
the per-inference time went, split by whether that inference encoded cameras or
imagined them. (b) is the outcome the mechanism was supposed to produce. (c) is
the check that the outcome was not bought by simply running the arm longer.
"""

import json
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

B = Path("/home/hrx/Projects/models/three_cubes_1/cosmos_policy_step20000/profiling")

# (tag, label, end-to-end seconds; None = task failed)
RUNS = [
    ("base24", "base24\n全真编码", 37.98),
    ("gain2b", "gain2b\ngain=2", 21.34),
    ("lf1", "lf1\n回灌 1假1真", 29.49),
    ("lf2", "lf2\n回灌 2假1真", 16.71),
    ("splice24", "splice24\n删 right 视角", None),
]


def stage_trace(tag):
    p = B / f"{tag}_server_stage_trace.jsonl"
    if not p.exists():
        p = B / f"_failed_{tag}_server_stage_trace.jsonl"
    return [json.loads(l) for l in open(p)]


fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
x = np.arange(len(RUNS))
labels = [lab for _, lab, _ in RUNS]

# (a) per-inference latency, real vs imagined
ax = axes[0]
real_med, fb_med, fb_frac = [], [], []
for tag, _, _ in RUNS:
    r = stage_trace(tag)
    tot = np.array([v["total_server_ms"] for v in r])
    enc = np.array([v["stages_ms"].get("vae_encode", 0) for v in r])
    fed = enc < 20  # an inference that never called the tokenizer
    real_med.append(float(np.median(tot[~fed])) if (~fed).any() else 0.0)
    fb_med.append(float(np.median(tot[fed])) if fed.any() else 0.0)
    fb_frac.append(100 * fed.mean())
w = 0.38
ax.bar(x - w / 2, real_med, w, label="真实编码步", color="tab:blue")
ax.bar(x + w / 2, fb_med, w, label="回灌步（跳过 VAE）", color="tab:orange")
for xi, (a, b, f) in enumerate(zip(real_med, fb_med, fb_frac)):
    ax.text(xi - w / 2, a + 15, f"{a:.0f}", ha="center", fontsize=8.5, fontweight="bold")
    if b > 0:
        ax.text(xi + w / 2, b + 15, f"{b:.0f}", ha="center", fontsize=8.5, fontweight="bold")
        ax.text(xi + w / 2, b / 2, f"{f:.0f}%", ha="center", fontsize=8, color="white", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("单次推理耗时中位 (ms)")
ax.set_title("(a) 机制：回灌步跳过 VAE 编码\n（橙柱内数字 = 回灌步占比）", fontsize=10)
ax.legend(fontsize=8)

# (b) end-to-end task latency
ax = axes[1]
vals = [v if v else 0 for _, _, v in RUNS]
cols = ["tab:green" if v else "tab:gray" for _, _, v in RUNS]
ax.bar(x, vals, color=cols)
for xi, (_, _, v) in zip(x, RUNS):
    ax.text(xi, (v or 0) + 0.8, f"{v:.2f}s" if v else "任务失败", ha="center",
            fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("首次 perception → 夹爪松开 (s)")
ax.set_ylim(0, 44)
ax.set_title("(b) 结果：端到端任务延迟\n（灰 = 未完成任务）", fontsize=10)

# (c) inference count within the task window
ax = axes[2]
counts, acts = [], []
for tag, _, tend in RUNS:
    d = B / tag
    if not d.exists():
        d = B / f"_failed_{tag}_saturated"
    a = [json.loads(l) for l in open(next(d.glob("action_trace_*.jsonl")))]
    o = [json.loads(l) for l in open(next(d.glob("observation_trace_*.jsonl")))]
    t0 = min(v["wall_time"] for v in o)
    hi = t0 + (tend if tend else 1e9)
    r = stage_trace(tag)
    counts.append(sum(1 for v in r if t0 <= v["server_recv_time"] <= hi))
    acts.append(sum(1 for v in a if v["exec_start"] <= hi))
ax.bar(x - w / 2, counts, w, label="WAM 推理次数", color="tab:purple")
ax.bar(x + w / 2, np.array(acts) / 10, w, label="执行动作数 ÷10", color="tab:cyan")
for xi, (c, m) in enumerate(zip(counts, acts)):
    ax.text(xi - w / 2, c + 1.5, str(c), ha="center", fontsize=8.5, fontweight="bold")
    ax.text(xi + w / 2, m / 10 + 1.5, str(m), ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("次数（任务窗口内）")
ax.set_title("(c) 代价核对：不是靠多跑换来的\n（splice24 未完成，取全程）", fontsize=10)
ax.legend(fontsize=8)

for a in axes:
    a.grid(alpha=0.3, axis="y")
fig.suptitle("Latent 回灌与减视角：2026-08-06 真机结果", fontsize=13, y=1.02)
fig.tight_layout()
out = B / "latent_feedback.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"-> {out}")
