#!/usr/bin/env python
"""One figure tying the 2026-08-05 configurations together.

Each panel is a different axis of the same question -- where the time goes, how
stale the action is when it lands, how well the arm tracks, and whether the task
actually gets done -- because the session repeatedly found that the first three
do not predict the fourth.
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
J = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
     "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]

# (tag, label, task successes as observed on hardware)
RUNS = [
    ("fs300", "fps8\nfuture-state 诊断", 1),
    ("deploy300", "fps8\n部署基准", 0),
    ("fps24", "fps24\n无反馈", 1),
    ("fb_final", "fps24\n+反馈", 1),
    ("fps30b", "fps30\n+反馈+自整定", 4),
    ("splice30", "fps30\n减视角(2路)", 0),
]
STAGES = ["image_preproc", "vae_encode", "vae_decode", "dit_denoise", "sampler_other"]
CN = {"image_preproc": "图像预处理", "vae_encode": "VAE 编码", "vae_decode": "VAE 解码(诊断)",
      "dit_denoise": "DiT 去噪", "sampler_other": "采样器其他"}


def load(tag):
    acts = [json.loads(l) for l in open(next((B / tag).glob("action_trace_*.jsonl")))]
    t0, t1 = acts[0]["exec_start"], acts[-1]["exec_end"]
    srv = [r for r in (json.loads(l) for l in open(B / f"{tag}_server_stage_trace.jsonl"))
           if t0 - 5 <= r["server_recv_time"] <= t1 + 1]
    return acts, srv, t1 - t0


data = {tag: load(tag) for tag, _, _ in RUNS}
labels = [lab for _, lab, _ in RUNS]
x = np.arange(len(RUNS))

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# (a) stage budget
ax = axes[0][0]
bottom = np.zeros(len(RUNS))
colors = plt.get_cmap("tab10")
for i, s in enumerate(STAGES):
    v = np.array([np.median([r["stages_ms"].get(s, 0) for r in data[t][1]]) for t, _, _ in RUNS])
    ax.bar(x, v, bottom=bottom, label=CN[s], color=colors(i))
    bottom += v
for xi, tot in zip(x, bottom):
    ax.text(xi, tot + 25, f"{tot:.0f}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("单次推理耗时 (ms)")
ax.set_title("(a) 服务端阶段预算：VAE 编码始终是大头，DiT 从不是")
ax.legend(fontsize=8)

# (b) observation-to-execution latency
ax = axes[0][1]
p50 = [np.median([a["observation_to_execution_ms"] for a in data[t][0] if a["observation_to_execution_ms"]])
       for t, _, _ in RUNS]
p90 = [np.percentile([a["observation_to_execution_ms"] for a in data[t][0] if a["observation_to_execution_ms"]], 90)
       for t, _, _ in RUNS]
ax.bar(x, p50, color="tab:purple", label="p50")
ax.plot(x, p90, "k^--", ms=7, label="p90")
for xi, v in zip(x, p50):
    ax.text(xi, v + 40, f"{v:.0f}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("观测 → 执行 (ms)")
ax.set_title("(b) 动作落地时，它依据的观测已经多旧")
ax.legend(fontsize=8)

# (c) tracking error
ax = axes[1][0]
w = 0.38
e50, emax = [], []
for t, _, _ in RUNS:
    acts = data[t][0]
    cmd = np.array([[a["commanded_action"][j] for j in J] for a in acts])
    pres = np.array([[a["present_position"][j] for j in J] for a in acts])
    g = np.abs(cmd - pres)
    e50.append(np.median(g)); emax.append(g.max())
ax.bar(x - w / 2, e50, w, label="p50", color="tab:blue")
ax.bar(x + w / 2, emax, w, label="max", color="tab:red")
for xi, v in zip(x, emax):
    ax.text(xi + w / 2, v + 1.5, f"{v:.0f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("|指令 − 实测| (deg)")
ax.set_title("(c) 跟踪误差：反馈守卫削平极端值，不动常态")
ax.legend(fontsize=8)

# (d) task successes -- the axis the others fail to predict
ax = axes[1][1]
succ = [s for _, _, s in RUNS]
bars = ax.bar(x, succ, color=["tab:green" if s > 0 else "tab:gray" for s in succ])
for xi, s in zip(x, succ):
    ax.text(xi, s + 0.08, str(s), ha="center", fontsize=12, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("任务成功次数 / 每次运行")
ax.set_ylim(0, max(succ) + 0.8)
ax.set_title("(d) 任务成功次数 —— (a)(b)(c) 都没能预测它")

for row in axes:
    for a in row:
        a.grid(alpha=0.3, axis="y")
fig.suptitle("Cosmos step-20000 × SO101：各配置横向对比（2026-08-05）", fontsize=14, y=1.00)
fig.tight_layout()
out = B / "experiment_summary.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"-> {out}")
