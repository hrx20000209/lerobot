#!/usr/bin/env python
"""Characterise the adaptive scheduler's signals -- including where they fail.

The controller assumes one thing: that imagining more consecutive steps makes
the prediction drift further from reality, so the residual can be read as "how
much sensing this moment needs". Panel (c) tests that assumption directly and it
does not hold, which is the point of the figure. Publishing the mechanism
without this panel would be claiming a control law that the data does not
support.
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
TAG = "ad2"
LO, HI = 0.3556, 0.3558          # the band the run calibrated
PHASES = [(0.40, 6.29, "接近"), (6.29, 12.20, "对准"), (12.20, 18.63, "抓取+搬运")]
T_END = 18.63

# The server served a SHADOW dry run before the real one and both wrote to this
# trace, so t0 has to come from the real run's own observation trace. Taking the
# file's first record instead silently mixes them -- and worse, the scheduler
# calibrated during the shadow segment, where a motionless arm makes every
# residual nearly identical (spread 0.265-0.366) and produced a band 0.0002
# wide. That band then governed the whole real run, in which the residual
# actually spans 0.261-0.734.
_obs = [json.loads(l) for l in open(next((B / TAG).glob("observation_trace_*.jsonl")))]
T0 = min(x["wall_time"] for x in _obs)
_all = [json.loads(l) for l in open(B / f"{TAG}_server_stage_trace.jsonl")]
r = [x for x in _all if x["server_recv_time"] >= T0 - 1]
n_shadow = len(_all) - len(r)
t0 = T0
t = np.array([x["server_recv_time"] - t0 for x in r])
enc = np.array([x["stages_ms"].get("vae_encode", 0) for x in r])
fed = enc < 20
k = np.array([x["lf_k"] for x in r])
intent = np.array([x.get("lf_intent") if x.get("lf_intent") is not None else np.nan for x in r], dtype=float)

rt = np.array([x["server_recv_time"] - t0 for x in r if x.get("lf_residual")])
rv = np.array([x["lf_residual"]["mean"] for x in r if x.get("lf_residual")])

# consecutive imagined steps preceding each real encode
runlen, c = [], 0
for i in range(len(r)):
    if fed[i]:
        c += 1
    else:
        runlen.append(c)
        c = 0
runlen = np.array(runlen)
n = min(len(runlen), len(rv))

fig, axes = plt.subplots(2, 2, figsize=(16, 9))

# (a) residual and k over time
ax = axes[0][0]
for lo, hi, name in PHASES:
    ax.axvspan(lo, hi, alpha=0.10, color="tab:green")
    ax.text((lo + hi) / 2, 0.72, name, ha="center", fontsize=8, color="tab:green")
ax.axvline(T_END, color="tab:red", ls="--", lw=1.2)
ax.text(T_END + 1, 0.72, "任务完成 18.63s", fontsize=8, color="tab:red")
ax.plot(rt, rv, ".-", ms=3, lw=0.7, color="tab:blue", label="残差")
ax.axhspan(LO, HI, color="tab:orange", alpha=0.55)
ax.text(30, HI + 0.012, f"标定带 lo={LO:.4f} hi={HI:.4f}（宽 0.0002）",
        fontsize=8, color="tab:orange", fontweight="bold")
ax2 = ax.twinx()
ax2.step(t, k, where="post", color="tab:purple", alpha=0.55, lw=1.0, label="k")
ax2.set_ylabel("k（回灌预算）", color="tab:purple"); ax2.set_ylim(-0.2, 2.6)
ax.set_xlabel("时间 (s)"); ax.set_ylabel("预测残差")
ax.set_title(f"(a) 残差与 k 随时间（真机段，已剔除 {n_shadow} 条 SHADOW 空跑）\n"
             "标定带来自空跑，落在真机分布第 20 百分位", fontsize=10)
ax.legend(fontsize=8, loc="upper left")

# (b) distribution vs the band
ax = axes[0][1]
ax.hist(rv, bins=30, color="tab:blue", alpha=0.75)
ax.axvspan(LO, HI, color="tab:orange", alpha=0.85)
for q, s in ((0.05, ":"), (0.5, "-"), (0.95, ":")):
    v = float(np.quantile(rv, q))
    ax.axvline(v, color="k", ls=s, lw=1.0)
    ax.text(v, ax.get_ylim()[1] * 0.92, f"p{int(q*100)}={v:.3f}", rotation=90,
            fontsize=7.5, ha="right", va="top")
ax.set_xlabel("预测残差"); ax.set_ylabel("次数")
ax.set_title(f"(b) 残差分布（n={len(rv)}，真机段）：橙带即控制器的判据\n"
             "带宽 0.0002，且位于 p20 → k 绝大多数时候被判为「残差偏高」", fontsize=10)

# (c) the assumption under test
ax = axes[1][0]
groups = sorted(set(runlen[:n].tolist()))
means = [rv[:n][runlen[:n] == g].mean() for g in groups]
stds = [rv[:n][runlen[:n] == g].std() for g in groups]
cnts = [int((runlen[:n] == g).sum()) for g in groups]
ax.bar(groups, means, yerr=stds, capsize=5, color="tab:red", alpha=0.8)
for g, m, c_ in zip(groups, means, cnts):
    ax.text(g, m + 0.01, f"{m:.4f}\nn={c_}", ha="center", fontsize=8.5)
cc = float(np.corrcoef(runlen[:n], rv[:n])[0, 1])
ax.set_xticks(groups)
ax.set_xlabel("该次真实编码之前，连续回灌了几步")
ax.set_ylabel("预测残差")
ax.set_title(f"(c) 控制器的核心假设：想象越多 → 漂移越大\n"
             f"实测 r = {cc:+.3f}（略负）→ **假设不成立**", fontsize=10, color="tab:red")

# (d) intent
ax = axes[1][1]
m = ~np.isnan(intent)
ax.plot(t[m], intent[m], ".-", ms=3, lw=0.7, color="tab:green")
ax.axvline(T_END, color="tab:red", ls="--", lw=1.2)
for lo, hi, name in PHASES:
    ax.axvspan(lo, hi, alpha=0.10, color="tab:green")
ax.set_yscale("log")
ax.set_xlabel("时间 (s)"); ax.set_ylabel("intent  mean|action − proprio| (deg, log)")
ax.set_title("(d) intent：与模型无关的停滞信号\n"
             f"p50 {np.nanmedian(intent):.2f}°，全程触发强制全感知 1 次", fontsize=10)

for row in axes:
    for a in row:
        a.grid(alpha=0.3)
fig.suptitle("自适应感知调度：信号刻画与假设检验（ad2，2026-08-06）", fontsize=13, y=1.00)
fig.tight_layout()
out = B / "adaptive_sensing.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"-> {out}")
print(f"lag-1 自相关 {np.corrcoef(rv[:-1], rv[1:])[0,1]:+.3f}   假设检验 r={cc:+.3f}")
