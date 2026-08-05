#!/usr/bin/env python
"""Where the arm dwells, is it because the commanded action and the measured
state have drifted apart?

Plots, on one shared time axis, the commanded action against the joint position
actually measured at that instant, their difference, the arm's speed, and the
servo load -- so a dwell can be attributed to one of:

  policy      the command tracks the measurement, both flat -> the policy is
              asking for no motion.
  actuation   the command leads the measurement by a persistent gap and the
              load is saturated -> the servo cannot deliver what is asked.
  pipeline    the action queue ran dry -> nothing to execute.

These are indistinguishable from the joint trajectory alone, which is why the
earlier dwell analysis could not say which one it was looking at.
"""

import argparse
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

JOINTS = ["shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
          "wrist_flex.pos", "wrist_roll.pos", "gripper.pos"]
SHORT = ["pan", "lift", "elbow", "wflex", "wroll", "grip"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--dwell-speed", type=float, default=1.5, help="deg/s below which the arm counts as dwelling")
    ap.add_argument("--dwell-min-s", type=float, default=1.5)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir or run_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    acts = [json.loads(l) for l in open(next(run_dir.glob("action_trace_*.jsonl")))]

    t0 = acts[0]["exec_start"]
    t = np.array([a["exec_start"] - t0 for a in acts])
    cmd = np.array([[a["commanded_action"][j] for j in JOINTS] for a in acts])
    pres = np.array([[a["present_position"][j] for j in JOINTS] for a in acts])
    gap = cmd - pres
    q = np.array([a["queue_size_before"] for a in acts])
    age = np.array([a["observation_to_execution_ms"] or np.nan for a in acts], dtype=float)

    # servo load, sampled sparsely
    lt, lv = [], []
    for a in acts:
        act = a.get("actuator")
        if act and "load" in act:
            lt.append(a["exec_start"] - t0)
            lv.append([abs(act["load"].get(j.removesuffix(".pos"), 0)) for j in JOINTS])
    lt, lv = np.array(lt), (np.array(lv) if lv else np.zeros((0, 6)))

    # speed from the measured positions
    speed = np.zeros(len(t))
    for i in range(len(t)):
        m = (t >= t[i] - 0.5) & (t <= t[i] + 0.5)
        if m.sum() >= 2:
            span = t[m][-1] - t[m][0]
            if span > 1e-3:
                speed[i] = np.abs(pres[m][-1] - pres[m][0]).max() / span

    dwell, start = [], None
    for i, s in enumerate(speed < args.dwell_speed):
        if s and start is None:
            start = i
        elif not s and start is not None:
            if t[i - 1] - t[start] >= args.dwell_min_s:
                dwell.append((start, i - 1))
            start = None
    if start is not None and t[-1] - t[start] >= args.dwell_min_s:
        dwell.append((start, len(t) - 1))

    print(f"运行 {t[-1]:.0f}s, {len(acts)} 动作, 检出 {len(dwell)} 段停留 (>{args.dwell_min_s}s)\n")
    print(f"{'停留区间':>16}{'时长':>7}{'最大|gap|':>10}{'gap关节':>9}{'队列min':>8}{'负载max':>8}{'归因':>10}")
    records = []
    for a, b in dwell:
        seg = slice(a, b + 1)
        g = np.abs(gap[seg])
        worst_j = int(np.argmax(g.max(axis=0)))
        qmin = int(q[seg].min())
        lm = (lv[(lt >= t[a]) & (lt <= t[b])].max() if len(lt) and ((lt >= t[a]) & (lt <= t[b])).any() else np.nan)
        if qmin == 0:
            why = "流水线"
        elif g.max() > 3.0 and (np.isnan(lm) or lm >= 90):
            why = "执行受限"
        elif g.max() > 3.0:
            why = "跟踪滞后"
        else:
            why = "策略静止"
        print(f"{t[a]:>7.1f}-{t[b]:<8.1f}{t[b] - t[a]:>7.1f}{g.max():>10.2f}{SHORT[worst_j]:>9}"
              f"{qmin:>8}{lm:>8.0f}{why:>10}")
        records.append({"start_s": float(t[a]), "end_s": float(t[b]), "duration_s": float(t[b] - t[a]),
                        "max_gap_deg": float(g.max()), "worst_joint": JOINTS[worst_j],
                        "queue_min": qmin, "load_max": None if np.isnan(lm) else float(lm),
                        "attribution": why})

    total = sum(r["duration_s"] for r in records)
    print(f"\n停留总计 {total:.1f}s / {t[-1]:.0f}s = {total / t[-1] * 100:.0f}%")
    by = {}
    for r in records:
        by.setdefault(r["attribution"], []).append(r["duration_s"])
    for k, v in sorted(by.items(), key=lambda kv: -sum(kv[1])):
        print(f"  {k}: {len(v)} 段, 共 {sum(v):.1f}s")

    print(f"\n全程 |指令-实测| 逐关节:")
    print(f"{'joint':<18}{'p50':>8}{'p95':>8}{'max':>8}")
    for i, j in enumerate(JOINTS):
        g = np.abs(gap[:, i])
        print(f"{j:<18}{np.median(g):>8.2f}{np.percentile(g, 95):>8.2f}{g.max():>8.2f}")

    # ---------------- figure ----------------
    fig, axes = plt.subplots(5, 1, figsize=(16, 13), sharex=True,
                             gridspec_kw={"height_ratios": [2, 2, 1.3, 1, 1]})
    colors = plt.get_cmap("tab10")

    for i in (1, 2, 5):  # lift, elbow, gripper: the three that actually move
        axes[0].plot(t, cmd[:, i], lw=1.0, color=colors(i), label=f"{SHORT[i]} 指令")
        axes[0].plot(t, pres[:, i], lw=1.0, ls="--", color=colors(i), alpha=0.75, label=f"{SHORT[i]} 实测")
    axes[0].legend(ncol=6, fontsize=8)
    axes[0].set_ylabel("角度 (deg)")
    axes[0].set_title("(a) 指令 vs 实测（实线=模型指令，虚线=机械臂读数）")

    for i in range(6):
        axes[1].plot(t, gap[:, i], lw=0.9, color=colors(i), label=SHORT[i])
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].legend(ncol=6, fontsize=8)
    axes[1].set_ylabel("指令 − 实测 (deg)")
    axes[1].set_title("(b) 跟踪误差：持续非零 = 机械臂跟不上指令")

    axes[2].plot(t, speed, color="tab:gray", lw=1.0)
    axes[2].axhline(args.dwell_speed, color="tab:red", ls="--", lw=1, label=f"停留阈值 {args.dwell_speed}")
    axes[2].set_ylabel("最大关节速度\n(deg/s)")
    axes[2].legend(fontsize=8)
    axes[2].set_title("(c) 实测运动速度")

    axes[3].step(t, q, where="post", color="tab:blue")
    axes[3].fill_between(t, 0, q, step="post", alpha=0.2, color="tab:blue")
    axes[3].axhline(0, color="tab:red", ls="--", lw=1)
    axes[3].set_ylabel("动作队列深度")
    axes[3].set_title("(d) 队列：触 0 才说明停留源于流水线饥饿")

    if len(lt):
        for i in (1, 2, 5):
            axes[4].plot(lt, lv[:, i], lw=1.0, color=colors(i), label=SHORT[i])
        axes[4].axhline(95, color="tab:red", ls="--", lw=1, label="饱和")
        axes[4].legend(ncol=4, fontsize=8)
    axes[4].set_ylabel("|舵机负载|")
    axes[4].set_xlabel("时间 (s)")
    axes[4].set_title("(e) 执行器负载：饱和 = 舵机已尽力")

    for ax in axes:
        for r in records:
            ax.axvspan(r["start_s"], r["end_s"], color="tab:orange", alpha=0.15)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "action_state_gap.png", dpi=140)
    plt.close(fig)

    (out_dir / "dwell_attribution.json").write_text(
        json.dumps({"dwells": records, "total_dwell_s": total,
                    "duration_s": float(t[-1]),
                    "gap_p50_deg": {j: float(np.median(np.abs(gap[:, i]))) for i, j in enumerate(JOINTS)}},
                   indent=2, ensure_ascii=False))
    print(f"\n-> {out_dir}")


if __name__ == "__main__":
    main()
