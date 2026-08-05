#!/usr/bin/env python
"""Analyse a profiled Cosmos SO101 run: stage budget, async timeline, dwell phases.

Consumes the three traces a profiled run produces:
  <run>/action_trace_*.jsonl        one record per executed action (client)
  <run>/observation_trace_*.jsonl   one record per captured observation (client)
  <server trace>.jsonl              one record per served chunk (server stages)

Emits a stage-budget table, an async timeline figure, a joint/dwell figure, and
a JSON summary.  Dwell phases are found from the measured joint positions, so
they reflect what the arm actually did rather than what was commanded.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Labels are Chinese; DejaVu Sans has no CJK glyphs and would render tofu.
# The Noto .ttc collections register under whichever face comes first (often
# "... CJK JP"), but the Han glyphs are shared, so that face renders Chinese.
for _f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"):
    if Path(_f).exists():
        try:
            fm.fontManager.addfont(_f)
        except Exception:  # noqa: BLE001
            pass
_have = {f.name for f in fm.fontManager.ttflist}
for _cand in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Serif CJK SC",
              "Noto Serif CJK JP", "Droid Sans Fallback", "WenQuanYi Zen Hei"):
    if _cand in _have:
        plt.rcParams["font.sans-serif"] = [_cand, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

JOINTS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]
STAGES = [
    "obs_prep",
    "t5_lookup",
    "image_preproc",
    "vae_encode",
    "dit_denoise",
    "sampler_other",
    "action_decode",
    "safety_filter",
    "serialize",
]
STAGE_CN = {
    "obs_prep": "观测预处理",
    "t5_lookup": "T5 文本嵌入",
    "image_preproc": "图像预处理",
    "vae_encode": "VAE 编码",
    "dit_denoise": "DiT 去噪",
    "sampler_other": "采样器其他",
    "action_decode": "动作解码",
    "safety_filter": "安全钳位",
    "serialize": "序列化",
}


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def pct(vals, q):
    return float(np.percentile(vals, q)) if len(vals) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--server-trace", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--dwell-window-s", type=float, default=1.0, help="Window for the motion-speed estimate")
    ap.add_argument("--dwell-speed-thresh", type=float, default=1.5, help="deg/s below which the arm counts as dwelling")
    ap.add_argument("--dwell-min-s", type=float, default=1.0, help="Shortest run of slow motion to report")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir or run_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    acts = load(next(run_dir.glob("action_trace_*.jsonl")))
    obs = load(next(run_dir.glob("observation_trace_*.jsonl")))
    srv_all = load(args.server_trace)

    t0 = min(a["exec_start"] for a in acts)
    t_end = max(a["exec_end"] for a in acts)
    # One long-lived server can serve several runs, so its trace accumulates
    # chunks from all of them. Keep only what belongs to this run's window.
    srv = [r for r in srv_all if t0 - 5.0 <= r["server_recv_time"] <= t_end + 1.0]
    if len(srv) != len(srv_all):
        print(f"(server trace: kept {len(srv)}/{len(srv_all)} chunks inside this run's window)")
    if not srv:
        raise SystemExit("No server chunks overlap this run; wrong --server-trace?")
    at = np.array([a["exec_start"] - t0 for a in acts])
    pres = np.array([[a["present_position"][j] for j in JOINTS] for a in acts if a["present_position"]])
    pres_t = np.array([a["exec_start"] - t0 for a in acts if a["present_position"]])
    cmd = np.array([[a["commanded_action"][j] for j in JOINTS] for a in acts])
    ages = np.array([a["observation_to_execution_ms"] or np.nan for a in acts], dtype=float)
    qsz = np.array([a["queue_size_before"] for a in acts])

    summary = {"run_dir": str(run_dir), "n_actions": len(acts), "n_observations": len(obs), "n_chunks": len(srv)}

    # ---------- 1. server stage budget ----------
    total = np.array([r["total_server_ms"] for r in srv])
    stage_p50 = {}
    for s in STAGES:
        v = np.array([r["stages_ms"].get(s, 0.0) for r in srv])
        stage_p50[s] = float(np.median(v))
    unacc = float(np.median([r["unaccounted_ms"] for r in srv]))
    tot_p50 = float(np.median(total))

    print(f"\n=== 1. 服务端单次推理阶段分解 (n={len(srv)} chunks) ===")
    print(f"{'stage':<16}{'p50 ms':>10}{'share':>9}")
    for s in STAGES:
        print(f"{STAGE_CN[s]:<16}{stage_p50[s]:>10.1f}{stage_p50[s] / tot_p50 * 100:>8.1f}%")
    print(f"{'未归因':<16}{unacc:>10.1f}{unacc / tot_p50 * 100:>8.1f}%")
    print(f"{'合计':<16}{tot_p50:>10.1f}")
    summary["server_stage_p50_ms"] = stage_p50
    summary["server_total_p50_ms"] = tot_p50
    summary["server_unaccounted_p50_ms"] = unacc

    # ---------- 2. end-to-end budget ----------
    print(f"\n=== 2. 端到端延迟 ===")
    for label, v in [
        ("observation -> execution (ms)", ages[~np.isnan(ages)]),
        ("action queue depth", qsz),
    ]:
        print(f"  {label:<32} p50={pct(v, 50):8.1f}  p90={pct(v, 90):8.1f}  max={np.nanmax(v):8.1f}")
    dt = np.diff(at) * 1000
    print(f"  {'command interval (ms)':<32} p50={pct(dt, 50):8.1f}  p90={pct(dt, 90):8.1f}  max={dt.max():8.1f}")
    starve = float((qsz == 0).mean() * 100)
    print(f"  queue empty at pop: {starve:.1f}% of actions")
    eff_hz = len(acts) / (at[-1] - at[0])
    print(f"  effective command rate: {eff_hz:.2f} Hz over {at[-1] - at[0]:.1f} s")
    summary.update(
        observation_to_execution_p50_ms=pct(ages[~np.isnan(ages)], 50),
        observation_to_execution_p90_ms=pct(ages[~np.isnan(ages)], 90),
        command_interval_p50_ms=pct(dt, 50),
        queue_empty_pct=starve,
        effective_hz=eff_hz,
        duration_s=float(at[-1] - at[0]),
    )

    # ---------- 3. motion + dwell ----------
    travel = pres.max(axis=0) - pres.min(axis=0)
    print(f"\n=== 3. 实测运动 ===")
    for j, tv in zip(JOINTS, travel):
        print(f"  {j:<20} 行程 {tv:7.2f} deg")

    # Speed from measured joint positions, smoothed over dwell-window.
    speed = np.zeros(len(pres_t))
    for i, t in enumerate(pres_t):
        m = (pres_t >= t - args.dwell_window_s / 2) & (pres_t <= t + args.dwell_window_s / 2)
        if m.sum() >= 2:
            seg, segt = pres[m], pres_t[m]
            span = segt[-1] - segt[0]
            if span > 1e-3:
                speed[i] = np.abs(seg[-1] - seg[0]).max() / span
    dwelling = speed < args.dwell_speed_thresh
    phases, start = [], None
    for i, d in enumerate(dwelling):
        if d and start is None:
            start = i
        elif not d and start is not None:
            if pres_t[i - 1] - pres_t[start] >= args.dwell_min_s:
                phases.append((pres_t[start], pres_t[i - 1]))
            start = None
    if start is not None and pres_t[-1] - pres_t[start] >= args.dwell_min_s:
        phases.append((pres_t[start], pres_t[-1]))

    print(f"\n=== 4. 停留期 (速度 < {args.dwell_speed_thresh} deg/s, 持续 >= {args.dwell_min_s}s) ===")
    dwell_records = []
    for s, e in phases:
        m = (pres_t >= s) & (pres_t <= e)
        grip = pres[m][:, 5]
        rec = {
            "start_s": float(s),
            "end_s": float(e),
            "duration_s": float(e - s),
            "gripper_mean": float(grip.mean()),
            "gripper_change": float(grip[-1] - grip[0]),
        }
        dwell_records.append(rec)
        print(
            f"  {s:6.1f}s -> {e:6.1f}s  ({e - s:5.1f}s)  gripper {grip[0]:5.1f} -> {grip[-1]:5.1f}"
        )
    total_dwell = sum(r["duration_s"] for r in dwell_records)
    print(f"  停留总时长 {total_dwell:.1f}s / {at[-1] - at[0]:.1f}s = {total_dwell / (at[-1] - at[0]) * 100:.0f}%")
    summary["dwell_phases"] = dwell_records
    summary["dwell_total_s"] = total_dwell
    summary["joint_travel_deg"] = {j: float(t) for j, t in zip(JOINTS, travel)}

    # ---------- figures ----------
    _fig_timeline(srv, acts, obs, t0, at, qsz, ages, out_dir, stage_p50, tot_p50)
    _fig_motion(pres_t, pres, speed, phases, args, out_dir)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n figures + summary -> {out_dir}")


def _fig_timeline(srv, acts, obs, t0, at, qsz, ages, out_dir, stage_p50, tot_p50):
    fig, axes = plt.subplots(4, 1, figsize=(15, 11), sharex=False,
                             gridspec_kw={"height_ratios": [1.5, 1, 1, 1.2]})

    # (a) async gantt over a zoom window -- at full-run scale each 844 ms chunk
    # is a hairline and the pipeline structure is invisible.
    ax = axes[0]
    colors = plt.get_cmap("tab10")
    stage_colors = {s: colors(i % 10) for i, s in enumerate(STAGES)}
    span = at[-1] - at[0]
    zoom_lo = at[0] + span * 0.45
    zoom_hi = min(zoom_lo + 8.0, at[-1])
    for r in srv:
        x = r["server_recv_time"] - t0
        if x < zoom_lo - 2 or x > zoom_hi + 2:
            continue
        for s in STAGES:
            w = r["stages_ms"].get(s, 0.0) / 1000.0
            if w <= 0:
                continue
            ax.barh(0.35, w, left=x, height=0.5, color=stage_colors[s], edgecolor="none", label=STAGE_CN[s])
            x += w
    obs_t = np.array([o["wall_time"] - t0 for o in obs])
    m = (obs_t >= zoom_lo) & (obs_t <= zoom_hi)
    ax.plot(obs_t[m], np.full(m.sum(), 0.9), "|", color="tab:green", ms=10, label="观测采集")
    ma = (at >= zoom_lo) & (at <= zoom_hi)
    ax.plot(at[ma], np.full(ma.sum(), -0.35), "|", color="tab:red", ms=10, label="动作执行")
    # Link each executed action to the observation it was computed from.
    for a in acts:
        x = a["exec_start"] - t0
        s_ts = a["source_observation_timestamp"]
        if s_ts is None or not (zoom_lo <= x <= zoom_hi):
            continue
        ax.plot([s_ts - t0, x], [0.9, -0.35], color="gray", lw=0.4, alpha=0.35)
    h, l = ax.get_legend_handles_labels()
    seen = {}
    for hh, ll in zip(h, l):
        seen.setdefault(ll, hh)
    ax.legend(seen.values(), seen.keys(), ncol=6, fontsize=7.5, loc="lower center",
              bbox_to_anchor=(0.5, 1.02), frameon=False)
    ax.set_xlim(zoom_lo, zoom_hi)
    ax.set_ylim(-0.8, 1.3)
    ax.set_yticks([0.9, 0.35, -0.35])
    ax.set_yticklabels(["观测", "服务端推理", "执行"], fontsize=9)
    ax.set_title(f"(a) 异步流水线放大视图 [{zoom_lo:.0f}-{zoom_hi:.0f}s]；灰线 = 观测到其动作的因果链", pad=34)

    # (b) queue depth
    axes[1].step(at, qsz, where="post", color="tab:blue")
    axes[1].fill_between(at, 0, qsz, step="post", alpha=0.25, color="tab:blue")
    axes[1].axhline(0, color="tab:red", lw=1, ls="--")
    axes[1].set_ylabel("动作队列深度")
    axes[1].set_title("(b) 动作队列深度：始终 >0 说明没有饥饿，但排队本身就是延迟", fontsize=10)

    # (c) observation age at execution
    axes[2].plot(at, ages, ".", ms=3, color="tab:purple")
    axes[2].set_ylabel("观测→执行\n延迟 (ms)")
    axes[2].axhline(np.nanmedian(ages), color="k", ls="--", lw=1,
                    label=f"p50={np.nanmedian(ages):.0f} ms")
    axes[2].legend(fontsize=8)
    axes[2].set_title("(c) 每个动作执行时，其依据的观测已经多旧", fontsize=10)

    # (d) stage budget bar
    ax = axes[3]
    vals = [stage_p50[s] for s in STAGES]
    ax.barh([STAGE_CN[s] for s in STAGES], vals, color=[stage_colors[s] for s in STAGES])
    for i, v in enumerate(vals):
        if v > 0.5:
            ax.text(v + tot_p50 * 0.01, i, f"{v:.0f} ms ({v / tot_p50 * 100:.0f}%)", va="center", fontsize=8)
    ax.set_xlabel("毫秒 (中位数)")
    ax.set_title(f"(d) 单次推理阶段预算，合计 {tot_p50:.0f} ms", fontsize=10)

    for a in axes[:3]:
        a.set_xlabel("时间 (s)")
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "async_timeline.png", dpi=140)
    plt.close(fig)


def _fig_motion(pres_t, pres, speed, phases, args, out_dir):
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    for i, j in enumerate(JOINTS[:5]):
        axes[0].plot(pres_t, pres[:, i], lw=1.2, label=j)
    axes[0].legend(ncol=5, fontsize=8)
    axes[0].set_ylabel("关节角 (deg)")
    axes[0].set_title("实测关节轨迹与停留期")

    axes[1].plot(pres_t, pres[:, 5], color="tab:brown", lw=1.4)
    axes[1].set_ylabel("gripper (0-100)")

    axes[2].plot(pres_t, speed, color="tab:gray", lw=1.2)
    axes[2].axhline(args.dwell_speed_thresh, color="tab:red", ls="--", lw=1,
                    label=f"停留阈值 {args.dwell_speed_thresh} deg/s")
    axes[2].set_ylabel("最大关节速度\n(deg/s)")
    axes[2].set_xlabel("时间 (s)")
    axes[2].legend(fontsize=8)

    for a in axes:
        for s, e in phases:
            a.axvspan(s, e, color="tab:orange", alpha=0.18)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "motion_and_dwell.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
