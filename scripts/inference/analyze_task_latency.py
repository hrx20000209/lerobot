#!/usr/bin/env python
"""End-to-end latency of one pick-and-place, measured on the robot.

The window is defined the way a user experiences the task: from the first
perception the system ever takes, to the instant the gripper opens to let go.
That deliberately includes everything the arm was doing in between -- approach,
grasp, transport -- and the pipeline overhead underneath it, because that is
what "how long does one task take" actually means.

Grasp and release are recovered from the *measured* gripper opening rather than
the commanded one, so the timing reflects the hardware, not the intent.
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
STAGES = ["image_preproc", "vae_encode", "dit_denoise", "sampler_other", "serialize"]
CN = {"image_preproc": "图像预处理", "vae_encode": "VAE 编码", "dit_denoise": "DiT 去噪",
      "sampler_other": "采样器其他", "serialize": "序列化"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--server-trace", required=True)
    ap.add_argument("--grasp-below", type=float, default=20.0, help="Gripper opening counting as closed")
    ap.add_argument("--release-above", type=float, default=30.0, help="Opening that counts as an approach")
    ap.add_argument("--release-delta", type=float, default=2.0,
                    help="Departure from the pinned held opening that counts as release")
    ap.add_argument("--hold-secs", type=float, default=2.0, help="How long it must stay closed to be a grasp")
    args = ap.parse_args()

    run = Path(args.run_dir)
    out = run / "analysis"
    out.mkdir(exist_ok=True)

    acts = [json.loads(l) for l in open(next(run.glob("action_trace_*.jsonl")))]
    obs = [json.loads(l) for l in open(next(run.glob("observation_trace_*.jsonl")))]
    srv_all = [json.loads(l) for l in open(args.server_trace)]

    t_first_obs = min(o["wall_time"] for o in obs)
    t = np.array([a["exec_start"] for a in acts])
    pres = np.array([[a["present_position"][j] for j in JOINTS] for a in acts])
    cmd = np.array([[a["commanded_action"][j] for j in JOINTS] for a in acts])
    grip = pres[:, 5]
    rel = t - t_first_obs  # everything measured from the first perception

    # --- grasp ---
    # The gripper starts the run *closed and empty*, so "first sustained close"
    # would fire at t=0 on the initial pose. A grasp is only a grasp after the
    # gripper has opened to approach the object, so find that opening first.
    opened = np.where(grip > args.release_above)[0]
    if not len(opened):
        raise SystemExit("gripper never opened; nothing was grasped")
    approach_i = int(opened[0])

    grasp_i = None
    closed = grip < args.grasp_below
    i = approach_i
    while i < len(t):
        if closed[i]:
            j = i
            while j < len(t) and closed[j]:
                j += 1
            if t[j - 1] - t[i] >= args.hold_secs:
                grasp_i = i
                break
            i = j
        else:
            i += 1
    if grasp_i is None:
        raise SystemExit("no sustained grasp found after the approach")

    # --- release ---
    # A fixed "opened past N degrees" test misses it: to drop the cube the
    # gripper only has to widen past the cube, which here is ~5 deg, never
    # approaching any sane open threshold. The reliable signal is that while a
    # rigid object is held the measured opening is *pinned* by it -- exactly
    # 14.92 for 5.5 s in this run -- so the release is the first departure from
    # that value in either direction. Closing further is the strongest evidence
    # of all: the fingers can only travel past where the cube was if it is gone.
    # Find the pinned plateau: the longest stretch after the grasp over which the
    # opening does not move. Anchoring on the grasp instant instead would read
    # the value while the fingers are still closing and fire immediately.
    best = (0, grasp_i, grasp_i)
    i = grasp_i
    while i < len(t):
        j = i + 1
        while j < len(t) and abs(grip[j] - grip[i]) <= 0.5:
            j += 1
        if t[j - 1] - t[i] > best[0]:
            best = (t[j - 1] - t[i], i, j - 1)
        i = j
    plateau_s, plat_lo, plat_hi = best
    if plateau_s < args.hold_secs:
        raise SystemExit(f"no pinned plateau after the grasp (longest {plateau_s:.1f}s)")
    hold = float(np.median(grip[plat_lo:plat_hi + 1]))
    release_i = int(plat_hi) + 1
    if release_i >= len(t):
        raise SystemExit("the run ended while the object was still held")
    print(f"（夹爪被方块卡定在 {hold:.2f}，持续 {plateau_s:.1f}s：{rel[plat_lo]:.1f}–{rel[plat_hi]:.1f}s）")
    # Follow the excursion to its extreme, i.e. fully let go.
    k = release_i
    while k + 1 < len(t) and abs(grip[k + 1] - hold) >= abs(grip[k] - hold):
        k += 1
    peak_i = k

    t_grasp, t_release, t_peak = rel[grasp_i], rel[release_i], rel[peak_i]
    # Count by when the server *started* a chunk, and bound the window by the
    # reported end-to-end figure (the release) rather than the excursion peak,
    # so the count and the percentages share one denominator.
    srv = [r for r in srv_all if t_first_obs <= r["server_recv_time"] <= t[release_i]]
    busy_s = sum(r["total_server_ms"] for r in srv) / 1000
    serve_lo = min(r["server_recv_time"] for r in srv) - t_first_obs
    serve_hi = max(r["server_reply_time"] for r in srv) - t_first_obs

    print(f"首次 perception        t = 0.00 s   (wall {t_first_obs:.3f})")
    print(f"首个动作落地           t = {rel[0]:6.2f} s")
    print(f"夹爪张开（接近）       t = {rel[approach_i]:6.2f} s")
    print(f"抓取（夹爪持续闭合）   t = {t_grasp:6.2f} s")
    print(f"夹爪开始松开           t = {t_release:6.2f} s")
    print(f"夹爪张到最大           t = {t_peak:6.2f} s  (开度 {grip[peak_i]:.1f})")
    print(f"\n=== end-to-end latency (首次 perception → 夹爪松开) = {t_release:.2f} s ===")
    print(f"    （到完全张开为 {t_peak:.2f} s）")
    print(f"\n分解：")
    print(f"  感知启动 → 首个动作     {rel[0]:6.2f} s   ({100 * rel[0] / t_release:4.1f}%)  流水线预热")
    print(f"  接近 + 抓取             {t_grasp - rel[0]:6.2f} s   ({100 * (t_grasp - rel[0]) / t_release:4.1f}%)")
    print(f"  搬运 → 松开             {t_release - t_grasp:6.2f} s   ({100 * (t_release - t_grasp) / t_release:4.1f}%)")
    print(f"\nWAM 推理次数（窗口内）  {len(srv)}")
    print(f"  单次耗时               中位 {np.median([r['total_server_ms'] for r in srv]):.0f} / "
          f"均值 {np.mean([r['total_server_ms'] for r in srv]):.0f} / 最大 {max(r['total_server_ms'] for r in srv):.0f} ms")
    print(f"  推理累计               {busy_s:.1f} s")
    # Against the period the server was actually in service -- dividing by the
    # whole window would credit it for time before the first observation landed.
    print(f"  服务端忙碌率           {100 * busy_s / (serve_hi - serve_lo):.0f}%  "
          f"(服务期 {serve_lo:.2f}–{serve_hi:.2f} s)")
    print(f"  执行动作数             {int((rel <= t_release).sum())}")

    # ---------------- figure 1: actions ----------------
    m = rel <= t_peak + 2
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.4, 1]})
    colors = plt.get_cmap("tab10")
    for i in (1, 2):
        axes[0].plot(rel[m], cmd[m, i], lw=1.1, color=colors(i), label=f"{SHORT[i]} 指令")
        axes[0].plot(rel[m], pres[m, i], lw=1.1, ls="--", color=colors(i), alpha=0.75, label=f"{SHORT[i]} 实测")
    axes[0].legend(ncol=4, fontsize=8); axes[0].set_ylabel("角度 (deg)")
    axes[0].set_title("(a) 指令 vs 实测（实线=模型指令，虚线=机械臂读数）")

    axes[1].plot(rel[m], cmd[m, 5], lw=1.3, color="tab:brown", label="gripper 指令")
    axes[1].plot(rel[m], grip[m], lw=1.3, ls="--", color="tab:orange", label="gripper 实测")
    axes[1].axhline(args.grasp_below, color="gray", ls=":", lw=1)
    axes[1].axhline(args.release_above, color="gray", ls=":", lw=1)
    axes[1].legend(fontsize=8); axes[1].set_ylabel("gripper (0-100)")
    axes[1].set_title("(b) 夹爪：抓取与松开的判定")

    q = np.array([a["queue_size_before"] for a in acts])
    axes[2].step(rel[m], q[m], where="post", color="tab:blue")
    axes[2].fill_between(rel[m], 0, q[m], step="post", alpha=0.2, color="tab:blue")
    axes[2].set_ylabel("动作队列深度"); axes[2].set_xlabel("自首次 perception 起的时间 (s)")
    axes[2].set_title("(c) 队列深度")

    for ax in axes:
        ax.axvline(0, color="tab:green", lw=1.5, ls="-")
        ax.axvline(t_grasp, color="tab:blue", lw=1.5, ls="-.")
        ax.axvline(t_release, color="tab:red", lw=2)
        ax.grid(alpha=0.3)
    axes[0].text(0.3, axes[0].get_ylim()[1] * 0.95, "首次 perception", color="tab:green", fontsize=9)
    axes[0].text(t_grasp + 0.3, axes[0].get_ylim()[1] * 0.95, "抓取", color="tab:blue", fontsize=9)
    axes[0].text(t_release + 0.3, axes[0].get_ylim()[1] * 0.95, "松开", color="tab:red", fontsize=9)
    fig.suptitle(f"一次取放的动作过程　end-to-end = {t_release:.1f} s", y=1.00)
    fig.tight_layout(); fig.savefig(out / "task_actions.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    # ---------------- figure 2: latency breakdown ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    segs = [("流水线预热\n(感知→首动作)", rel[0]), ("接近 + 抓取", t_grasp - rel[0]),
            ("搬运 → 松开", t_release - t_grasp)]  # 三段和为 end-to-end
    left = 0
    for i, (lab, v) in enumerate(segs):
        ax.barh(0, v, left=left, height=0.5, color=plt.get_cmap("Set2")(i), label=f"{lab}  {v:.1f}s")
        ax.text(left + v / 2, 0, f"{v:.1f}s\n{100 * v / t_release:.0f}%", ha="center", va="center", fontsize=9)
        left += v
    ax.set_yticks([]); ax.set_xlabel("秒")
    ax.set_title(f"(a) 任务阶段分解　总计 {t_release:.1f} s")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22))

    ax = axes[1]
    vals = [np.median([r["stages_ms"].get(s, 0) for r in srv]) for s in STAGES]
    tot = np.median([r["total_server_ms"] for r in srv])
    ax.barh([CN[s] for s in STAGES], vals, color=[plt.get_cmap("tab10")(i) for i in range(len(STAGES))])
    for i, v in enumerate(vals):
        if v > 1:
            ax.text(v + tot * 0.02, i, f"{v:.0f} ms ({100 * v / tot:.0f}%)", va="center", fontsize=8)
    ax.set_xlabel("毫秒（中位数）")
    ax.set_title(f"(b) 单次推理阶段预算　{tot:.0f} ms × {len(srv)} 次")

    ax = axes[2]
    ax.bar(["墙钟时间", "推理累计"], [t_release, busy_s],
           color=["tab:gray", "tab:red"])
    for i, v in enumerate([t_release, busy_s]):
        ax.text(i, v + 0.6, f"{v:.1f}s", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("秒")
    ax.set_title(f"(c) 服务端忙碌率 {100 * busy_s / (serve_hi - serve_lo):.0f}%")
    for a in axes:
        a.grid(alpha=0.3, axis="x" if a is not axes[2] else "y")
    fig.tight_layout(); fig.savefig(out / "task_latency_breakdown.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    (out / "task_latency.json").write_text(json.dumps({
        "t_first_observation": t_first_obs,
        "first_action_s": float(rel[0]), "grasp_s": float(t_grasp),
        "release_s": float(t_release), "release_peak_s": float(t_peak),
        "end_to_end_s": float(t_release),
        "n_inferences": len(srv),
        "median_inference_ms": float(np.median([r["total_server_ms"] for r in srv])),
        "inference_total_s": float(busy_s),
        "server_busy_fraction": float(busy_s / (serve_hi - serve_lo)),
        "n_actions": int((rel <= t_release).sum()),
    }, indent=2, ensure_ascii=False))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
