#!/usr/bin/env python
"""Summarise a live LingBot-VA async run: per-stage latency breakdown + duty cycle.

Reads the JSONL written by ``--record_timeline`` on both sides of the async stack and
reports what the arm actually experienced:

  * the model's own per-stage timings (``model_*``, from
    ``LingBotVAPolicy.last_latency_profile`` via ``PolicyServer._extract_policy_profile``),
  * the server-side envelope around them (prepare / preprocess / postprocess),
  * the client-side transport and execution costs,
  * duty cycle -- commanded actions vs. the number the control rate implies, which is the
    number that explains "the arm barely moves" far better than any single latency figure.

Usage:
    python scripts/inference/analyze_lingbot_va_latency.py logs/async_timeline/lingbot_va
    python scripts/inference/analyze_lingbot_va_latency.py <dir> --json summary.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st
from pathlib import Path

# Stage key -> (display label, note). Order is the order they run in.
STAGES = [
    ("model_text_encode_ms", "Text encoder (UMT5-XXL)", "once per episode"),
    ("model_vae_encode_ms", "VAE encode keyframes", "16 frames x 3 cams"),
    ("model_kv_cache_ms", "KV-cache update", "2 transformer passes"),
    ("model_video_denoise_ms", "Video denoising", "dual-stream DiT"),
    ("model_action_denoise_ms", "Action denoising", "dual-stream DiT"),
    ("model_other_ms", "Scheduler / reshape", "unprofiled remainder"),
]


def load(dirpath: str, pattern: str) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(os.path.join(dirpath, pattern))):
        with open(f) as fh:
            for line in fh:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def stat(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": st.mean(values),
        "median": st.median(values),
        "min": min(values),
        "max": max(values),
    }


def fmt(label: str, s: dict, unit: str = "ms") -> str:
    if not s.get("n"):
        return f"  {label:<32}  (no samples)"
    return (
        f"  {label:<32}{s['mean']:>10.1f}{s['median']:>10.1f}"
        f"{s['min']:>10.1f}{s['max']:>10.1f}  {unit}"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("log_dir", nargs="?", default="logs/async_timeline/lingbot_va")
    p.add_argument("--json", default=None)
    p.add_argument(
        "--skip-first-chunk",
        action="store_true",
        default=True,
        help="Exclude the first chunk from steady-state stats (it pays the one-off prompt encode).",
    )
    args = p.parse_args()

    server = load(args.log_dir, "policy_server_latency_*.jsonl")
    client = load(args.log_dir, "robot_client_latency_*.jsonl")
    if not server:
        raise SystemExit(f"No policy_server_latency_*.jsonl under {args.log_dir}")

    chunks = [r for r in server if r.get("kind") == "server_action_chunk"]
    steady = [c for c in chunks if not c.get("model_first_chunk")]
    use = steady if (args.skip_first_chunk and steady) else chunks

    print(f"\nLingBot-VA async run — {args.log_dir}")
    print(f"chunks: {len(chunks)} total, {len(steady)} steady-state\n")

    print(f"  {'stage':<32}{'mean':>10}{'median':>10}{'min':>10}{'max':>10}")
    print("  " + "-" * 72)
    stage_stats = {}
    for key, label, _note in STAGES:
        vals = [c[key] for c in use if key in c]
        if key == "model_text_encode_ms" and not vals:
            vals = [c[key] for c in chunks if key in c]  # only on the first chunk
        s = stat(vals)
        stage_stats[key] = s
        print(fmt(label, s))

    print()
    envelope = [
        ("model_chunk_total_ms", "Model chunk total"),
        ("prepare_ms", "Server: raw obs -> tensors"),
        ("preprocess_ms", "Server: preprocessor"),
        ("postprocess_ms", "Server: postprocessor (unnorm)"),
        ("policy_predict_ms", "Server: predict total"),
    ]
    env_stats = {}
    for key, label in envelope:
        s = stat([c[key] for c in use if key in c])
        env_stats[key] = s
        print(fmt(label, s))

    print()
    recv = [r for r in client if r.get("kind") == "client_receive_actions"]
    execs = [r for r in client if r.get("kind") == "client_execute_action"]
    caps = [r for r in client if r.get("kind") == "client_capture_observation"]
    client_stats = {}
    for rows, key, label in [
        (recv, "observation_to_actions_ms", "Client: obs -> actions (e2e)"),
        (recv, "server_to_client_ms", "Client: chunk transport"),
        (caps, "capture_ms", "Client: camera capture"),
        (execs, "robot_send_action_ms", "Client: serial write to arm"),
        (execs, "action_age_ms", "Client: action age at execution"),
    ]:
        s = stat([r[key] for r in rows if key in r])
        client_stats[label] = s
        print(fmt(label, s))

    # Duty cycle: the number that actually explains the observed motion.
    summary = {}
    if execs and caps:
        span = max(r["time"] for r in execs) - min(r["time"] for r in caps)
        n_exec = len(execs)
        # The control loop ticks at fps; every tick could have commanded an action.
        loops = len([r for r in client if r.get("kind") == "client_control_loop"])
        print(f"\n  Run span                        {span:>10.1f} s")
        print(f"  Actions actually commanded      {n_exec:>10d}")
        print(f"  Control-loop ticks              {loops:>10d}")
        print(f"  Duty cycle (commanded / ticks)  {100 * n_exec / max(loops, 1):>9.1f} %")
        print(f"  Effective action rate           {n_exec / max(span, 1e-6):>10.2f} Hz")
        summary |= {
            "span_s": span,
            "actions_commanded": n_exec,
            "control_loop_ticks": loops,
            "duty_cycle_pct": 100 * n_exec / max(loops, 1),
            "effective_action_hz": n_exec / max(span, 1e-6),
        }

    total = stage_stats.get("model_chunk_total_ms") or env_stats.get("model_chunk_total_ms")
    if env_stats.get("model_chunk_total_ms", {}).get("n"):
        print(f"\n  Steady-state chunk: {env_stats['model_chunk_total_ms']['mean'] / 1000:.2f} s "
              f"for {16} actions = {16 / (env_stats['model_chunk_total_ms']['mean'] / 1000):.1f} Hz sustainable")

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "log_dir": args.log_dir,
                    "chunks_total": len(chunks),
                    "chunks_steady": len(steady),
                    "stages": {k: stage_stats[k] for k, _, _ in STAGES},
                    "stage_labels": {k: {"label": lb, "note": nt} for k, lb, nt in STAGES},
                    "envelope": env_stats,
                    "client": client_stats,
                    **summary,
                },
                indent=2,
            )
        )
        print(f"\nWrote {args.json}")
    del total


if __name__ == "__main__":
    main()
