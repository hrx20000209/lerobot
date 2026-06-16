#!/usr/bin/env python
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Plot async inference observation, VLA inference, and action timelines."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _resolve_inputs(inputs: list[Path]) -> list[Path]:
    paths = []
    for item in inputs:
        if item.is_dir():
            paths.extend(sorted(item.glob("*_latency_*.jsonl")))
        else:
            paths.append(item)
    return [path for path in paths if path.is_file()]


def _event_interval(event: dict[str, Any]) -> tuple[float, float] | None:
    start = event.get("start_time")
    end = event.get("end_time")
    if not isinstance(start, int | float) or not isinstance(end, int | float):
        return None
    if end < start:
        return None
    return float(start), float(end)


def _dedupe_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    unique = []
    for event in events:
        key = (
            event.get("kind"),
            event.get("timestep"),
            round(float(event.get("start_time", 0.0)), 6),
            round(float(event.get("end_time", 0.0)), 6),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(event)
    return unique


def _overlap_ms(interval: tuple[float, float], other_intervals: list[tuple[float, float]]) -> float:
    total = 0.0
    start, end = interval
    for other_start, other_end in other_intervals:
        total += max(0.0, min(end, other_end) - max(start, other_start))
    return total * 1000.0


def _build_action_runtime_intervals(actions: list[dict[str, Any]], max_step_s: float | None):
    actions = sorted(actions, key=lambda event: (event.get("start_time", 0.0), event.get("timestep", -1)))
    intervals = []
    for index, event in enumerate(actions):
        interval = _event_interval(event)
        if interval is None:
            continue

        start, end = interval
        if index + 1 < len(actions):
            display_end = float(actions[index + 1].get("start_time", end))
        elif max_step_s is not None:
            display_end = start + max_step_s
        else:
            display_end = end

        if max_step_s is not None:
            display_end = min(display_end, start + max_step_s)
        display_end = max(display_end, end)
        intervals.append((event, (start, display_end), (start, end)))
    return intervals


def _write_summary(
    output_csv: Path,
    action_intervals: list[tuple[dict[str, Any], tuple[float, float], tuple[float, float]]],
    inference_intervals: list[tuple[float, float]],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestep",
                "action_start_s",
                "action_end_s",
                "action_runtime_ms",
                "send_action_ms",
                "inference_overlap_ms",
                "source_observation_timestep",
                "source_observation_age_ms",
            ],
        )
        writer.writeheader()
        if not action_intervals:
            return
        t0 = min(interval[1][0] for interval in action_intervals)
        for event, runtime_interval, send_interval in action_intervals:
            runtime_ms = (runtime_interval[1] - runtime_interval[0]) * 1000.0
            send_ms = (send_interval[1] - send_interval[0]) * 1000.0
            writer.writerow(
                {
                    "timestep": event.get("timestep"),
                    "action_start_s": runtime_interval[0] - t0,
                    "action_end_s": runtime_interval[1] - t0,
                    "action_runtime_ms": runtime_ms,
                    "send_action_ms": send_ms,
                    "inference_overlap_ms": _overlap_ms(runtime_interval, inference_intervals),
                    "source_observation_timestep": event.get("source_observation_timestep"),
                    "source_observation_age_ms": event.get("source_observation_age_ms"),
                }
            )


def plot_timeline(
    records: list[dict[str, Any]],
    output_png: Path,
    output_csv: Path | None = None,
    max_action_step_s: float | None = 0.2,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    observations = _dedupe_events([r for r in records if r.get("kind") == "timeline_observation"])
    inferences = _dedupe_events([r for r in records if r.get("kind") == "timeline_vla_inference"])
    actions = _dedupe_events([r for r in records if r.get("kind") == "timeline_action"])

    all_interval_events = observations + inferences + actions
    intervals = [_event_interval(event) for event in all_interval_events]
    intervals = [interval for interval in intervals if interval is not None]
    if not intervals:
        raise ValueError("No timeline events found. Run async inference after this instrumentation change.")

    t0 = min(start for start, _ in intervals)
    inference_intervals = [interval for event in inferences if (interval := _event_interval(event))]
    action_intervals = _build_action_runtime_intervals(actions, max_action_step_s)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    if output_csv is not None:
        _write_summary(output_csv, action_intervals, inference_intervals)

    fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)
    lanes = {"Observation": 2.0, "VLA inference": 1.0, "Action": 0.0}

    for event in sorted(observations, key=lambda r: r.get("start_time", 0.0)):
        interval = _event_interval(event)
        if interval is None:
            continue
        start, end = interval
        x = start - t0
        width = max(end - start, 0.002)
        color = "#4C78A8" if int(event.get("must_go", 0)) == 0 else "#1F4E79"
        ax.broken_barh([(x, width)], (lanes["Observation"] - 0.14, 0.28), facecolors=color)
        ax.plot([float(event.get("observation_time", end)) - t0], [lanes["Observation"]], "o", color=color, ms=4)

    for event in sorted(inferences, key=lambda r: r.get("start_time", 0.0)):
        interval = _event_interval(event)
        if interval is None:
            continue
        start, end = interval
        ax.broken_barh(
            [(start - t0, max(end - start, 0.002))],
            (lanes["VLA inference"] - 0.18, 0.36),
            facecolors="#F58518",
            alpha=0.85,
        )
        label_x = start - t0
        ax.text(
            label_x,
            lanes["VLA inference"] + 0.23,
            f"obs {event.get('timestep', '')}",
            fontsize=7,
            color="#7A3B00",
            clip_on=True,
        )

    for event, runtime_interval, send_interval in action_intervals:
        runtime_start, runtime_end = runtime_interval
        send_start, send_end = send_interval
        overlap = _overlap_ms(runtime_interval, inference_intervals)
        ax.broken_barh(
            [(runtime_start - t0, max(runtime_end - runtime_start, 0.002))],
            (lanes["Action"] - 0.18, 0.36),
            facecolors="#54A24B",
            alpha=0.55,
        )
        ax.broken_barh(
            [(send_start - t0, max(send_end - send_start, 0.002))],
            (lanes["Action"] - 0.23, 0.46),
            facecolors="#2F6B2F",
            alpha=0.9,
        )
        if overlap > 0:
            ax.text(
                runtime_start - t0,
                lanes["Action"] - 0.42,
                f"{overlap:.0f}ms",
                fontsize=7,
                color="#245124",
                clip_on=True,
            )

    ax.set_yticks(list(lanes.values()), list(lanes.keys()))
    ax.set_xlabel("Time since first event (s)")
    ax.set_title("Async Inference Timeline")
    ax.grid(axis="x", alpha=0.25)
    ax.set_ylim(-0.65, 2.6)

    legend_items = [
        Patch(facecolor="#4C78A8", label="Observation capture"),
        Patch(facecolor="#1F4E79", label="Must-go observation"),
        Patch(facecolor="#F58518", label="VLA inference"),
        Patch(facecolor="#54A24B", alpha=0.55, label="Action runtime until next command"),
        Patch(facecolor="#2F6B2F", label="send_action() call"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#4C78A8", label="Observation timestamp"),
    ]
    ax.legend(handles=legend_items, loc="upper right", ncols=2, fontsize=8)

    fig.savefig(output_png, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="JSONL file(s) or a logs directory.")
    parser.add_argument("--output", type=Path, default=Path("logs/async_timeline.png"))
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional CSV with per-action inference overlap and observation age.",
    )
    parser.add_argument(
        "--max-action-step-s",
        type=float,
        default=0.2,
        help="Cap inferred action runtime when drawing action bars. Use <=0 to disable.",
    )
    args = parser.parse_args()

    paths = _resolve_inputs(args.inputs)
    if not paths:
        raise FileNotFoundError(f"No JSONL latency files found in: {args.inputs}")

    records = []
    for path in paths:
        records.extend(_iter_jsonl(path))

    max_step = args.max_action_step_s if args.max_action_step_s > 0 else None
    plot_timeline(records, args.output, args.summary_csv, max_step)
    print(f"Wrote {args.output}")
    if args.summary_csv is not None:
        print(f"Wrote {args.summary_csv}")


if __name__ == "__main__":
    main()
