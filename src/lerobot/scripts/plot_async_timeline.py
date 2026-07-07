# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Plot continuous async inference timeline logs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Sequence


def _load_events(log_path: str | Path | Sequence[str | Path]) -> list[dict[str, Any]]:
    if isinstance(log_path, str | Path):
        paths = [log_path]
    else:
        paths = list(log_path)
    events = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    event = json.loads(line)
                    event["_source_log_path"] = str(path)
                    events.append(event)
    return _normalize_events(events)


def _record_ts(event: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        value = event.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            return float(value)
    value = event.get("monotonic_ts")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    value = event.get("time")
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return None


def _normalized_event(event_type: str, source: dict[str, Any], **fields: Any) -> dict[str, Any]:
    monotonic_ts = fields.get("monotonic_ts")
    if not isinstance(monotonic_ts, int | float) or isinstance(monotonic_ts, bool):
        monotonic_ts = _record_ts(source)
    return {
        "event_type": event_type,
        "monotonic_ts": monotonic_ts,
        "wall_time": source.get("wall_time", source.get("time")),
        "_source_log_path": source.get("_source_log_path"),
        **fields,
    }


def _normalize_latency_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    inference_ids: dict[tuple[Any, Any, Any], int] = {}

    def inference_id_for(event: dict[str, Any]) -> int:
        key = (event.get("timestep"), event.get("start_time"), event.get("end_time"))
        if key not in inference_ids:
            inference_ids[key] = len(inference_ids)
        return inference_ids[key]

    for event in events:
        kind = event.get("kind")
        if kind == "timeline_observation":
            obs_id = event.get("timestep")
            capture_end_ts = _record_ts(event, "end_time", "observation_time", "time")
            normalized.append(
                _normalized_event(
                    "observation_capture_end",
                    event,
                    monotonic_ts=capture_end_ts,
                    obs_id=obs_id,
                    obs_capture_start_ts=_record_ts(event, "start_time"),
                    obs_capture_end_ts=capture_end_ts,
                    queue_size=event.get("queue_size"),
                    must_go=event.get("must_go"),
                    image_paths=event.get("image_paths"),
                )
            )
        elif kind == "timeline_server_inference":
            inference_id = inference_id_for(event)
            source_obs_id = event.get("timestep")
            start_ts = _record_ts(event, "start_time")
            end_ts = _record_ts(event, "end_time")
            normalized.append(
                _normalized_event(
                    "inference_started",
                    event,
                    monotonic_ts=start_ts,
                    inference_id=inference_id,
                    source_obs_id=source_obs_id,
                    obs_id=source_obs_id,
                    inference_start_ts=start_ts,
                    source_obs_capture_ts=_record_ts(event, "observation_time"),
                    first_timestep=event.get("first_timestep"),
                    last_timestep=event.get("last_timestep"),
                )
            )
            latency = end_ts - start_ts if isinstance(start_ts, float) and isinstance(end_ts, float) else None
            normalized.append(
                _normalized_event(
                    "inference_finished",
                    event,
                    monotonic_ts=end_ts,
                    inference_id=inference_id,
                    source_obs_id=source_obs_id,
                    obs_id=source_obs_id,
                    inference_start_ts=start_ts,
                    inference_end_ts=end_ts,
                    action_chunk_size=event.get("actions_count"),
                    model_latency=latency,
                    first_timestep=event.get("first_timestep"),
                    last_timestep=event.get("last_timestep"),
                )
            )
        elif kind == "timeline_receive_actions":
            source_obs_id = event.get("first_timestep")
            normalized.append(
                _normalized_event(
                    "queue_update_finished",
                    event,
                    monotonic_ts=_record_ts(event, "receive_time", "time"),
                    inference_id=source_obs_id,
                    source_obs_id=source_obs_id,
                    aggregation_fn="client_aggregate",
                    old_queue_length=event.get("queue_size_before"),
                    new_queue_length=event.get("queue_size_after"),
                    num_actions_replaced=None,
                    num_actions_blended=None,
                    first_timestep=event.get("first_timestep"),
                    last_timestep=event.get("last_timestep"),
                    actions_count=event.get("actions_count"),
                )
            )
        elif kind == "timeline_action":
            timestep = event.get("timestep")
            source_obs_id = event.get("source_observation_timestep")
            if source_obs_id is None:
                source_obs_id = event.get("timestep")
            try:
                chunk_index = int(timestep) - int(source_obs_id)
            except (TypeError, ValueError):
                chunk_index = None
            normalized.append(
                _normalized_event(
                    "action_execution_finished",
                    event,
                    monotonic_ts=_record_ts(event, "end_time", "time"),
                    action_id=timestep,
                    source_obs_id=source_obs_id,
                    source_inference_id=source_obs_id,
                    chunk_index=chunk_index,
                    predicted_for_step_index=timestep,
                    actual_exec_start_ts=_record_ts(event, "start_time"),
                    actual_exec_end_ts=_record_ts(event, "end_time"),
                    queue_size_before=event.get("queue_size_before"),
                    queue_size_after=event.get("queue_size_after"),
                )
            )
        else:
            normalized.append(event)

    return normalized


def _normalize_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if any("event_type" in event for event in events):
        normalized = []
        for event in events:
            if "event_type" in event:
                if "monotonic_ts" not in event:
                    event["monotonic_ts"] = _record_ts(event)
                normalized.append(event)
        return normalized
    return _normalize_latency_events(events)


def _numbers(events: list[dict[str, Any]], key: str, event_type: str | None = None) -> list[float]:
    values = []
    for event in events:
        if event_type is not None and event.get("event_type") != event_type:
            continue
        value = event.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            values.append(float(value))
    return values


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    idx = min(len(values) - 1, max(0, int(round((len(values) - 1) * q))))
    return values[idx]


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    obs_ids = {
        event.get("obs_id") for event in events if event.get("event_type") == "observation_capture_end"
    }
    used_obs = {
        event.get("source_obs_id") for event in events if event.get("event_type") == "inference_started"
    }
    inference_ids = {
        event.get("inference_id") for event in events if event.get("event_type") == "inference_finished"
    }
    discarded_ids = {
        event.get("inference_id") for event in events if event.get("event_type") == "inference_discarded"
    }
    inference_latencies = _numbers(events, "model_latency", "inference_finished")
    obs_age_at_inference = []
    capture_by_obs = {
        event.get("obs_id"): event.get("obs_capture_end_ts")
        for event in events
        if event.get("event_type") == "observation_capture_end"
    }
    for event in events:
        if event.get("event_type") != "inference_started":
            continue
        capture_ts = capture_by_obs.get(event.get("source_obs_id"))
        start_ts = event.get("inference_start_ts")
        if isinstance(capture_ts, int | float) and isinstance(start_ts, int | float):
            obs_age_at_inference.append(float(start_ts - capture_ts))

    queue_lengths = _numbers(events, "new_queue_length", "queue_update_finished")
    return {
        "total_observations": len(obs_ids),
        "observations_used_for_inference": len(used_obs),
        "observations_dropped": sum(1 for e in events if e.get("event_type") == "observation_dropped"),
        "total_inferences": len(inference_ids),
        "discarded_inferences": len(discarded_ids),
        "mean_inference_latency": mean(inference_latencies) if inference_latencies else None,
        "p95_inference_latency": _percentile(inference_latencies, 0.95),
        "mean_obs_age_at_inference_start": mean(obs_age_at_inference) if obs_age_at_inference else None,
        "mean_obs_age_at_queue_update": (
            mean(_numbers(events, "source_obs_age", "queue_update_finished"))
            if _numbers(events, "source_obs_age", "queue_update_finished")
            else None
        ),
        "total_actions_executed": sum(
            1 for e in events if e.get("event_type") == "action_execution_finished"
        ),
        "total_actions_replaced": sum(
            int(e.get("num_actions_replaced") or 0)
            for e in events
            if e.get("event_type") == "queue_update_finished"
        ),
        "mean_queue_length": mean(queue_lengths) if queue_lengths else None,
        "mean_action_jump_before_update": (
            mean(_numbers(events, "mean_action_jump_before_update", "queue_update_finished"))
            if _numbers(events, "mean_action_jump_before_update", "queue_update_finished")
            else None
        ),
        "mean_action_jump_after_update": (
            mean(_numbers(events, "mean_action_jump_after_update", "queue_update_finished"))
            if _numbers(events, "mean_action_jump_after_update", "queue_update_finished")
            else None
        ),
    }


def plot_timeline(
    log_path: str | Path | Sequence[str | Path],
    output_path: str | Path,
    window_start: float = 0.0,
    window_duration: float = 8.0,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    events = _load_events(log_path)
    if not events:
        raise ValueError(f"No events found in {log_path}")

    first_ts = min(float(event["monotonic_ts"]) for event in events if "monotonic_ts" in event)
    start_abs = first_ts + window_start
    end_abs = start_abs + window_duration

    def rel(ts: float | int | None) -> float | None:
        if not isinstance(ts, int | float):
            return None
        return float(ts) - first_ts

    obs_events = [e for e in events if e.get("event_type") == "observation_capture_end"]
    inf_start_by_id = {
        e.get("inference_id"): e for e in events if e.get("event_type") == "inference_started"
    }
    inf_finish = [e for e in events if e.get("event_type") == "inference_finished"]
    discarded = {e.get("inference_id") for e in events if e.get("event_type") == "inference_discarded"}
    used_obs = {e.get("source_obs_id") for e in events if e.get("event_type") == "inference_started"}
    queue_updates = [e for e in events if e.get("event_type") == "queue_update_finished"]
    action_finish = [e for e in events if e.get("event_type") == "action_execution_finished"]

    fig, ax = plt.subplots(figsize=(18, 7))
    lanes = {
        "observation capture/send": 2.2,
        "server inference": 1.2,
        "client action execution": 0.2,
    }

    for y in lanes.values():
        ax.hlines(y, window_start, window_start + window_duration, color="#555555", linewidth=1.5)

    for event in obs_events:
        ts = rel(event.get("obs_capture_end_ts") or event.get("monotonic_ts"))
        if ts is None or not (window_start <= ts <= window_start + window_duration):
            continue
        obs_id = event.get("obs_id")
        color = "#f28e2b" if obs_id in used_obs else "#f28e2b55"
        linewidth = 2.5 if obs_id in used_obs else 0.8
        ax.vlines(
            ts,
            lanes["observation capture/send"] - 0.12,
            lanes["observation capture/send"] + 0.12,
            color=color,
            linewidth=linewidth,
        )
        if obs_id in used_obs:
            ax.text(
                ts,
                lanes["observation capture/send"] + 0.18,
                f"obs {obs_id}\n{ts:.2f}s",
                rotation=35,
                fontsize=8,
                color="#8a4f1d",
            )

    by_source = defaultdict(list)
    for event in action_finish:
        by_source[(event.get("source_inference_id"), event.get("source_obs_id"))].append(event)

    for event in inf_finish:
        inference_id = event.get("inference_id")
        start_event = inf_start_by_id.get(inference_id, {})
        start = rel(start_event.get("inference_start_ts") or event.get("inference_start_ts"))
        end = rel(event.get("inference_end_ts"))
        if start is None or end is None or end < window_start or start > window_start + window_duration:
            continue
        y = lanes["server inference"]
        color = "#4e79a7"
        linestyle = "--" if inference_id in discarded else "-"
        ax.plot(
            [start, end],
            [y, y],
            color=color,
            linewidth=8,
            alpha=0.85,
            linestyle=linestyle,
            solid_capstyle="butt",
        )
        latency = event.get("model_latency")
        obs_id = event.get("source_obs_id")
        latency_txt = f"{float(latency):.2f}s" if isinstance(latency, int | float) else ""
        ax.text(
            start + (end - start) / 2,
            y + 0.16,
            f"inf {inference_id}\nobs {obs_id}\n{latency_txt}",
            ha="center",
            fontsize=8,
            color="#2f5c89",
        )

    action_groups = defaultdict(list)
    for event in action_finish:
        start = rel(event.get("actual_exec_start_ts"))
        end = rel(event.get("actual_exec_end_ts"))
        if start is None or end is None:
            continue
        key = (event.get("source_inference_id"), event.get("source_obs_id"))
        action_groups[key].append((start, end, event))
    for (inf_id, obs_id), items in action_groups.items():
        items = sorted(items)
        if not items:
            continue
        start = items[0][0]
        end = items[-1][1]
        if end < window_start or start > window_start + window_duration:
            continue
        y = lanes["client action execution"]
        ax.plot(
            [start, end],
            [y, y],
            color="#59a14f",
            linewidth=8,
            alpha=0.85,
            solid_capstyle="butt",
        )
        first_idx = items[0][2].get("chunk_index")
        last_idx = items[-1][2].get("chunk_index")
        ax.text(
            start + (end - start) / 2,
            y - 0.24,
            f"inf {inf_id}\nobs {obs_id}\na{first_idx}-{last_idx}",
            ha="center",
            fontsize=8,
            color="#2f7532",
        )

    for event in queue_updates:
        ts = rel(event.get("monotonic_ts"))
        if ts is None or not (window_start <= ts <= window_start + window_duration):
            continue
        color = "#9c755f" if not event.get("discard_reason") else "#e15759"
        ax.vlines(
            ts,
            lanes["client action execution"] - 0.35,
            lanes["server inference"] - 0.2,
            color=color,
            linewidth=1.2,
            linestyle=":",
        )
        label = event.get("aggregation_fn", "update")
        if event.get("num_actions_blended"):
            label = f"{label}\nblend {event.get('num_actions_blended')}"
        ax.text(
            ts + 0.015,
            lanes["client action execution"] + 0.32,
            label,
            fontsize=7,
            color=color,
            rotation=0,
        )

    duration = max(1e-6, window_duration)
    obs_in_window = [
        e
        for e in obs_events
        if isinstance(rel(e.get("obs_capture_end_ts") or e.get("monotonic_ts")), float)
        and window_start
        <= rel(e.get("obs_capture_end_ts") or e.get("monotonic_ts"))
        <= window_start + window_duration
    ]
    dropped = sum(1 for e in events if e.get("event_type") == "observation_dropped")
    ax.text(
        window_start + 0.02,
        lanes["observation capture/send"] - 0.32,
        f"{len(obs_in_window) / duration:.1f} obs FPS in window; dropped events: {dropped}",
        fontsize=9,
        color="#777777",
    )

    ax.set_xlim(window_start, window_start + window_duration)
    ax.set_ylim(-0.55, 2.85)
    ax.set_yticks(list(lanes.values()), labels=list(lanes.keys()))
    ax.set_xlabel("seconds since first event in log")
    ax.set_title("Continuous async inference timeline")
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(
        handles=[
            Line2D([0], [0], color="#f28e2b", linewidth=3, label="observation"),
            Patch(color="#4e79a7", label="inference"),
            Patch(color="#59a14f", label="action execution"),
            Line2D([0], [0], color="#9c755f", linestyle=":", label="queue update"),
            Line2D([0], [0], color="#4e79a7", linestyle="--", label="discarded inference"),
            Line2D([0], [0], color="#e15759", linestyle=":", label="replaced/discarded action"),
        ],
        loc="upper right",
    )
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    pdf_path = output_path.with_suffix(".pdf")
    fig.savefig(pdf_path)
    plt.close(fig)

    summary = summarize_events(events)
    if summary_path is None:
        summary_path = output_path.with_name(output_path.stem + "_summary.json")
    summary_path = Path(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_path",
        required=True,
        nargs="+",
        help="One or more JSONL logs. Supports continuous event logs and classic *_latency_*.jsonl logs.",
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--window_start", type=float, default=0.0)
    parser.add_argument("--window_duration", type=float, default=8.0)
    parser.add_argument("--summary_path", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = plot_timeline(
        log_path=args.log_path,
        output_path=args.output_path,
        window_start=args.window_start,
        window_duration=args.window_duration,
        summary_path=args.summary_path,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
