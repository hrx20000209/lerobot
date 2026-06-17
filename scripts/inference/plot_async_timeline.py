#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SERVER_KINDS = {"timeline_server_inference", "timeline_vla_inference"}


def _jsonl_paths(paths: list[Path]) -> list[Path]:
    if not paths:
        paths = [Path("logs/async_timeline")]

    out: list[Path] = []
    for path in paths:
        if path.is_dir():
            out.extend(sorted(path.glob("*_latency_*.jsonl")))
        elif path.is_file():
            out.append(path)
    return out


def _read_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in _jsonl_paths(paths):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                record["_path"] = str(path)
                records.append(record)
    return records


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _event_start(record: dict[str, Any]) -> float | None:
    for key in ("start_time", "observation_time", "receive_time", "time"):
        value = _number(record.get(key))
        if value is not None:
            return value
    return None


def _event_end(record: dict[str, Any]) -> float | None:
    for key in ("end_time", "received_time", "receive_time", "time"):
        value = _number(record.get(key))
        if value is not None:
            return value
    return _event_start(record)


def _dedupe(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for record in records:
        key = (
            record.get("kind"),
            record.get("timestep"),
            record.get("first_timestep"),
            record.get("last_timestep"),
            round(_event_start(record) or 0.0, 4),
            round(_event_end(record) or 0.0, 4),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(record)
    return out


def _collect(records: list[dict[str, Any]], window_s: float | None) -> tuple[list, list, list, float]:
    server = [r for r in records if r.get("kind") in SERVER_KINDS]
    observations = [r for r in records if r.get("kind") == "timeline_observation"]
    actions = [r for r in records if r.get("kind") == "timeline_action"]

    server = _dedupe(server)
    observations = _dedupe(observations)
    actions = _dedupe(actions)

    times = []
    for record in [*server, *observations, *actions]:
        start = _event_start(record)
        end = _event_end(record)
        if start is not None:
            times.append(start)
        if end is not None:
            times.append(end)
    if not times:
        raise SystemExit("No timeline events found. Run client/server with --record_timeline=true first.")

    base = min(times)
    if window_s is not None:
        cutoff = base + window_s
        server = [r for r in server if (_event_start(r) or 0) <= cutoff]
        observations = [r for r in observations if (_event_start(r) or 0) <= cutoff]
        actions = [r for r in actions if (_event_start(r) or 0) <= cutoff]

    return server, observations, actions, base


def _rel(value: Any, base: float) -> float:
    number = _number(value)
    if number is None:
        return 0.0
    return number - base


def _duration(record: dict[str, Any], base: float, min_width: float) -> tuple[float, float]:
    start_abs = _event_start(record)
    end_abs = _event_end(record)
    if start_abs is None:
        start_abs = end_abs or base
    if end_abs is None:
        end_abs = start_abs
    start = start_abs - base
    width = max(end_abs - start_abs, min_width)
    return start, width


def _maybe_int(value: Any) -> str:
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if value is None:
        return "?"
    return str(value)


def plot_timeline(
    records: list[dict[str, Any]],
    output: Path,
    window_s: float | None,
    max_actions: int,
    title: str,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    server, observations, actions, base = _collect(records, window_s)
    if max_actions > 0 and len(actions) > max_actions:
        actions = actions[:max_actions]

    fig, ax = plt.subplots(figsize=(16, 7))
    rows = {
        "server": (30, 5),
        "camera": (20, 5),
        "action": (10, 5),
    }
    colors = {
        "server": "#4C78A8",
        "camera": "#F58518",
        "action": "#54A24B",
    }
    min_width = 0.004

    for idx, record in enumerate(sorted(server, key=lambda r: _event_start(r) or 0.0)):
        start, width = _duration(record, base, min_width)
        y, h = rows["server"]
        ax.broken_barh([(start, width)], (y, h), facecolors=colors["server"], alpha=0.85)
        if idx < 40:
            queue_size = record.get("queue_size_at_observation", record.get("queue_size"))
            label = (
                f"obs {record.get('timestep', '?')}\n"
                f"a{_maybe_int(record.get('first_timestep'))}-{_maybe_int(record.get('last_timestep'))}"
            )
            if queue_size is not None:
                label += f"\nq={queue_size}"
            ax.text(start, y + h + 0.25, label, fontsize=7, rotation=35, va="bottom")

    for _idx, record in enumerate(sorted(observations, key=lambda r: _event_start(r) or 0.0)):
        start, width = _duration(record, base, min_width)
        y, h = rows["camera"]
        must_go = bool(record.get("must_go"))
        alpha = 0.95 if must_go else 0.35
        ax.broken_barh([(start, width)], (y, h), facecolors=colors["camera"], alpha=alpha)
        if must_go or len(observations) <= 30:
            label = f"obs {record.get('timestep', '?')}\nq={record.get('queue_size', '?')}"
            ax.text(start, y + h + 0.25, label, fontsize=7, rotation=35, va="bottom")

    for idx, record in enumerate(sorted(actions, key=lambda r: _event_start(r) or 0.0)):
        start, width = _duration(record, base, min_width)
        y, h = rows["action"]
        ax.broken_barh([(start, width)], (y, h), facecolors=colors["action"], alpha=0.75)
        if len(actions) <= 80 or idx % 10 == 0:
            source = record.get("source_observation_timestep")
            label = f"a{record.get('timestep', '?')}"
            if source is not None:
                label += f"\nobs {source}"
            ax.text(start, y - 0.5, label, fontsize=6, rotation=55, va="top")

    ax.set_title(title)
    ax.set_xlabel("time since first recorded event (s)")
    ax.set_yticks([rows["server"][0] + 2.5, rows["camera"][0] + 2.5, rows["action"][0] + 2.5])
    ax.set_yticklabels(["server inference", "camera observation", "client action"])
    ax.grid(True, axis="x", alpha=0.25)
    ax.set_ylim(5, 40)
    ax.legend(
        handles=[
            Patch(facecolor=colors["server"], label="server inference"),
            Patch(facecolor=colors["camera"], label="camera read/send"),
            Patch(facecolor=colors["action"], label="action execution"),
        ],
        loc="upper right",
    )
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LeRobot async inference timeline JSONL logs.")
    parser.add_argument("paths", nargs="*", type=Path, help="Timeline JSONL files or directories")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/async_timeline/async_timeline.png"),
        help="Output image path",
    )
    parser.add_argument("--window-s", type=float, default=None, help="Only plot the first N seconds")
    parser.add_argument("--max-actions", type=int, default=300, help="Maximum action bars to draw")
    parser.add_argument("--title", default="LeRobot async inference timeline")
    args = parser.parse_args()

    records = _read_records(args.paths)
    output = plot_timeline(records, args.output, args.window_s, args.max_actions, args.title)
    print(output)


if __name__ == "__main__":
    main()
