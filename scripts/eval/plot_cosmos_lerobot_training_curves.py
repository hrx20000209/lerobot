#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No metrics found in {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max_step", type=int, default=None)
    args = parser.parse_args()

    rows = _read_jsonl(args.metrics_path)
    if args.max_step is not None:
        rows = [row for row in rows if int(row.get("step", row.get("steps", 0))) <= args.max_step]
    steps = np.asarray([int(row.get("step", row.get("steps"))) for row in rows])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(10, 6))
    for key, label in (("loss", "total loss"), ("action_loss", "action loss"), ("grad_norm", "grad norm")):
        values = np.asarray([float(row[key]) for row in rows if key in row and row[key] is not None])
        key_steps = np.asarray([int(row.get("step", row.get("steps"))) for row in rows if key in row and row[key] is not None])
        if len(values):
            axis.plot(key_steps, values, label=label, linewidth=1.8)
    axis.set_xlabel("step")
    axis.set_ylabel("value")
    axis.grid(alpha=0.25)
    axis.legend()
    axis.set_title(f"Cosmos LeRobot training curves through step {int(steps.max())}")
    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
