#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METRIC_LINE = re.compile(r"policy_metrics step:(\d+)\s+(.*)")
METRIC_VALUE = re.compile(r"([A-Za-z0-9_./-]+):([-+0-9.eE]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor GigaWorld loss and evaluate saved checkpoints.")
    parser.add_argument("--training-log", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--training-pid", type=int, default=None)
    parser.add_argument("--eval-gpu", default="7")
    parser.add_argument("--interval-s", type=float, default=60.0)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--visual-loss-weight", type=float, default=0.1)
    parser.add_argument("--python", default="/home/rxhuang/anaconda3/envs/lerobot/bin/python")
    parser.add_argument("--dataset-root", type=Path, default=Path("/data/rxhuang/three_cubes_1"))
    parser.add_argument("--repo-id", default="hrx2000/Three_Cubes_1")
    parser.add_argument("--revision", default="v0.1.0")
    return parser.parse_args()


def read_metrics(log_path: Path) -> list[dict[str, float]]:
    if not log_path.exists():
        return []
    records = []
    for line in log_path.read_text(errors="replace").splitlines():
        match = METRIC_LINE.search(line)
        if match is None:
            continue
        record = {"step": int(match.group(1))}
        record.update({key: float(value) for key, value in METRIC_VALUE.findall(match.group(2))})
        records.append(record)
    return records


def moving_average(values: np.ndarray, window: int = 10) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    return np.concatenate([np.full(window - 1, np.nan), smoothed])


def plot_losses(records: list[dict[str, float]], output_dir: Path, visual_weight: float) -> None:
    if not records:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    steps = np.asarray([record["step"] for record in records])
    action = np.asarray([record.get("loss_action", np.nan) for record in records])
    visual = np.asarray([record.get("loss_visual", np.nan) for record in records])
    total = action + visual_weight * visual

    figure, axis = plt.subplots(figsize=(11, 6))
    axis.plot(steps, total, alpha=0.25, color="#111827", label="weighted total")
    axis.plot(steps, moving_average(total), color="#111827", linewidth=2.0, label="weighted total (MA10)")
    axis.plot(steps, moving_average(action), color="#2563eb", label="action loss (MA10)")
    axis.plot(steps, moving_average(visual), color="#f97316", label="visual loss (MA10)")
    axis.set_xlabel("training step")
    axis.set_ylabel("loss")
    axis.set_title("GigaWorld fine-tuning loss")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "loss_curve.png", dpi=180)
    plt.close(figure)
    (output_dir / "loss_metrics.json").write_text(json.dumps(records, indent=2))


def process_alive(pid: int | None) -> bool:
    if pid is None:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def checkpoint_dirs(run_dir: Path) -> list[Path]:
    root = run_dir / "checkpoints"
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir() and path.name.isdigit())


def evaluate_checkpoint(args: argparse.Namespace, checkpoint_dir: Path, monitor_dir: Path) -> None:
    checkpoint = checkpoint_dir / "pretrained_model"
    adapter = checkpoint / "transformer" / "adapter_model.safetensors"
    if not adapter.exists():
        return
    output_dir = monitor_dir / "action_curves" / f"{checkpoint_dir.name}_episode_{args.episode:03d}"
    marker = output_dir / f"episode_{args.episode:03d}_metrics.json"
    if marker.exists():
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        args.python,
        "scripts/analyze_pi05_dataset_actions.py",
        "--checkpoint",
        str(checkpoint),
        "--dataset-root",
        str(args.dataset_root),
        "--repo-id",
        args.repo_id,
        "--revision",
        args.revision,
        "--episode",
        str(args.episode),
        "--device",
        "cuda",
        "--execution-horizon",
        "16",
        "--policy-label",
        f"giga_world_{checkpoint_dir.name}",
        "--output-dir",
        str(output_dir),
    ]
    if args.max_frames is not None:
        command.extend(["--max-frames", str(args.max_frames)])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.eval_gpu
    env.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    with open(output_dir / "evaluation.log", "w") as log_file:
        subprocess.run(command, cwd=Path(__file__).resolve().parents[1], env=env, stdout=log_file, stderr=subprocess.STDOUT)


def main() -> None:
    args = parse_args()
    monitor_dir = args.run_dir / "monitor"
    while not args.run_dir.exists():
        if not process_alive(args.training_pid):
            return
        time.sleep(min(args.interval_s, 5.0))
    monitor_dir.mkdir(parents=True, exist_ok=True)
    while True:
        plot_losses(read_metrics(args.training_log), monitor_dir, args.visual_loss_weight)
        for checkpoint_dir in checkpoint_dirs(args.run_dir):
            evaluate_checkpoint(args, checkpoint_dir, monitor_dir)
        if not process_alive(args.training_pid):
            plot_losses(read_metrics(args.training_log), monitor_dir, args.visual_loss_weight)
            break
        time.sleep(args.interval_s)


if __name__ == "__main__":
    main()
