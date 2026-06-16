#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PI0.5 checkpoints as they are saved.")
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--gpu", default="7")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def checkpoint_ready(checkpoint: Path) -> bool:
    model_dir = checkpoint / "pretrained_model"
    state_file = checkpoint / "training_state" / "training_step.json"
    weights_ready = (model_dir / "adapter_model.safetensors").exists() or (
        model_dir / "model.safetensors"
    ).exists()
    return weights_ready and state_file.exists() and time.time() - state_file.stat().st_mtime > 10


def checkpoint_steps(run_dir: Path) -> list[Path]:
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        return []
    return sorted(
        path for path in checkpoints_dir.iterdir() if path.is_dir() and path.name.isdigit()
    )


def evaluate(run_dir: Path, checkpoint: Path, eval_root: Path, gpu: str, episode: int) -> None:
    output_dir = eval_root / run_dir.name / checkpoint.name
    metrics_path = output_dir / f"episode_{episode:03d}_metrics.json"
    if metrics_path.exists():
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "evaluation.log"
    command = [
        "/home/rxhuang/anaconda3/envs/lerobot/bin/python",
        "scripts/analyze_pi05_dataset_actions.py",
        "--checkpoint",
        str(checkpoint / "pretrained_model"),
        "--dataset-root",
        "/data/rxhuang/three_cubes_1",
        "--repo-id",
        "hrx2000/Three_Cubes_1",
        "--revision",
        "v0.1.0",
        "--episode",
        str(episode),
        "--device",
        "cuda",
        "--output-dir",
        str(output_dir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("HF_HOME", "/data/hf_cache")
    with log_path.open("w") as log_file:
        subprocess.run(command, check=True, cwd=Path(__file__).resolve().parents[1], env=env, stdout=log_file, stderr=subprocess.STDOUT)


def write_ranking(eval_root: Path, episode: int) -> None:
    rows = []
    for metrics_path in eval_root.glob(f"*/*/episode_{episode:03d}_metrics.json"):
        metrics = json.loads(metrics_path.read_text())
        mae = metrics["mae"]
        boundary = metrics["chunk_boundary_abs_jump_mean"]
        rows.append(
            {
                "run": metrics_path.parents[1].name,
                "checkpoint": metrics_path.parent.name,
                "mean_mae": sum(mae) / len(mae),
                "mean_boundary_jump": sum(boundary) / len(boundary),
                "joint_mae": json.dumps(mae),
                "metrics": str(metrics_path),
            }
        )
    rows.sort(key=lambda row: (row["mean_mae"], row["mean_boundary_jump"]))
    ranking_path = eval_root / "ranking.csv"
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    with ranking_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else ["run", "checkpoint"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.eval_root.mkdir(parents=True, exist_ok=True)
    while True:
        for run_dir in args.run_dir:
            for checkpoint in checkpoint_steps(run_dir):
                if checkpoint_ready(checkpoint):
                    try:
                        evaluate(run_dir, checkpoint, args.eval_root, args.gpu, args.episode)
                    except subprocess.CalledProcessError as error:
                        print(f"Evaluation failed for {checkpoint}: {error}", flush=True)
                    write_ranking(args.eval_root, args.episode)
        if args.once:
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
