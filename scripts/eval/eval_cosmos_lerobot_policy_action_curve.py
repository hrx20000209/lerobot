#!/usr/bin/env python

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def _to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    image = torch.as_tensor(tensor).detach().cpu()
    if image.ndim == 4:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected image tensor, got shape={tuple(image.shape)}")
    if image.shape[0] in (1, 3, 4):
        image = image[:3].permute(1, 2, 0)
    if image.dtype.is_floating_point:
        if float(image.max()) <= 1.5:
            image = image * 255.0
        image = image.round().clamp(0, 255)
    return image.to(torch.uint8).numpy()


def _save_plot(path: Path, predicted: np.ndarray, target: np.ndarray, names: list[str]) -> None:
    error = np.abs(predicted - target)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    steps = np.arange(predicted.shape[0])
    for dim, axis in enumerate(axes.flat):
        axis.plot(steps, target[:, dim], label="ground truth", linewidth=2)
        axis.plot(steps, predicted[:, dim], label="predicted", linewidth=1.6)
        axis.plot(steps, error[:, dim], label="absolute error", linewidth=1.0, alpha=0.75)
        axis.set_title(f"{dim}: {names[dim]} | MAE={error[:, dim].mean():.3f}")
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel("action step")
    axes[-1, 1].set_xlabel("action step")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_path", type=Path, required=True)
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--repo_id", default="hrx2000/Three_Cubes_1")
    parser.add_argument("--episode_id", type=int, default=0)
    parser.add_argument("--start_frame", type=int, default=0)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    repo_src = Path(__file__).resolve().parents[2] / "src"
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class

    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    policy_cls = get_policy_class(cfg.type)
    policy = policy_cls.from_pretrained(args.policy_path, config=cfg)
    policy.eval()

    fps = int(json.loads((args.dataset_root / "meta" / "info.json").read_text())["fps"])
    delta_timestamps = {
        cfg.primary_camera_key: [0.0],
        cfg.wrist_camera_key: [0.0],
        cfg.state_key: [0.0],
        cfg.action_key: [step / fps for step in range(cfg.chunk_size)],
    }
    if cfg.wrist_left_camera_key:
        delta_timestamps[cfg.wrist_left_camera_key] = [0.0]
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.dataset_root,
        episodes=[args.episode_id],
        delta_timestamps=delta_timestamps,
        return_uint8=True,
    )
    index = int(args.start_frame)
    sample = dataset[index]
    batch = {}
    for key in (cfg.primary_camera_key, cfg.wrist_camera_key, cfg.wrist_left_camera_key, cfg.state_key):
        if key in sample:
            value = sample[key]
            if isinstance(value, torch.Tensor):
                batch[key] = value.unsqueeze(0)
    task = sample.get("task", cfg.default_task)
    batch["task"] = [str(task)]

    with torch.no_grad():
        predicted = policy.predict_action_chunk(batch)[0].detach().cpu().numpy().astype(np.float32)
    target = torch.as_tensor(sample[cfg.action_key], dtype=torch.float32).numpy()[: cfg.chunk_size, : cfg.action_dim]
    predicted = predicted[: target.shape[0], : cfg.action_dim]

    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    error = np.abs(predicted - target)
    metrics = {
        "policy_path": str(args.policy_path),
        "dataset_root": str(args.dataset_root),
        "episode_id": args.episode_id,
        "start_frame": args.start_frame,
        "overall_mae": float(error.mean()),
        "per_dim_mae": {name: float(error[:, idx].mean()) for idx, name in enumerate(names)},
        "first_8_step_mae": float(error[:8].mean()),
        "first_16_step_mae": float(error[:16].mean()),
        "pred_action_min": float(predicted.min()),
        "pred_action_max": float(predicted.max()),
        "gt_action_min": float(target.min()),
        "gt_action_max": float(target.max()),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _save_plot(args.output_dir / "predicted_vs_gt_action.png", predicted, target, names)
    with (args.output_dir / "predicted_vs_gt_action.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["step", *[f"pred_{name}" for name in names], *[f"gt_{name}" for name in names]])
        for step in range(predicted.shape[0]):
            writer.writerow([step, *predicted[step].tolist(), *target[step].tolist()])
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    Image.fromarray(_to_uint8_image(sample[cfg.primary_camera_key])).save(args.output_dir / "front_input.png")
    Image.fromarray(_to_uint8_image(sample[cfg.wrist_camera_key])).save(args.output_dir / "wrist_input.png")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
