#!/usr/bin/env python

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from peft import PeftConfig, PeftModel

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import get_policy_class, make_pre_post_processors


ACTION_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--second-camera-key", default="observation.images.right")
    parser.add_argument("--second-camera-policy-key", default="observation.images.wrist")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--npz-out", type=Path, default=None)
    return parser.parse_args()


def load_policy(model_path: Path, device: str):
    cfg = PreTrainedConfig.from_pretrained(model_path)
    cfg.pretrained_path = model_path
    cfg.device = device

    policy_cls = get_policy_class(cfg.type)
    if cfg.use_peft:
        peft_config = PeftConfig.from_pretrained(str(model_path))
        base_path = peft_config.base_model_name_or_path
        policy = policy_cls.from_pretrained(pretrained_name_or_path=base_path, config=cfg)
        policy = PeftModel.from_pretrained(policy, str(model_path), config=peft_config)
    else:
        policy = policy_cls.from_pretrained(pretrained_name_or_path=str(model_path), config=cfg)

    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(cfg, pretrained_path=str(model_path))
    return cfg, policy, preprocessor, postprocessor


def load_episode_indices(dataset_root: Path, episode_index: int, max_frames: int | None) -> list[int]:
    parquet_files = sorted((dataset_root / "data").glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_root / 'data'}")

    parts = []
    for path in parquet_files:
        df = pd.read_parquet(path, columns=["episode_index", "index"])
        parts.append(df[df["episode_index"] == episode_index])
    episode_df = pd.concat(parts, ignore_index=True).sort_values("index")
    if episode_df.empty:
        raise ValueError(f"Episode {episode_index} not found in {dataset_root}")
    indices = episode_df["index"].astype(int).tolist()
    if max_frames is not None:
        indices = indices[:max_frames]
    return indices


def postprocess_chunk(postprocessor, action_chunk: torch.Tensor) -> torch.Tensor:
    if action_chunk.ndim != 3:
        return postprocessor(action_chunk).squeeze(0)

    processed = []
    for i in range(action_chunk.shape[1]):
        processed.append(postprocessor(action_chunk[:, i, :]))
    return torch.stack(processed, dim=1).squeeze(0)


def predict_episode(
    ds: LeRobotDataset,
    indices: list[int],
    policy,
    preprocessor,
    postprocessor,
    device: str,
    horizon: int,
    second_camera_key: str,
    second_camera_policy_key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
    n = len(indices)
    pred = np.full((n, len(ACTION_NAMES)), np.nan, dtype=np.float32)
    gt = np.zeros_like(pred)
    state = np.zeros_like(pred)
    timestamps = np.zeros((n,), dtype=np.float32)
    boundaries: list[int] = []

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    for local_i, ds_idx in enumerate(indices):
        item = ds[ds_idx]
        gt[local_i] = item["action"].detach().cpu().numpy()
        state[local_i] = item["observation.state"].detach().cpu().numpy()
        timestamps[local_i] = float(item["timestamp"])

        if local_i % horizon != 0:
            continue

        obs = {
            "observation.images.front": item["observation.images.front"],
            second_camera_policy_key: item[second_camera_key],
            "observation.state": item["observation.state"],
            "task": [item["task"]],
        }
        with torch.inference_mode():
            batch = preprocessor(obs)
            action_chunk = policy.predict_action_chunk(batch)
            action_chunk = postprocess_chunk(postprocessor, action_chunk).detach().cpu().numpy()

        end = min(local_i + len(action_chunk), n)
        pred[local_i:end] = action_chunk[: end - local_i]
        boundaries.append(local_i)
        logging.info("episode frame %d/%d predicted chunk=%s", local_i, n, action_chunk.shape)

    missing = np.isnan(pred[:, 0])
    if missing.any():
        valid = np.where(~missing)[0]
        if len(valid) == 0:
            raise RuntimeError("No predicted actions were produced")
        pred[missing] = pred[valid[-1]]

    return timestamps, pred, gt, state, boundaries


def plot_curves(
    out: Path,
    timestamps: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray,
    state: np.ndarray,
    boundaries: list[int],
    title: str,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(24, 8), sharex=True)
    groups = [(0, 3), (3, 6)]
    colors = ["tab:orange", "tab:blue", "tab:green"]

    for ax, (start, end) in zip(axes, groups, strict=True):
        for color, dim in zip(colors, range(start, end), strict=True):
            name = ACTION_NAMES[dim]
            ax.plot(timestamps, pred[:, dim], color=color, linewidth=2.0, label=f"{name} pred")
            ax.plot(timestamps, gt[:, dim], color=color, linestyle="--", linewidth=1.6, label=f"{name} GT action")
            ax.plot(
                timestamps,
                state[:, dim],
                color=color,
                linestyle=":",
                linewidth=1.4,
                alpha=0.75,
                label=f"{name} observation.state",
            )
        for idx in boundaries:
            ax.axvline(timestamps[idx], color="red", alpha=0.16, linewidth=1.0)
        ax.set_title(", ".join(ACTION_NAMES[start:end]))
        ax.set_xlabel("time (s)")
        ax.set_ylabel("joint position")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=9)

    fig.suptitle(title + "\nred lines = replanning boundaries", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(asctime)s %(message)s")
    args = parse_args()

    cfg, policy, preprocessor, postprocessor = load_policy(args.model_path, args.device)
    horizon = args.horizon or int(getattr(cfg, "chunk_size", 7))

    ds = LeRobotDataset("local/offline_eval", root=args.dataset_root, video_backend="pyav")
    indices = load_episode_indices(args.dataset_root, args.episode_index, args.max_frames)
    logging.info(
        "Loaded dataset=%s episode=%d frames=%d horizon=%d task='%s'",
        args.dataset_root,
        args.episode_index,
        len(indices),
        horizon,
        ds[indices[0]]["task"],
    )
    if args.second_camera_key not in ds.features:
        raise KeyError(f"{args.second_camera_key} not found in dataset features: {list(ds.features)}")

    timestamps, pred, gt, state, boundaries = predict_episode(
        ds=ds,
        indices=indices,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=args.device,
        horizon=horizon,
        second_camera_key=args.second_camera_key,
        second_camera_policy_key=args.second_camera_policy_key,
    )

    title = (
        f"VLA-JEPA h={horizon} episode {args.episode_index} dataset replay: "
        "prediction vs GT action vs observation.state"
    )
    plot_curves(args.out, timestamps, pred, gt, state, boundaries, title)

    npz_out = args.npz_out or args.out.with_suffix(".npz")
    np.savez(
        npz_out,
        timestamps=timestamps,
        pred=pred,
        gt=gt,
        state=state,
        boundaries=np.asarray(boundaries),
        action_names=np.asarray(ACTION_NAMES),
    )
    logging.info("Saved plot to %s", args.out)
    logging.info("Saved arrays to %s", npz_out)


if __name__ == "__main__":
    main()
