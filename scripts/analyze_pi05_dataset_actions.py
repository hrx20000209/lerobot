#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.configs import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE

DEFAULT_JOINT_NAMES = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a LeRobot episode through a chunked policy checkpoint and compare predicted actions."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--repo-id", default="hrx2000/Three_Cubes_1")
    parser.add_argument("--revision", default="v0.1.0")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--policy-label", default=None)
    parser.add_argument(
        "--execution-horizon",
        type=int,
        default=None,
        help="Number of predicted actions executed before replanning. Defaults to policy.n_action_steps.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--num-inference-timesteps", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("output_lerobot_eval/pi05_dataset"))
    return parser.parse_args()


def load_episode(args: argparse.Namespace) -> LeRobotDataset:
    return LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        episodes=[args.episode],
        revision=args.revision,
        video_backend="torchcodec",
        return_uint8=False,
    )


def raw_episode_arrays(dataset: LeRobotDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = dataset.hf_dataset
    actions = np.asarray(raw[ACTION], dtype=np.float32)
    states = np.asarray(raw[OBS_STATE], dtype=np.float32)
    timestamps = np.asarray(raw["timestamp"], dtype=np.float32)
    return actions, states, timestamps


def make_noise(policy, seed: int) -> torch.Tensor:
    generator = torch.Generator(device=policy.config.device).manual_seed(seed)
    return torch.randn(
        1,
        policy.config.chunk_size,
        policy.config.max_action_dim,
        generator=generator,
        device=policy.config.device,
        dtype=torch.float32,
    )


def read_saved_rename_map(checkpoint: Path) -> dict[str, str]:
    preprocessor_path = checkpoint / "policy_preprocessor.json"
    if not preprocessor_path.exists():
        return {}

    data = json.loads(preprocessor_path.read_text())
    for step in data.get("steps", []):
        if step.get("registry_name") == "rename_observations_processor":
            return dict(step.get("config", {}).get("rename_map", {}))
    return {}


def make_observation(sample: dict) -> dict:
    observation = {
        key: value
        for key, value in sample.items()
        if key.startswith("observation.") or key in {"task", "task_index"}
    }
    if "task" in sample:
        observation["task"] = sample["task"]
    return observation


def make_prediction_kwargs(
    policy,
    seed: int,
    anchor_idx: int,
    num_inference_steps: int | None,
) -> dict:
    kwargs = {}
    # PI0/PI05 expose stochastic flow sampling through an optional noise tensor and `num_steps`.
    # Other policies, such as VLA-JEPA, own their sampling loop and should be called without these extras.
    if hasattr(policy.config, "max_action_dim"):
        kwargs["noise"] = make_noise(policy, seed + anchor_idx)
        if num_inference_steps is not None:
            kwargs["num_steps"] = num_inference_steps
    return kwargs


def predict_chunked_episode(
    dataset: LeRobotDataset,
    policy,
    preprocessor,
    postprocessor,
    execution_horizon: int,
    seed: int,
    num_inference_steps: int | None,
) -> tuple[np.ndarray, list[int], np.ndarray]:
    episode_length = len(dataset)
    action_dim = policy.config.output_features[ACTION].shape[0]
    predictions = np.full((episode_length, action_dim), np.nan, dtype=np.float32)
    per_horizon_errors: list[list[np.ndarray]] = [[] for _ in range(policy.config.chunk_size)]
    ground_truth = np.asarray(dataset.hf_dataset[ACTION], dtype=np.float32)
    anchors = list(range(0, episode_length, execution_horizon))

    for anchor_idx, anchor in enumerate(anchors):
        sample = dataset[anchor]
        observation = make_observation(sample)
        processed = preprocessor(observation)
        kwargs = make_prediction_kwargs(policy, seed, anchor_idx, num_inference_steps)

        with torch.inference_mode():
            normalized_chunk = policy.predict_action_chunk(processed, **kwargs)
            action_chunk = postprocessor(normalized_chunk).squeeze(0).cpu().float().numpy()

        valid = min(execution_horizon, episode_length - anchor, action_chunk.shape[0])
        predictions[anchor : anchor + valid] = action_chunk[:valid]

        comparison_length = min(action_chunk.shape[0], episode_length - anchor)
        absolute_error = np.abs(
            action_chunk[:comparison_length] - ground_truth[anchor : anchor + comparison_length]
        )
        for horizon in range(comparison_length):
            per_horizon_errors[horizon].append(absolute_error[horizon])

        print(f"anchor={anchor:4d}/{episode_length - 1} chunk_range=[{anchor}, {anchor + valid - 1}]")

    horizon_mae = np.full((policy.config.chunk_size, action_dim), np.nan, dtype=np.float32)
    for horizon, errors in enumerate(per_horizon_errors):
        if errors:
            horizon_mae[horizon] = np.mean(errors, axis=0)
    return predictions, anchors, horizon_mae


def finite_diff_mean(values: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros(values.shape[-1], dtype=np.float32)
    return np.mean(np.abs(np.diff(values, axis=0)), axis=0)


def compute_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    states: np.ndarray,
    anchors: list[int],
    joint_names: list[str],
) -> dict:
    valid = np.isfinite(predictions).all(axis=1)
    pred = predictions[valid]
    gt = ground_truth[valid]
    state = states[valid]
    errors = pred - gt

    boundary_indices = [anchor for anchor in anchors[1:] if anchor < len(predictions)]
    if boundary_indices:
        boundary_jumps = np.stack(
            [np.abs(predictions[index] - predictions[index - 1]) for index in boundary_indices]
        )
    else:
        boundary_jumps = np.zeros((0, predictions.shape[-1]), dtype=np.float32)

    all_jumps = np.abs(np.diff(predictions, axis=0))
    boundary_mask = np.zeros(len(all_jumps), dtype=bool)
    for index in boundary_indices:
        boundary_mask[index - 1] = True
    internal_jumps = all_jumps[~boundary_mask]
    state_baseline_mae = np.mean(np.abs(gt - state), axis=0)
    model_mae = np.mean(np.abs(errors), axis=0)

    metrics = {
        "num_frames": int(valid.sum()),
        "num_chunks": len(anchors),
        "joint_names": joint_names,
        "mae": model_mae.tolist(),
        "rmse": np.sqrt(np.mean(np.square(errors), axis=0)).tolist(),
        "prediction_step_abs_mean": finite_diff_mean(pred).tolist(),
        "ground_truth_step_abs_mean": finite_diff_mean(gt).tolist(),
        "state_step_abs_mean": finite_diff_mean(state).tolist(),
        "prediction_to_state_abs_mean": np.mean(np.abs(pred - state), axis=0).tolist(),
        "ground_truth_to_state_abs_mean": state_baseline_mae.tolist(),
        "mae_vs_current_state_baseline_ratio": (model_mae / np.maximum(state_baseline_mae, 1e-8)).tolist(),
        "prediction_internal_step_abs_mean": internal_jumps.mean(axis=0).tolist(),
        "chunk_boundary_abs_jump_mean": (
            boundary_jumps.mean(axis=0).tolist() if len(boundary_jumps) else [0.0] * pred.shape[-1]
        ),
        "chunk_boundary_abs_jump_max": (
            boundary_jumps.max(axis=0).tolist() if len(boundary_jumps) else [0.0] * pred.shape[-1]
        ),
        "prediction_boundary_to_internal_jump_ratio": (
            boundary_jumps.mean(axis=0) / np.maximum(internal_jumps.mean(axis=0), 1e-8)
        ).tolist(),
        "prediction_min": pred.min(axis=0).tolist(),
        "prediction_max": pred.max(axis=0).tolist(),
        "ground_truth_min": gt.min(axis=0).tolist(),
        "ground_truth_max": gt.max(axis=0).tolist(),
    }
    return metrics


def add_chunk_boundaries(axis, timestamps: np.ndarray, anchors: list[int]) -> None:
    for anchor in anchors[1:]:
        if anchor < len(timestamps):
            axis.axvline(timestamps[anchor], color="#ef4444", alpha=0.25, linewidth=0.8)


def plot_overview(
    output_path: Path,
    policy_label: str,
    timestamps: np.ndarray,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    states: np.ndarray,
    anchors: list[int],
    joint_names: list[str],
) -> None:
    groups = [range(0, 3), range(3, 6)]
    colors = ["#f97316", "#3b82f6", "#22c55e"]
    figure, axes = plt.subplots(1, 2, figsize=(18, 7), sharex=True)
    for axis, group in zip(axes, groups, strict=True):
        for color, joint_idx in zip(colors, group, strict=True):
            name = joint_names[joint_idx]
            axis.plot(timestamps, predictions[:, joint_idx], color=color, linewidth=1.8, label=f"{name} pred")
            axis.plot(
                timestamps,
                ground_truth[:, joint_idx],
                color=color,
                linewidth=1.4,
                linestyle="--",
                label=f"{name} GT action",
            )
            axis.plot(
                timestamps,
                states[:, joint_idx],
                color=color,
                linewidth=1.0,
                linestyle=":",
                alpha=0.8,
                label=f"{name} observation.state",
            )
        add_chunk_boundaries(axis, timestamps, anchors)
        axis.grid(alpha=0.25)
        axis.set_title(", ".join(joint_names[index] for index in group))
        axis.set_xlabel("time (s)")
        axis.set_ylabel("joint position")
        axis.legend(fontsize=8, ncol=2)
    figure.suptitle(
        f"{policy_label} dataset replay: prediction vs GT action vs observation.state\n"
        "red lines = replanning boundaries"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_per_joint(
    output_path: Path,
    policy_label: str,
    timestamps: np.ndarray,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    states: np.ndarray,
    anchors: list[int],
    joint_names: list[str],
) -> None:
    figure, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    for joint_idx, axis in enumerate(axes.flat):
        axis.plot(timestamps, predictions[:, joint_idx], label=f"{policy_label} prediction", linewidth=1.8)
        axis.plot(timestamps, ground_truth[:, joint_idx], label="GT action", linestyle="--")
        axis.plot(timestamps, states[:, joint_idx], label="observation.state", linestyle=":")
        add_chunk_boundaries(axis, timestamps, anchors)
        axis.set_title(joint_names[joint_idx])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("time (s)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_horizon_error(output_path: Path, horizon_mae: np.ndarray, joint_names: list[str]) -> None:
    figure, axis = plt.subplots(figsize=(12, 6))
    horizons = np.arange(len(horizon_mae))
    for joint_idx, name in enumerate(joint_names):
        axis.plot(horizons, horizon_mae[:, joint_idx], label=name)
    axis.set_xlabel("predicted action horizon")
    axis.set_ylabel("mean absolute error")
    axis.set_title("Action error by position inside each predicted chunk")
    axis.grid(alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_episode(args)
    config = PreTrainedConfig.from_pretrained(checkpoint)
    config.device = args.device
    config.pretrained_path = checkpoint
    if args.num_inference_steps is not None:
        config.num_inference_steps = args.num_inference_steps
    if args.num_inference_timesteps is not None:
        config.num_inference_timesteps = args.num_inference_timesteps

    saved_rename_map = read_saved_rename_map(checkpoint)
    policy = make_policy(config, ds_meta=dataset.meta, rename_map=saved_rename_map or None)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={"device_processor": {"device": args.device}},
        postprocessor_overrides={"device_processor": {"device": args.device}},
    )

    execution_horizon = args.execution_horizon or config.n_action_steps
    if execution_horizon > config.chunk_size:
        raise ValueError("execution_horizon cannot exceed policy.chunk_size")

    ground_truth, states, timestamps = raw_episode_arrays(dataset)
    joint_names = list(getattr(config, "action_feature_names", None) or DEFAULT_JOINT_NAMES)
    if len(joint_names) != ground_truth.shape[-1]:
        joint_names = DEFAULT_JOINT_NAMES[: ground_truth.shape[-1]]
    policy_label = args.policy_label or config.type

    predictions, anchors, horizon_mae = predict_chunked_episode(
        dataset=dataset,
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        execution_horizon=execution_horizon,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
    )
    metrics = compute_metrics(predictions, ground_truth, states, anchors, joint_names)
    metrics.update(
        {
            "checkpoint": str(checkpoint),
            "dataset_root": str(args.dataset_root.expanduser().resolve()),
            "episode": args.episode,
            "fps": dataset.meta.fps,
            "execution_horizon": execution_horizon,
            "chunk_size": config.chunk_size,
            "num_inference_steps": getattr(config, "num_inference_steps", None),
            "num_inference_timesteps": getattr(config, "num_inference_timesteps", None),
            "seed": args.seed,
            "rename_map": saved_rename_map,
            "first_action_mae": horizon_mae[0].tolist(),
        }
    )

    prefix = f"episode_{args.episode:03d}"
    np.savez_compressed(
        output_dir / f"{prefix}_actions.npz",
        timestamp=timestamps,
        prediction=predictions,
        ground_truth=ground_truth,
        observation_state=states,
        chunk_anchors=np.asarray(anchors),
        horizon_mae=horizon_mae,
    )
    (output_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_overview(
        output_dir / f"{prefix}_overview.png",
        policy_label,
        timestamps,
        predictions,
        ground_truth,
        states,
        anchors,
        joint_names,
    )
    plot_per_joint(
        output_dir / f"{prefix}_per_joint.png",
        policy_label,
        timestamps,
        predictions,
        ground_truth,
        states,
        anchors,
        joint_names,
    )
    plot_horizon_error(output_dir / f"{prefix}_horizon_mae.png", horizon_mae, joint_names)

    print(json.dumps(metrics, indent=2))
    print(f"Saved analysis to {output_dir}")


if __name__ == "__main__":
    main()
