#!/usr/bin/env python
"""Print Pi0.5 checkpoint config, processor, and normalizer diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


def _json(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def _print_json_field(data: dict, key: str) -> None:
    print(f"{key}: {json.dumps(data.get(key), ensure_ascii=False, indent=2)}")


def _processor_steps(path: Path) -> list[str]:
    data = _json(path)
    return [step["registry_name"] for step in data["steps"]]


def _stats_from_flat_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, dict[str, torch.Tensor]]:
    stats: dict[str, dict[str, torch.Tensor]] = {}
    for key, value in tensors.items():
        feature, stat = key.rsplit(".", 1)
        stats.setdefault(feature, {})[stat] = value
    return stats


def _print_feature_stats(stats: dict[str, dict[str, torch.Tensor]], feature: str) -> None:
    print(f"\n{feature} stats:")
    feature_stats = stats.get(feature)
    if feature_stats is None:
        print("  missing")
        return

    for name in ("count", "min", "q01", "q10", "q50", "q90", "q99", "max", "mean", "std"):
        value = feature_stats.get(name)
        if value is None:
            continue
        flat = value.flatten().tolist()
        print(f"  {name:>5} shape={tuple(value.shape)} values={[round(float(x), 6) for x in flat]}")

    if "q01" in feature_stats and "q99" in feature_stats:
        value = feature_stats["q99"] - feature_stats["q01"]
        print(f"  q99-q01 shape={tuple(value.shape)} values={[round(float(x), 6) for x in value.flatten().tolist()]}")


def _print_action_unnorm_examples(stats: dict[str, dict[str, torch.Tensor]]) -> None:
    action = stats["action"]
    q01 = action["q01"]
    q99 = action["q99"]
    print("\naction quantile unnormalize examples:")
    for normalized in (-1.0, 0.0, 1.0):
        values = (normalized + 1.0) * (q99 - q01) / 2.0 + q01
        print(f"  norm {normalized:+.1f}: {[round(float(x), 6) for x in values.flatten().tolist()]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    checkpoint = args.checkpoint.expanduser().resolve()
    config = _json(checkpoint / "config.json")
    preprocessor = _json(checkpoint / "policy_preprocessor.json")
    postprocessor = _json(checkpoint / "policy_postprocessor.json")

    print(f"checkpoint: {checkpoint}")
    for key in (
        "type",
        "chunk_size",
        "n_action_steps",
        "max_action_dim",
        "max_state_dim",
        "input_features",
        "output_features",
        "action_feature_names",
        "use_relative_actions",
        "relative_exclude_joints",
        "normalization_mapping",
    ):
        _print_json_field(config, key)

    image_features = [key for key, value in config["input_features"].items() if value["type"] == "VISUAL"]
    print(f"image_features: {json.dumps(image_features, ensure_ascii=False, indent=2)}")

    print(f"saved preprocessor steps: {_processor_steps(checkpoint / 'policy_preprocessor.json')}")
    print(f"saved postprocessor steps: {_processor_steps(checkpoint / 'policy_postprocessor.json')}")

    for label, processor in (("preprocessor", preprocessor), ("postprocessor", postprocessor)):
        print(f"\n{label} normalizer/unnormalizer config:")
        for step in processor["steps"]:
            if step["registry_name"] in {"normalizer_processor", "unnormalizer_processor"}:
                print(json.dumps(step["config"], ensure_ascii=False, indent=2))

    tensors = load_file(checkpoint / "policy_postprocessor_step_0_unnormalizer_processor.safetensors")
    stats = _stats_from_flat_tensors(tensors)
    print(f"\nnormalizer stats keys: {sorted(stats)}")
    _print_feature_stats(stats, "action")
    _print_feature_stats(stats, "observation.state")
    _print_action_unnorm_examples(stats)


if __name__ == "__main__":
    main()
