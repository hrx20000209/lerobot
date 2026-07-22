#!/usr/bin/env python
"""Repackage the local diffusers-style LingBot-VA base transformer as a LeRobot checkpoint.

The three_cubes_1 LoRA adapter names ``lerobot/lingbot_va_base`` as its
``base_model_name_or_path``. That Hub repo is a 10.2 GB ``model.safetensors`` -- but its
weights are the same dual-stream Wan transformer already present locally in diffusers
sharded form at ``~/Projects/models/lingbot-va-base/transformer`` (LeRobot vendors the
architecture in ``lerobot/policies/lingbot_va/utils.py:WanTransformer3DModel``; a
name-by-name check matches all 839 parameters/buffers, the checkpoint's only extras being
the unused ``patch_embedding.{weight,bias}`` superseded by ``patch_embedding_mlp``).

Repackaging locally avoids re-downloading 10 GB over a ~1.5 MB/s link. Pass
``--verify_against`` once the Hub copy is available to confirm the two are identical.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Superseded by patch_embedding_mlp in the dual-stream model; absent from the LeRobot module.
UNUSED_KEYS = {"patch_embedding.weight", "patch_embedding.bias"}


def load_diffusers_transformer(src: Path) -> dict[str, torch.Tensor]:
    index_path = src / "diffusion_pytorch_model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        shards = sorted(set(weight_map.values()))
    else:
        shards = ["diffusion_pytorch_model.safetensors"]
    tensors: dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(src / shard, "pt") as f:
            for k in f.keys():
                if k not in UNUSED_KEYS:
                    tensors[k] = f.get_tensor(k)
    return tensors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="/home/hrx/Projects/models/lingbot-va-base/transformer")
    p.add_argument("--out", default="/home/hrx/Projects/models/lingbot_va_base_lerobot")
    p.add_argument(
        "--verify_against",
        default=None,
        help="Optional path to the Hub lerobot/lingbot_va_base model.safetensors to diff against.",
    )
    args = p.parse_args()

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig
    from lerobot.policies.lingbot_va.utils import WanTransformer3DModel
    from lerobot.utils.constants import ACTION

    arch = {k: v for k, v in json.loads((src / "config.json").read_text()).items() if not k.startswith("_")}
    arch["patch_size"] = tuple(arch["patch_size"])
    arch["attn_mode"] = "torch"

    # Name-check against the actual module before writing 10 GB.
    with torch.device("meta"):
        module = WanTransformer3DModel(**arch)
    expected = set(dict(module.named_parameters())) | set(dict(module.named_buffers()))

    tensors = load_diffusers_transformer(src)
    missing, extra = expected - set(tensors), set(tensors) - expected
    if missing or extra:
        raise SystemExit(f"state-dict mismatch\n  missing: {sorted(missing)[:8]}\n  extra: {sorted(extra)[:8]}")
    print(f"matched all {len(expected)} transformer tensors")

    if args.verify_against:
        with safe_open(args.verify_against, "pt") as f:
            hub_keys = set(f.keys())
            ours = {f"transformer.{k}" for k in tensors}
            if hub_keys != ours:
                print(f"  !! key sets differ: only-hub={sorted(hub_keys - ours)[:5]} only-ours={sorted(ours - hub_keys)[:5]}")
            worst, worst_k = 0.0, None
            for k in sorted(hub_keys & ours):
                d = (f.get_tensor(k).float() - tensors[k[len("transformer.") :]].float()).abs().max().item()
                if d > worst:
                    worst, worst_k = d, k
            print(f"  max abs diff vs Hub base: {worst} (at {worst_k})")
            if worst > 0:
                raise SystemExit("local repackage differs from the Hub base -- do not use it for this adapter")

    config = LingBotVAConfig(attn_mode="torch")
    config.input_features = {
        "observation.images.image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640))
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(len(config.used_action_channel_ids),))
    }
    config._save_pretrained(out)

    save_file(
        {f"transformer.{k}": v for k, v in tensors.items()},
        str(out / "model.safetensors"),
        metadata={"format": "pt"},
    )
    print(f"wrote {out}/model.safetensors ({sum(t.numel() for t in tensors.values()) / 1e9:.2f}B params)")


if __name__ == "__main__":
    main()
