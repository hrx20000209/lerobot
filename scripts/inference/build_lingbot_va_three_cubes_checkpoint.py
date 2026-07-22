#!/usr/bin/env python
"""Assemble an async-servable LingBot-VA checkpoint from the three_cubes_1 LoRA adapter.

``examples/lingbot_va_so101/train_lingbot_va_so101.py`` saves its checkpoints with
``PeftModel.save_pretrained``, which writes only ``adapter_config.json`` +
``adapter_model.safetensors``. The async ``PolicyServer`` needs more than that
(``policy_server.py:_load_policy`` / ``SendPolicyInstructions``):

  * ``config.json``     -- ``PreTrainedConfig.from_pretrained`` is the first thing
                           ``_load_policy`` calls, and it must carry ``use_peft: true``
                           or the adapter branch is skipped entirely.
  * ``policy_preprocessor.json`` / ``policy_postprocessor.json`` (+ the unnormalizer
                           safetensors) -- ``make_pre_post_processors(..., pretrained_path=...)``
                           loads them from the same directory. The postprocessor holds the
                           q01/q99 that map the policy's [-1, 1] actions back to degrees.

The policy config here mirrors ``examples/lingbot_va_so101/common.py:build_config()``
exactly, because the adapter was trained against it -- notably the joint-space action
channels [14..18, 28] and the 3-camera width_concat layout. Inference-only knobs
(``attn_mode``, the two ``*_inference_steps``) are the documented exceptions and are
overridable from the CLI.

Usage:
    python scripts/inference/build_lingbot_va_three_cubes_checkpoint.py
    python scripts/inference/build_lingbot_va_three_cubes_checkpoint.py --video_steps 5 --action_steps 10
"""

import argparse
import json
import shutil
from pathlib import Path

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig

# Kept in sync with examples/lingbot_va_so101/common.py (the adapter's training config).
OBS_CAM_KEYS = ["observation.images.front", "observation.images.right", "observation.images.wrist"]
USED_ACTION_CHANNEL_IDS = [14, 15, 16, 17, 18, 28]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter_dir", default="/home/hrx/Projects/models/three_cubes_1/lingbot_va_adapter")
    p.add_argument("--output_dir", default="/home/hrx/Projects/models/three_cubes_1/lingbot_va_async")
    p.add_argument("--dataset_root", default="/home/hrx/Datasets/three_cubes_1")
    p.add_argument("--dataset_repo_id", default="three_cubes_1")
    p.add_argument(
        "--wan_pretrained_path",
        default="/home/hrx/Projects/models/lingbot-va-base",
        help="Local diffusers-style dir holding vae/, text_encoder/, tokenizer/ (~20 GB frozen).",
    )
    p.add_argument(
        "--base_model_path",
        default="/home/hrx/Projects/models/lingbot_va_base_lerobot",
        help="Overrides the adapter's base_model_name_or_path (default 'lerobot/lingbot_va_base', a "
        "10.2 GB Hub download) with a local LeRobot-format base. Build one with "
        "scripts/inference/build_lingbot_va_base_lerobot.py. Pass '' to keep the Hub id.",
    )
    p.add_argument(
        "--video_steps",
        type=int,
        default=5,
        help="num_inference_steps. Training default is 20; the upstream SO-101 deploy config "
        "(wan_va/configs/va_so101_cfg.py) serves at 5.",
    )
    p.add_argument(
        "--action_steps",
        type=int,
        default=10,
        help="action_num_inference_steps. Training default is 50; upstream SO-101 deploy uses 10.",
    )
    p.add_argument(
        "--text_encoder_device",
        default="cuda",
        help="The config default is 'cpu' (saves ~11 GB VRAM), but UMT5-XXL then takes ~230 s for the "
        "one prompt encode at episode start vs ~1.7 s on GPU (measured on Jetson Thor, where the "
        "37 GB peak fits the unified 122 GB anyway).",
    )
    args = p.parse_args()

    adapter_dir = Path(args.adapter_dir)
    out = Path(args.output_dir)
    if not (adapter_dir / "adapter_model.safetensors").exists():
        raise SystemExit(f"No adapter_model.safetensors in {adapter_dir}")
    out.mkdir(parents=True, exist_ok=True)

    ds_meta = LeRobotDatasetMetadata(args.dataset_repo_id, root=args.dataset_root)

    # attn_mode="torch": "flex" is training-only -- its block-causal mask is a class-level
    # attribute sized for the training chunk, and predict_action_chunk() crashes on it
    # (see the docstring of examples/lingbot_va_so101/eval_checkpoint.py).
    config = LingBotVAConfig(
        obs_cam_keys=OBS_CAM_KEYS,
        camera_layout="width_concat",
        used_action_channel_ids=USED_ACTION_CHANNEL_IDS,
        attn_mode="torch",
        wan_pretrained_path=args.wan_pretrained_path,
        text_encoder_device=args.text_encoder_device,
        device="cuda",
        num_inference_steps=args.video_steps,
        action_num_inference_steps=args.action_steps,
    )
    config.use_peft = True

    # The server builds the policy from these features; the dataset is the source of truth.
    from lerobot.configs.types import FeatureType, PolicyFeature
    from lerobot.utils.constants import ACTION, OBS_STATE

    config.input_features = {
        k: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)) for k in OBS_CAM_KEYS
    }
    config.input_features[OBS_STATE] = PolicyFeature(type=FeatureType.STATE, shape=(6,))
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(len(USED_ACTION_CHANNEL_IDS),))
    }
    config._save_pretrained(out)

    # _save_pretrained() serializes the dataclass fields only; use_peft lives on the base
    # PreTrainedConfig, so make sure it survives into the JSON the server reads back.
    cfg_path = out / "config.json"
    cfg_json = json.loads(cfg_path.read_text())
    cfg_json["use_peft"] = True
    cfg_path.write_text(json.dumps(cfg_json, indent=4))

    for name in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copy(adapter_dir / name, out / name)

    # _load_policy resolves the base model through the adapter config, so repoint it at the
    # local repackaged base rather than re-downloading 10.2 GB from the Hub.
    if args.base_model_path:
        acfg_path = out / "adapter_config.json"
        acfg = json.loads(acfg_path.read_text())
        acfg["base_model_name_or_path"] = args.base_model_path
        acfg_path.write_text(json.dumps(acfg, indent=2))

    stats = {k: {kk: torch.as_tensor(vv) for kk, vv in v.items()} for k, v in ds_meta.stats.items()}
    preprocessor, postprocessor = make_pre_post_processors(config, dataset_stats=stats)
    preprocessor.save_pretrained(out)
    postprocessor.save_pretrained(out)

    q01 = [round(float(x), 3) for x in ds_meta.stats["action"]["q01"]]
    q99 = [round(float(x), 3) for x in ds_meta.stats["action"]["q99"]]
    print(f"Wrote async-servable checkpoint to {out}")
    print(f"  chunk_size            = {config.chunk_size} actions ({config.chunk_size / 30:.2f}s @30Hz)")
    print(f"  inference steps       = video {config.num_inference_steps} / action {config.action_num_inference_steps}")
    print(f"  action q01            = {q01}")
    print(f"  action q99            = {q99}")
    print("  files:", sorted(f.name for f in out.iterdir()))


if __name__ == "__main__":
    main()
