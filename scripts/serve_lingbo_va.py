#!/usr/bin/env python
"""Start a LingBoVA inference server (websocket) for a trained LeRobot checkpoint.

This is a thin wrapper around the vendored `wan_va.wan_va_server.VA_Server` +
`run_async_server_mode`: it loads a `LingBoVAPolicy` checkpoint (for its already-correct
`_make_va_job_config()` — camera keys, action-channel mapping, and train-only-scoped norm stats
all come straight from what the checkpoint was actually trained with, not from a possibly-stale
vendored config registry entry), builds the equivalent server job config, and serves it.

Deploy-time inference REQUIRES `inference_attn_mode` to be "torch" or "flashattn" — "flex" is
architecturally training-only (the block-causal training mask is only built inside
`WanTransformer3DModel.forward_train`, never in the streaming/KV-cache `forward()` path used
here). Every checkpoint saved by `scripts/train_lingbo_va.sh` bakes `attn_mode="flex"` into
`transformer/config.json` (that's what the training-time transformer instance used); the vendored
`VA_Server.__init__` already force-overrides this to "torch" unconditionally when it loads the
transformer, so no manual config.json editing is needed — this script additionally asserts the
checkpoint's `--inference-attn-mode` argument itself is never "flex", as a second line of defense.

Usage:
    python scripts/serve_lingbo_va.py \\
        --checkpoint-dir output_lerobot_train/three_cubes/lingbo_va/checkpoints/020000/pretrained_model \\
        --host 0.0.0.0 --port 29056

    # Then point a client at it, e.g.:
    #   --policy.inference_backend=server --policy.server_host=<ip> --policy.server_port=29056
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("serve_lingbo_va")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve a trained LingBoVA checkpoint over websocket for remote inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint-dir", required=True, type=Path, help="Path to pretrained_model/ dir.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=29056)
    parser.add_argument("--save-root", default=None, help="Override job_config.save_root (server-side debug dumps).")
    parser.add_argument(
        "--inference-attn-mode",
        default="torch",
        choices=["torch", "flashattn"],
        help="Attention op for the loaded transformer. 'flex' is rejected (training-only).",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.inference_attn_mode == "flex":
        raise ValueError("inference_attn_mode='flex' is not supported for serving; use torch or flashattn.")

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.port + 10000))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")

    from lerobot.policies.lingbo_va.modeling_lingbo_va import LingBoVAPolicy

    logger.info("Loading LingBoVA policy from %s (for its job config only, no transformer load).", args.checkpoint_dir)
    policy = LingBoVAPolicy.from_pretrained(args.checkpoint_dir)
    policy.config.inference_attn_mode = args.inference_attn_mode
    policy.config.device = args.device

    job_config = policy._make_va_job_config()
    job_config.host = args.host
    job_config.port = args.port
    job_config.infer_mode = "server"
    if args.save_root is not None:
        job_config.save_root = args.save_root

    policy._install_lingbo_import_path()
    import torch
    from wan_va.distributed.util import init_distributed
    from wan_va.utils import run_async_server_mode
    from wan_va.wan_va_server import VA_Server

    init_distributed(world_size=1, local_rank=0, rank=0)
    torch.cuda.set_device(0)

    job_config.rank = 0
    job_config.local_rank = 0
    job_config.world_size = 1

    logger.info(
        "Starting LingBoVA server on %s:%d — camera_keys=%s used_action_channel_ids=%s attn_mode(forced by VA_Server)=torch",
        args.host,
        args.port,
        job_config.obs_cam_keys,
        job_config.used_action_channel_ids,
    )
    model = VA_Server(job_config)
    run_async_server_mode(model, local_rank=0, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
