#!/usr/bin/env python
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Real-robot synchronous inference for a joint-space lingbot_va checkpoint on SO-101.

*** HAS NOT BEEN RUN ON REAL HARDWARE. Start with --dry_run (the default) and read a few
    printed actions before ever passing --execute. ***

Why this is a standalone script and not `lerobot-rollout`
-----------------------------------------------------------
`lerobot-rollout --inference.type=sync` is the normal path and its per-tick call pattern
(one policy call per control tick) is exactly what we validated works correctly for
lingbot_va (see the session's sync_select_action_test.py: `select_action()`'s internal
KV-cache/frame_st_id bookkeeping advances correctly when called this way). Two things
about *this* checkpoint don't fit `lerobot-rollout`'s standard loading path, though:

  1. It's a LoRA adapter (PEFT) over `lerobot/lingbot_va_base`, saved via plain
     `policy.save_pretrained()` from a hand-rolled training loop -- no
     preprocessor/postprocessor pipeline was ever built or saved alongside it (our
     training script normalized actions manually with dataset quantiles, not through
     lerobot's processor pipeline).
  2. The channels used are 14-18+28 (joint-space, not the EEF channels the released
     checkpoints/postprocessors assume) -- the correct unnormalization constants are this
     dataset's own q01/q99 (from `three_cubes_1`'s meta/stats.json), not anything bundled
     with the checkpoint.

This script therefore loads the base + adapter explicitly and reconstructs the same
quantile denormalization used during training/eval this session, then drives the policy
through `select_action()` exactly like lerobot-rollout's sync backend would.

DO NOT use the async (`lerobot.async_inference`) path for this policy: `lingbot_va` isn't
in `SUPPORTED_POLICIES`, and separately, calling `predict_action_chunk()` directly (what
the async server does) bypasses `select_action()`'s observation buffering -- confirmed via
async_repro_test.py this session that the KV cache silently stops updating after the first
chunk, i.e. the policy goes blind to all further observations without erroring.

Safety
------
- `--dry_run` (default true): prints what *would* be sent, never calls `robot.send_action`.
- `--execute` must be passed explicitly (and implies dry_run=false) to command real motors.
- `--max_relative_target` (degrees) is forwarded to the robot config; SOFollower clips any
  single-step goal that's too far from the current position before it ever reaches the bus.
- `--max_steps` / `--duration_s` bound how long the loop can run.
- Every commanded action is checked for NaN/Inf before being sent; the loop stops on the
  first offending action rather than sending it.
- Ctrl+C triggers a clean robot.disconnect() (torque disabled per SOFollowerConfig default).

Before running for real, confirm the robot's calibration file matches the one used to
record `three_cubes_1` (this session's Step 0 found it at ~/Projects/follower_arm.json, not
the LeRobot-default cache path) and that the three cameras are wired to the same physical
positions as `front` / `right` / `wrist` were during recording -- camera order is
order-sensitive for this model (latents are concatenated on width).

Example
-------
    # Dry run first -- just prints actions, sends nothing to the motors
    python -m lerobot.scripts.inference.run_lingbot_va_so101 \\
        --robot.port=/dev/ttyACM0 \\
        --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, \\
right: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, \\
wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}}" \\
        --checkpoint_dir=/data/rxhuang/lingbot_va_runs/full_20k/checkpoints/step_2000 \\
        --duration_s=15

    # Only after the dry-run output looks sane:
    python -m lerobot.scripts.inference.run_lingbot_va_so101 \\
        --robot.port=/dev/ttyACM0 --robot.cameras=... \\
        --checkpoint_dir=/data/rxhuang/lingbot_va_runs/full_20k/checkpoints/step_2000 \\
        --duration_s=15 --max_relative_target=5 --execute
"""

import logging
import time
from dataclasses import dataclass, field

import draccus
import numpy as np
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy
from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig
from lerobot.robots import Robot, RobotConfig, make_robot_from_config, so_follower  # noqa: F401
from lerobot.utils.utils import init_logging

logger = logging.getLogger(__name__)

# Matches this session's common.py: 5 arm joints + gripper, in the dataset's own column
# order, mapped to the "left-arm joints (unused by released checkpoints)" + "left gripper"
# channels -- a joint-space fine-tune of the base (non-EEF) checkpoint.
USED_ACTION_CHANNEL_IDS = [14, 15, 16, 17, 18, 28]
ACTION_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
OBS_CAM_KEYS = ["observation.images.front", "observation.images.right", "observation.images.wrist"]
# robot.get_observation() camera keys, in the same order as OBS_CAM_KEYS
ROBOT_CAM_KEYS = ["front", "right", "wrist"]
DEFAULT_TASK = "go to red cube. take the red cube. go to box. put the red cube in box."
BASE_CHECKPOINT = "lerobot/lingbot_va_base"
WAN_PRETRAINED_PATH = "/home/rxhuang/Projects/models/lingbot-va-base"
DATASET_ROOT = "/data/rxhuang/three_cubes_1"


@dataclass
class RunConfig:
    robot: RobotConfig
    checkpoint_dir: str
    task: str = DEFAULT_TASK
    fps: int = 30
    duration_s: float = 30.0
    max_steps: int | None = None
    device: str = "cuda"
    dry_run: bool = True
    execute: bool = False
    max_relative_target: float | None = 5.0
    log_every: int = 10

    def __post_init__(self):
        if self.execute:
            self.dry_run = False
        if self.max_steps is None:
            self.max_steps = int(self.duration_s * self.fps)


def load_policy(checkpoint_dir: str, device: str):
    """Base checkpoint + LoRA adapter, attn_mode='torch' (select_action/predict_action_chunk
    are inference-only; attn_mode='flex' is training-only, see this session's finding that
    mixing the two throws 'block_mask was created for ... but got q_len=...')."""
    config = LingBotVAConfig(
        obs_cam_keys=OBS_CAM_KEYS,
        camera_layout="width_concat",
        used_action_channel_ids=USED_ACTION_CHANNEL_IDS,
        attn_mode="torch",
        wan_pretrained_path=WAN_PRETRAINED_PATH,
        text_encoder_device="cpu",
        device=device,
        pretrained_path=BASE_CHECKPOINT,
    )
    ds_meta = LeRobotDatasetMetadata("three_cubes_1", root=DATASET_ROOT)
    base_policy = make_policy(config, ds_meta=ds_meta)

    from peft import PeftModel

    policy = PeftModel.from_pretrained(base_policy, checkpoint_dir, is_trainable=False).get_base_model()
    policy.eval()

    q01 = torch.tensor(ds_meta.stats["action"]["q01"], dtype=torch.float32)
    q99 = torch.tensor(ds_meta.stats["action"]["q99"], dtype=torch.float32)
    return policy, config, q01, q99


def denormalize_action(action_norm: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor) -> torch.Tensor:
    q01 = q01.to(action_norm.device, action_norm.dtype)
    q99 = q99.to(action_norm.device, action_norm.dtype)
    return (action_norm + 1) / 2 * (q99 - q01) + q01


def raw_image_to_tensor(image: np.ndarray, device: str) -> torch.Tensor:
    """(H, W, C) uint8 from robot.get_observation() -> (1, C, H, W) float32 in [0, 1]."""
    t = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0).to(device)


def build_policy_batch(raw_obs: dict, task: str, device: str) -> dict:
    batch = {
        cam_key: raw_image_to_tensor(raw_obs[robot_key], device)
        for cam_key, robot_key in zip(OBS_CAM_KEYS, ROBOT_CAM_KEYS, strict=True)
    }
    batch["task"] = [task]
    return batch


def action_tensor_to_robot_dict(action_deg: torch.Tensor) -> dict[str, float]:
    values = action_deg.detach().float().cpu().tolist()
    return {f"{name}.pos": value for name, value in zip(ACTION_NAMES, values, strict=True)}


@draccus.wrap()
def main(cfg: RunConfig):
    init_logging()
    logger.info("Config: %s", cfg)
    if cfg.dry_run:
        logger.warning(
            "DRY RUN: predicted actions will be printed but NEVER sent to the robot. "
            "Pass --execute once the printed actions look sane."
        )
    else:
        logger.warning(
            "LIVE MODE: actions WILL be sent to real motors. max_relative_target=%s deg. "
            "This checkpoint/script has not been validated on real hardware -- watch the "
            "e-stop / power switch.",
            cfg.max_relative_target,
        )

    if cfg.max_relative_target is not None:
        cfg.robot.max_relative_target = cfg.max_relative_target

    logger.info("Loading policy from %s ...", cfg.checkpoint_dir)
    policy, policy_config, q01, q99 = load_policy(cfg.checkpoint_dir, cfg.device)
    policy.reset()
    logger.info("Policy loaded. n_action_steps (chunk_size)=%d", policy_config.n_action_steps)

    robot: Robot = make_robot_from_config(cfg.robot)
    robot.connect()
    logger.info("Robot connected: %s", robot.name)

    control_dt = 1.0 / cfg.fps
    step = 0
    try:
        start_t = time.perf_counter()
        while step < cfg.max_steps:
            loop_t0 = time.perf_counter()

            raw_obs = robot.get_observation()
            batch = build_policy_batch(raw_obs, cfg.task, cfg.device)

            with torch.no_grad():
                action_norm = policy.select_action(batch)  # [1, n_used], normalized [-1, 1]-ish
            action_deg = denormalize_action(action_norm.reshape(-1), q01, q99)

            if torch.isnan(action_deg).any() or torch.isinf(action_deg).any():
                logger.error("Predicted action has NaN/Inf at step %d: %s -- stopping.", step, action_deg.tolist())
                break

            action_dict = action_tensor_to_robot_dict(action_deg)

            if step % cfg.log_every == 0:
                logger.info(
                    "step=%d frame_st_id=%s action=%s",
                    step,
                    getattr(policy, "_frame_st_id", None),
                    {k: round(v, 2) for k, v in action_dict.items()},
                )

            if not cfg.dry_run:
                sent = robot.send_action(action_dict)
                if step % cfg.log_every == 0:
                    logger.debug("sent (post-clip)=%s", sent)

            step += 1
            elapsed = time.perf_counter() - loop_t0
            time.sleep(max(0.0, control_dt - elapsed))

        logger.info("Finished: %d steps in %.1fs", step, time.perf_counter() - start_t)

    except KeyboardInterrupt:
        logger.info("Interrupted by user at step %d", step)
    finally:
        robot.disconnect()
        logger.info("Robot disconnected.")


if __name__ == "__main__":
    main()
