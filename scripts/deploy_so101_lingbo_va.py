#!/usr/bin/env python
"""Deploy a trained LingBoVA checkpoint on a real SO101 arm.

This is a thin wrapper around `lerobot-rollout` (the repo's real-robot policy-deployment
entrypoint — NOT `lerobot-eval`, which is simulation-only) using the
`--strategy.type=lingbo_va_kv_cache` strategy (`LingboVaKvCacheStrategy`, registered in
`src/lerobot/rollout/strategies/lingbo_va_kv_cache.py`). That strategy correctly drives the
protocol LingBoVA's server needs: reset once, infer a chunk, execute it on the robot, then commit
the ACTUALLY-EXECUTED (post safety-clip) actions to the server's KV cache before requesting the
next chunk — plain `--strategy.type=base` does not do the KV-cache commit step and would silently
"replan from scratch" on every chunk instead of streaming.

Safety:
  - Per-joint absolute-range clip to the checkpoint's train-only-scoped [q01, q99] action range
    (LingboVaKvCacheStrategyConfig.action_range_pad_fraction, 0 = clip exactly to that range).
  - Per-step delta clip is the robot's own job: pass --max-relative-target-deg (default 5),
    forwarded to --robot.max_relative_target (SOFollowerConfig, uses degrees by default).
  - Ctrl+C triggers RolloutConfig's normal shutdown path, which smoothly interpolates the robot
    back to its startup position before disconnecting (--robot.max_relative_target and
    --return-to-initial-position control this; pass --no-return-to-initial-position to disable).
  - --dry-run runs the full sense -> preprocess -> infer loop (so you can see predicted actions
    and measure end-to-end latency) WITHOUT ever calling robot.send_action().

This script does not itself talk to the robot or the network — it builds and execs (or prints,
with --print-only) the equivalent `lerobot-rollout` command, so all the actual control-loop,
safety-clipping, and KV-cache logic lives in one place (LingboVaKvCacheStrategy /
SOFollower.send_action), not duplicated here.

Usage:
    python scripts/deploy_so101_lingbo_va.py \\
        --checkpoint-dir output_lerobot_train/three_cubes/lingbo_va/checkpoints/020000/pretrained_model \\
        --robot-port /dev/ttyACM1 \\
        --front-camera-index 4 --wrist-camera-index 2 --right-camera-index 0 \\
        --task "go to red cube. take the red cube. go to box. put the red cube in box." \\
        --dry-run

    # Once --dry-run output looks sane, drop --dry-run to actually execute:
    python scripts/deploy_so101_lingbo_va.py --checkpoint-dir ... --robot-port /dev/ttyACM1 \\
        --front-camera-index 4 --wrist-camera-index 2 --right-camera-index 0 --task "..."
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy a LingBoVA checkpoint on a real SO101 arm via lerobot-rollout.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint-dir", required=True, type=Path, help="Path to pretrained_model/ dir.")
    parser.add_argument("--robot-port", required=True, help="Serial port, e.g. /dev/ttyACM1.")
    parser.add_argument("--robot-id", default="follower_arm")
    parser.add_argument("--front-camera-index", type=int, default=4)
    parser.add_argument("--wrist-camera-index", type=int, default=2)
    parser.add_argument("--right-camera-index", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=640)
    parser.add_argument("--camera-height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--task", required=True, help="Task text (must match training's task string).")
    parser.add_argument("--fps", type=float, default=30.0, help="Control loop rate (Hz).")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds; 0 = run until Ctrl+C.")
    parser.add_argument(
        "--max-relative-target-deg",
        type=float,
        default=5.0,
        help="Per-step joint delta safety clip (degrees), enforced robot-side by SOFollower.",
    )
    parser.add_argument(
        "--action-range-pad-fraction",
        type=float,
        default=0.0,
        help="Padding (as a fraction of [q01,q99]) applied before the absolute-range action clip.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Never call robot.send_action().")
    parser.add_argument(
        "--no-return-to-initial-position",
        action="store_true",
        help="Skip smoothly returning to the startup pose on shutdown.",
    )
    parser.add_argument("--print-only", action="store_true", help="Print the lerobot-rollout command and exit.")
    return parser.parse_args()


def build_command(args: argparse.Namespace) -> list[str]:
    cameras = (
        "{"
        f"front: {{type: opencv, index_or_path: {args.front_camera_index}, "
        f"width: {args.camera_width}, height: {args.camera_height}, fps: {args.camera_fps}}}, "
        f"wrist: {{type: opencv, index_or_path: {args.wrist_camera_index}, "
        f"width: {args.camera_width}, height: {args.camera_height}, fps: {args.camera_fps}}}, "
        f"right: {{type: opencv, index_or_path: {args.right_camera_index}, "
        f"width: {args.camera_width}, height: {args.camera_height}, fps: {args.camera_fps}}}"
        "}"
    )
    cmd = [
        "lerobot-rollout",
        "--robot.type=so101_follower",
        f"--robot.port={args.robot_port}",
        f"--robot.id={args.robot_id}",
        f"--robot.cameras={cameras}",
        f"--robot.max_relative_target={args.max_relative_target_deg}",
        f"--policy.path={args.checkpoint_dir}",
        "--strategy.type=lingbo_va_kv_cache",
        f"--strategy.dry_run={'true' if args.dry_run else 'false'}",
        f"--strategy.action_range_pad_fraction={args.action_range_pad_fraction}",
        f"--fps={args.fps}",
        f"--duration={args.duration}",
        "--interpolation_multiplier=1",
        f"--task={args.task}",
        f"--return_to_initial_position={'false' if args.no_return_to_initial_position else 'true'}",
    ]
    return cmd


def main() -> None:
    args = parse_args()
    cmd = build_command(args)
    print("Running:", " ".join(cmd))
    if args.print_only:
        return
    os.execvp(cmd[0], cmd)  # noqa: S606 - intentional: replace this process with lerobot-rollout


if __name__ == "__main__":
    sys.exit(main())
