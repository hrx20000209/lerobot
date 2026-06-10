#!/usr/bin/env python3
"""Run STRIDE image-to-actions, then feed the parsed plan to LeRobot async client."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_CAMERAS = (
    '{ front: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, '
    'wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30, fourcc: "MJPG"}, '
    'right: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30, fourcc: "MJPG"}}'
)


def expand_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def parse_action_plan(text: str) -> list[str]:
    match = re.search(
        r"^=== Action plan ===\s*\n(?P<plan>.*?)\n=+\s*$",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if not match:
        return []

    lines = []
    for line in match.group("plan").splitlines():
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
    return lines


def read_plan_from_file(plan_file: Path) -> list[str]:
    if not plan_file.exists():
        return []

    lines = []
    for line in plan_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
    return lines


def format_plan(lines: list[str], mode: str) -> str:
    if mode == "newline":
        return "\n".join(lines)
    if mode == "semicolon":
        return "; ".join(lines)
    if mode == "sentence":
        return ". ".join(line.rstrip(".") for line in lines) + "."
    raise ValueError(f"Unknown plan format: {mode}")


def print_command(argv: list[str]) -> None:
    print(" ".join(subprocess.list2cmdline([arg]) for arg in argv))


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    home = Path.home()

    parser = argparse.ArgumentParser(
        description=(
            "Run image_to_actions/run_image_to_actions.sh, extract the Action plan, "
            "and pass it as --task to lerobot.async_inference.robot_client."
        )
    )
    parser.add_argument(
        "--image",
        default=str(home / "Projects/lerobot/outputs/captured_images/opencv__dev_video0.png"),
        help="Image used by STRIDE image_to_actions.",
    )
    parser.add_argument(
        "--planning-task",
        default="move the blue cube into the box",
        help="High-level task passed to run_image_to_actions.sh.",
    )
    parser.add_argument(
        "--stride-dir",
        default=str(
            home
            / "Projects/stride_image_to_actions_thor/trip_final_ckpt_transfer_20260521/image_to_actions"
        ),
        help="Directory containing run_image_to_actions.sh.",
    )
    parser.add_argument(
        "--lerobot-dir",
        default=str(repo_root),
        help="LeRobot repo directory. The client is run from this directory.",
    )
    parser.add_argument("--python", default=sys.executable, help="Python executable for the LeRobot client.")
    parser.add_argument("--server-address", default="127.0.0.1:8080")
    parser.add_argument("--robot-type", default="so101_follower")
    parser.add_argument("--robot-port", default="/dev/ttyACM0")
    parser.add_argument("--robot-id", default="follower_arm")
    parser.add_argument("--robot-cameras", default=DEFAULT_CAMERAS)
    parser.add_argument("--policy-type", default="pi0")
    parser.add_argument("--pretrained-name-or-path", default="/home/hrx/Projects/red_rectangle/models/pi0/")
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--actions-per-chunk", type=int, default=50)
    parser.add_argument("--chunk-size-threshold", type=float, default=0.9)
    parser.add_argument("--aggregate-fn-name", default="conservative")
    parser.add_argument("--debug-visualize-queue-size", default="True")
    parser.add_argument(
        "--plan-format",
        choices=("newline", "semicolon", "sentence"),
        default="newline",
        help="How to turn plan lines into the client --task string.",
    )
    parser.add_argument(
        "--task-prefix",
        default="",
        help="Optional text prepended before the generated plan in --task.",
    )
    parser.add_argument(
        "--plan-file",
        default="",
        help="Use an existing action_plan.txt instead of running run_image_to_actions.sh.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run/parse the planner, print the LeRobot command, but do not start the robot client.",
    )
    parser.add_argument(
        "--client-extra-arg",
        action="append",
        default=[],
        help="Extra raw argument passed to robot_client. Can be repeated.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    stride_dir = expand_path(args.stride_dir)
    lerobot_dir = expand_path(args.lerobot_dir)
    image = expand_path(args.image)

    if args.plan_file:
        plan_file = expand_path(args.plan_file)
        plan_lines = read_plan_from_file(plan_file)
        out_dir = plan_file.parent
    else:
        run_script = stride_dir / "run_image_to_actions.sh"
        if not run_script.exists():
            print(f"run_image_to_actions.sh not found: {run_script}", file=sys.stderr)
            return 2
        if not image.exists():
            print(f"image not found: {image}", file=sys.stderr)
            return 2

        out_dir = stride_dir / "runs" / f"lerobot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        env = os.environ.copy()
        env["OUT_DIR"] = str(out_dir)

        planner_cmd = [str(run_script), str(image), args.planning_task]
        print("[stride] running image_to_actions:")
        print_command(planner_cmd)
        result = subprocess.run(
            planner_cmd,
            cwd=str(stride_dir),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(result.stdout, end="")
        if result.returncode != 0:
            print(f"[stride] failed with exit code {result.returncode}", file=sys.stderr)
            return result.returncode

        plan_lines = parse_action_plan(result.stdout)
        if not plan_lines:
            plan_lines = read_plan_from_file(out_dir / "action_plan.txt")

    if not plan_lines:
        print("[stride] no action plan was parsed.", file=sys.stderr)
        print(f"[stride] artifacts dir: {out_dir}", file=sys.stderr)
        return 3

    plan_text = format_plan(plan_lines, args.plan_format)
    if args.task_prefix.strip():
        plan_text = f"{args.task_prefix.strip()}\n{plan_text}"

    print("\n[lerobot] task text:")
    print(plan_text)
    print(f"[stride] artifacts dir: {out_dir}")

    client_cmd = [
        args.python,
        "-m",
        "lerobot.async_inference.robot_client",
        f"--server_address={args.server_address}",
        f"--robot.type={args.robot_type}",
        f"--robot.port={args.robot_port}",
        f"--robot.id={args.robot_id}",
        f"--robot.cameras={args.robot_cameras}",
        f"--task={plan_text}",
        f"--policy_type={args.policy_type}",
        f"--pretrained_name_or_path={args.pretrained_name_or_path}",
        f"--policy_device={args.policy_device}",
        f"--actions_per_chunk={args.actions_per_chunk}",
        f"--chunk_size_threshold={args.chunk_size_threshold}",
        f"--aggregate_fn_name={args.aggregate_fn_name}",
        f"--debug_visualize_queue_size={args.debug_visualize_queue_size}",
        *args.client_extra_arg,
    ]

    print("\n[lerobot] command:")
    print_command(client_cmd)

    if args.dry_run:
        print("[lerobot] dry-run enabled; robot client was not started.")
        return 0

    return subprocess.run(client_cmd, cwd=str(lerobot_dir)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
