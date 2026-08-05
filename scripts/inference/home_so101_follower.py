#!/usr/bin/env python
"""Ramp an SO101 follower back to a rest pose, slowly and under position control.

This writes goal positions -- the arm moves.  Motion is a coordinated linear
interpolation from the measured pose to the target: every command step moves the
fastest joint by at most ``--step-deg``, and the other joints are scaled by the
same factor so the arm follows a straight line in joint space instead of each
joint racing to its target independently.

``--max-relative-target`` stays enabled as a per-command backstop.

Default target is the pose the arm rested at before the 2026-08-05 closed-loop
run (folded down, gripper closed).
"""

import argparse
import time

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Measured resting pose prior to the first closed-loop run.
DEFAULT_HOME = [0.80, -101.63, 91.43, 73.05, -2.01, 1.78]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--id", default="follower_arm")
    parser.add_argument(
        "--target",
        default=",".join(str(v) for v in DEFAULT_HOME),
        help=f"Comma-separated goal for {JOINTS}",
    )
    parser.add_argument("--step-deg", type=float, default=1.0, help="Max degrees the fastest joint moves per command")
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--max-relative-target", type=float, default=2.0)
    parser.add_argument("--settle-s", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit without moving")
    args = parser.parse_args()

    target = [float(v) for v in args.target.split(",")]
    if len(target) != len(JOINTS):
        raise SystemExit(f"--target needs {len(JOINTS)} values for {JOINTS}, got {len(target)}")

    robot = SOFollower(
        SOFollowerRobotConfig(
            port=args.port,
            id=args.id,
            cameras={},
            use_degrees=True,
            max_relative_target=args.max_relative_target,
            # Leave the arm holding at the end rather than letting it drop.
            disable_torque_on_disconnect=False,
        )
    )
    robot.connect()
    try:
        present = robot.bus.sync_read("Present_Position")
        start = [present[j] for j in JOINTS]
        deltas = [t - s for s, t in zip(start, target)]
        largest = max(abs(d) for d in deltas)
        steps = max(1, int(round(largest / args.step_deg)))

        print(f"{'joint':<16}{'present':>10}{'target':>10}{'delta':>10}")
        for j, s, t, d in zip(JOINTS, start, target, deltas):
            print(f"{j:<16}{s:>10.2f}{t:>10.2f}{d:>+10.2f}")
        print(f"\nlargest move {largest:.2f} deg -> {steps} steps @ {args.rate_hz:.0f} Hz "
              f"(~{steps / args.rate_hz:.1f} s), max {args.step_deg} deg/step")

        if args.dry_run:
            print("\n--dry-run: nothing was written.")
            return

        period = 1.0 / args.rate_hz
        for i in range(1, steps + 1):
            frac = i / steps
            goal = {f"{j}.pos": s + d * frac for j, s, d in zip(JOINTS, start, deltas)}
            robot.send_action(goal)
            time.sleep(period)

        time.sleep(args.settle_s)
        after = robot.bus.sync_read("Present_Position")
        print(f"\n{'joint':<16}{'final':>10}{'target':>10}{'error':>10}")
        worst = 0.0
        for j, t in zip(JOINTS, target):
            err = after[j] - t
            worst = max(worst, abs(err))
            print(f"{j:<16}{after[j]:>10.2f}{t:>10.2f}{err:>+10.2f}")
        print(f"\nworst residual: {worst:.2f} deg")
        print("Arm is holding position (torque left enabled).")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
