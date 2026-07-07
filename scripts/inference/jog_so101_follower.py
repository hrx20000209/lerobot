#!/usr/bin/env python
"""Small hardware jog for an SO101 follower.

This bypasses policy inference and sends one tiny absolute-position command so
we can verify that the motor command path is alive before running a policy.
"""

from __future__ import annotations

import argparse
import time

from lerobot.robots.so_follower import SOFollower, SOFollowerRobotConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyACM1")
    parser.add_argument("--id", default="follower_arm")
    parser.add_argument("--joint", default="wrist_roll")
    parser.add_argument("--delta", type=float, default=2.0)
    parser.add_argument("--hold-s", type=float, default=0.8)
    parser.add_argument("--max-relative-target", type=float, default=2.0)
    args = parser.parse_args()

    robot = SOFollower(
        SOFollowerRobotConfig(
            port=args.port,
            id=args.id,
            cameras={},
            max_relative_target=args.max_relative_target,
        )
    )
    robot.connect()
    try:
        present = robot.bus.sync_read("Present_Position")
        if args.joint not in present:
            raise ValueError(f"Unknown joint {args.joint!r}; available joints: {sorted(present)}")

        target = dict(present)
        target[args.joint] = present[args.joint] + args.delta
        action = {f"{joint}.pos": value for joint, value in target.items()}
        sent = robot.send_action(action)
        time.sleep(args.hold_s)
        after = robot.bus.sync_read("Present_Position")

        print("present:", present)
        print("sent:", sent)
        print("after:", after)
        print(f"{args.joint} delta observed: {after[args.joint] - present[args.joint]:.3f}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
