#!/usr/bin/env python
"""Compute train-only-scoped action normalization stats for LingBoVA.

`LeRobotDatasetMetadata.stats` (meta/stats.json) is always computed over the FULL dataset,
independent of any `--dataset.episodes` filter used at training time. Training must never let
normalization statistics leak information from held-out validation episodes, so this script
recomputes q01/q99 (and min/max/mean/std) from raw per-frame actions restricted to an explicit
list of train episode indices, and writes them in the same JSON schema
`LingBoVAPolicy._save_action_stats`/`_load_action_stats` already use (`{"q01": [...], "q99":
[...], "min": [...], ...}`, one float per dataset action dimension).

Point `--policy.action_stats_path` at the resulting file when training (see
`scripts/train_lingbo_va.sh`). Every checkpoint's `_save_pretrained` call re-persists whatever
stats are loaded into `lingbo_va_action_stats.json` alongside the weights, so deploy-time
`from_pretrained` automatically picks up the same train-only-scoped stats with no extra wiring.

Example:
    python scripts/compute_lingbo_va_train_stats.py \\
        --repo-id hrx2000/Three_Cubes_1 \\
        --root /data/rxhuang/three_cubes_1 \\
        --revision v0.1.0 \\
        --train-episodes 0-94 \\
        --output /data/rxhuang/three_cubes_1/lingbo_va_train_action_stats_ep0-94.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_episode_range(spec: str) -> list[int]:
    """Parses "0-94" or "0,1,2,95" or a mix "0-10,20,30-40" into a sorted list of ints."""
    episodes: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-")
            episodes.update(range(int(start), int(end) + 1))
        else:
            episodes.add(int(part))
    return sorted(episodes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute train-only-scoped q01/q99 action normalization stats for LingBoVA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--repo-id", required=True, help="LeRobot dataset repo_id, e.g. hrx2000/Three_Cubes_1")
    parser.add_argument("--root", default=None, help="Local dataset root, e.g. /data/rxhuang/three_cubes_1")
    parser.add_argument("--revision", default=None, help="Dataset revision/tag, e.g. v0.1.0")
    parser.add_argument(
        "--train-episodes",
        required=True,
        help='Train episode indices, e.g. "0-94" or "0,1,2,...". Must exclude every validation episode.',
    )
    parser.add_argument("--output", required=True, type=Path, help="Output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from lerobot.policies.lingbo_va.eval_utils import compute_train_only_action_stats

    train_episodes = parse_episode_range(args.train_episodes)
    print(f"Computing action stats over {len(train_episodes)} train episodes: {train_episodes[:5]}...")
    stats = compute_train_only_action_stats(
        repo_id=args.repo_id, root=args.root, revision=args.revision, train_episodes=train_episodes
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"q01={stats['q01']}")
    print(f"q99={stats['q99']}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
