#!/usr/bin/env python
"""Offline replay verification for a LingBoVA checkpoint — the last gate before deploying to a
real robot.

Runs the SAME primitives as the training-time validation callback
(`LingBoVAPolicy.run_lingbo_va_validation`, see `src/lerobot/policies/lingbo_va/eval_utils.py`)
against a saved checkpoint and an arbitrary episode:
  - teacher-forced: single-shot conditional generation at fixed offsets, compared to GT
  - open-loop: full-episode stride-requery replay chaining the model's own predictions through
    the real KV-cache protocol (reset -> infer -> commit -> infer -> ...)

Outputs per-joint MAE / direction-consistency / std-ratio plots and a markdown summary, matching
the training-time callback's file layout, so results are directly comparable across checkpoints.

Usage:
    python scripts/eval_lingbo_va_offline.py \\
        --checkpoint-dir output_lerobot_train/three_cubes/lingbo_va/checkpoints/020000/pretrained_model \\
        --repo-id hrx2000/Three_Cubes_1 --root /data/rxhuang/three_cubes_1 --revision v0.1.0 \\
        --episode-ids 95,97 --mode all \\
        --output-dir output_lerobot_train/three_cubes/lingbo_va/offline_eval/020000
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline teacher-forced + open-loop replay verification for a LingBoVA checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", default=None)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--episode-ids", required=True, help="Comma-separated episode indices, e.g. 95,97")
    parser.add_argument(
        "--mode", default="all", choices=["teacher_forced", "open_loop", "val_loss", "all"]
    )
    parser.add_argument(
        "--teacher-forced-offsets",
        default="0,50,100,150,200,250,300,350",
        help="Comma-separated frame offsets for teacher-forced eval.",
    )
    parser.add_argument("--open-loop-stride", type=int, default=8)
    parser.add_argument("--val-num-samples", type=int, default=32)
    parser.add_argument("--val-seed", type=int, default=12345)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--step", type=int, default=0, help="Label used in output filenames/plots.")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from lerobot.policies.lingbo_va import eval_utils
    from lerobot.policies.lingbo_va.modeling_lingbo_va import LingBoVAPolicy

    episode_ids = [int(e) for e in args.episode_ids.split(",") if e.strip()]
    offsets = [int(o) for o in args.teacher_forced_offsets.split(",") if o.strip()]

    policy = LingBoVAPolicy.from_pretrained(args.checkpoint_dir)
    policy.config.device = args.device
    policy._ensure_local_runtime(for_training=True)  # loads transformer/vae/text_encoder in-process

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}

    if args.mode in ("val_loss", "all"):
        val_metrics = eval_utils.run_fixed_timestep_validation(
            policy,
            args.repo_id,
            args.root,
            args.revision,
            episode_ids,
            num_samples=args.val_num_samples,
            seed=args.val_seed,
        )
        results.update(val_metrics)
        print("Fixed-timestep val loss:", json.dumps(val_metrics, indent=2))

    if args.mode in ("teacher_forced", "all"):
        for episode_id in episode_ids:
            tf_metrics = eval_utils.run_teacher_forced_eval(
                policy,
                args.repo_id,
                args.root,
                args.revision,
                episode_id=episode_id,
                offsets=offsets,
                output_dir=args.output_dir,
                step=args.step,
            )
            results[f"teacher_forced_ep{episode_id}"] = tf_metrics
            print(f"Teacher-forced ep{episode_id}:", json.dumps(tf_metrics, indent=2, default=str))

    if args.mode in ("open_loop", "all"):
        for episode_id in episode_ids:
            ol_metrics = eval_utils.run_open_loop_episode_eval(
                policy,
                args.repo_id,
                args.root,
                args.revision,
                episode_id=episode_id,
                stride=args.open_loop_stride,
                output_dir=args.output_dir,
                step=args.step,
            )
            results[f"open_loop_ep{episode_id}"] = ol_metrics
            print(f"Open-loop ep{episode_id}:", json.dumps(ol_metrics, indent=2, default=str))

    summary_path = args.output_dir / "offline_eval_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
