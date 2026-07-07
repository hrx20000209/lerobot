# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from dataclasses import dataclass, field

from .configs import PolicyServerConfig, RobotClientConfig


@dataclass
class ContinuousPolicyServerConfig(PolicyServerConfig):
    """Configuration for the separate continuous async policy server."""

    async_mode: str = field(
        default="continuous", metadata={"help": "Async mode for this entrypoint. Must be continuous."}
    )
    continuous_inference_workers: int = field(
        default=1,
        metadata={"help": "Number of continuous inference workers. The initial implementation supports 1."},
    )
    max_pending_observations: int = field(
        default=1, metadata={"help": "Latest-only observation buffer size. Must be 1."}
    )
    timeline_log_path: str = field(
        default="outputs/continuous_async/timeline.jsonl",
        metadata={"help": "Structured JSONL timeline event log path."},
    )

    def __post_init__(self):
        super().__post_init__()
        if self.async_mode != "continuous":
            raise ValueError("ContinuousPolicyServerConfig only supports async_mode='continuous'")
        if self.continuous_inference_workers != 1:
            raise ValueError("continuous_inference_workers > 1 is not implemented yet")
        if self.max_pending_observations != 1:
            raise ValueError(
                "continuous server uses latest-only observations; max_pending_observations must be 1"
            )


@dataclass
class ContinuousRobotClientConfig(RobotClientConfig):
    """Configuration for the separate continuous async robot client."""

    async_mode: str = field(
        default="continuous", metadata={"help": "Async mode for this entrypoint. Must be continuous."}
    )
    continuous_obs_fps: float = field(default=30.0, metadata={"help": "Observation capture/send FPS."})
    aggregation_fn: str = field(
        default="latency_aligned_blend",
        metadata={
            "help": "Continuous queue aggregation: latency_aligned_blend, replace_remaining, "
            "splice_by_timestamp, smooth_blend_remaining, conservative_update"
        },
    )
    max_pending_observations: int = field(
        default=1, metadata={"help": "Latest-only client send buffer size."}
    )
    timeline_log_path: str = field(
        default="outputs/continuous_async/timeline.jsonl",
        metadata={"help": "Structured JSONL timeline event log path."},
    )
    timeline_plot_path: str = field(
        default="outputs/continuous_async/timeline.png", metadata={"help": "Timeline PNG output path."}
    )
    stale_inference_max_age: float = field(
        default=2.0, metadata={"help": "Reject inference results older than this."}
    )
    min_usable_actions: int = field(
        default=5,
        metadata={"help": "Minimum remaining actions required after latency alignment."},
    )
    blend_horizon: int = field(default=5, metadata={"help": "Actions to blend for smooth_blend_remaining."})
    blend_alpha: float = field(default=0.5, metadata={"help": "Old-action weight for smooth blending."})
    max_joint_delta: float | None = field(
        default=None, metadata={"help": "Reject conservative updates above this max action delta."}
    )
    max_gripper_delta: float | None = field(
        default=None, metadata={"help": "Reject conservative updates above this gripper delta."}
    )
    max_joint_delta_per_step: float | None = field(
        default=None, metadata={"help": "Reserved robot safety limit."}
    )
    max_joint_abs_range: float | None = field(
        default=None, metadata={"help": "Reject actions outside abs range."}
    )
    max_gripper_delta_per_step: float | None = field(
        default=None, metadata={"help": "Reserved gripper safety limit."}
    )
    max_control_steps: int | None = field(
        default=None, metadata={"help": "Stop the continuous client after this many executed control steps."}
    )
    emergency_stop: bool = field(
        default=False, metadata={"help": "Reject queue updates and action execution."}
    )
    shadow_mode: bool = field(
        default=True, metadata={"help": "Run no-op action execution without commanding motors."}
    )
    enable_robot_execution: bool = field(
        default=False, metadata={"help": "Must be true, and shadow_mode false, to send actions to motors."}
    )

    def __post_init__(self):
        super().__post_init__()
        legacy_aggregate_map = {
            "latest_only": "replace_remaining",
            "weighted_average": "smooth_blend_remaining",
            "average": "smooth_blend_remaining",
            "conservative": "conservative_update",
        }
        if self.aggregation_fn in legacy_aggregate_map:
            self.aggregation_fn = legacy_aggregate_map[self.aggregation_fn]
        if self.async_mode != "continuous":
            raise ValueError("ContinuousRobotClientConfig only supports async_mode='continuous'")
        if self.continuous_obs_fps <= 0:
            raise ValueError("continuous_obs_fps must be positive")
        if self.max_pending_observations != 1:
            raise ValueError(
                "continuous client uses latest-only observations; max_pending_observations must be 1"
            )
        if self.aggregation_fn not in {
            "latency_aligned_blend",
            "replace_remaining",
            "splice_by_timestamp",
            "smooth_blend_remaining",
            "conservative_update",
        }:
            raise ValueError(f"Unknown continuous aggregation_fn: {self.aggregation_fn}")
        if self.min_usable_actions < 0:
            raise ValueError("min_usable_actions must be non-negative")
        if self.max_control_steps is not None and self.max_control_steps <= 0:
            raise ValueError("max_control_steps must be positive when set")
