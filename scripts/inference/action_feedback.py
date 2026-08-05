#!/usr/bin/env python
"""Closed-loop guards between the policy's action and the motor bus.

The pipeline is otherwise pure feedforward: the policy emits a target, the
target is written, and nothing ever checks whether the arm got there. Every
hardware problem seen on 2026-08-05 came from that gap. Three guards, each
aimed at one measured failure, each independently switchable so its effect can
be attributed:

``slew``  bounds |command_t - command_{t-1}|.
    A chunk is computed from an observation ~700 ms old, so its early steps
    describe where the arm *was*. Splicing it in front of the current command
    makes the arm retract at the chunk boundary -- visible on hardware and
    measurable as command step p95 rising from 1.5 deg (fps=24, gain=1) to
    19.0 deg (gain=2). The limit clips those boundary jumps while leaving
    ordinary motion untouched.

``lead``  bounds |command - measured|.
    Tracking error grew 6.4 -> 25.6 -> 119 deg (p95/p95/max) as fps and gain
    went up. Once the command runs far ahead of the arm it is describing a pose
    the arm will not reach, and the extra distance only shows up as motor
    current. This holds the command a bounded distance in front of where the
    arm actually is. It differs from the server's clamp, which is anchored on
    the pose at *inference* time -- by execution that pose is ~700 ms stale.

``stall`` freezes a joint whose load is saturated while its error is not closing.
    This is the one that burned hardware: the gripper held the cube, the policy
    kept commanding it ~9 deg tighter, load sat at saturation for 34 s, and the
    servo latched its overload protection. Three times. Nothing in the system
    read Present_Load.

All three are off by default. Every intervention is counted and logged so a run
can report how often each fired rather than leaving it to inference.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FeedbackConfig:
    enabled: bool = False
    # Max change in the commanded value between consecutive commands, degrees.
    # 0 disables. Ordinary motion at fps=24 sits at ~1.5 deg p95, so a few
    # degrees clips boundary discontinuities without touching normal motion.
    max_step_deg: float = 0.0
    # Max |command - measured|, degrees. 0 disables.
    max_lead_deg: float = 0.0
    # A joint is "stalled" when |load| >= its threshold and its error is not
    # shrinking. This MUST be per joint: shoulder_lift and elbow_flex hold the
    # arm against gravity, so they sit at high load during perfectly normal
    # motion. Measured on the deploy300 baseline (|load| percentiles):
    #
    #   joint           p50   p90   p99   max   fraction >= 95
    #   shoulder_pan     27    41    48    69     0.0%
    #   shoulder_lift    44   132   291   343    17.8%
    #   elbow_flex       62   196   259   307    36.9%
    #   wrist_flex       37    48    59    94     0.0%
    #   wrist_roll       30    41    55    80     0.0%
    #   gripper          37   100   216   410    26.4%
    #
    # A flat threshold of 95 therefore fires on a third of elbow_flex's normal
    # operation. The defaults below sit just above each joint's p99, so only a
    # genuine stall reaches them. Any joint missing from the map falls back to
    # stall_load.
    stall_load: float = 95.0
    stall_load_per_joint: dict = field(
        default_factory=lambda: {
            "shoulder_pan.pos": 95.0,
            "shoulder_lift.pos": 300.0,
            "elbow_flex.pos": 280.0,
            "wrist_flex.pos": 95.0,
            "wrist_roll.pos": 95.0,
            # The gripper burn happened at a sustained ~210 against a rigid cube.
            "gripper.pos": 230.0,
        }
    )
    # How long that must hold before the joint's command is frozen, seconds.
    stall_secs: float = 1.0
    # Degrees of error reduction that counts as "still making progress".
    stall_progress_deg: float = 0.5


@dataclass
class FeedbackStats:
    n_commands: int = 0
    n_slew_clipped: int = 0
    n_lead_clipped: int = 0
    n_stall_frozen: int = 0
    max_slew_clip_deg: float = 0.0
    max_lead_clip_deg: float = 0.0
    stalled_joints: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "n_commands": self.n_commands,
            "n_slew_clipped": self.n_slew_clipped,
            "n_lead_clipped": self.n_lead_clipped,
            "n_stall_frozen": self.n_stall_frozen,
            "max_slew_clip_deg": self.max_slew_clip_deg,
            "max_lead_clip_deg": self.max_lead_clip_deg,
            "stalled_joints": self.stalled_joints,
        }


class ActionFeedback:
    """Applies the guards to one command at a time, in joint order."""

    def __init__(self, joints: list[str], config: FeedbackConfig):
        self.joints = joints
        self.cfg = config
        self.stats = FeedbackStats()
        self._prev_cmd: np.ndarray | None = None
        # per-joint: when the current stall episode started, and the error then
        self._stall_since: list[float | None] = [None] * len(joints)
        self._stall_err0: list[float] = [0.0] * len(joints)
        self._stall_thresh = [
            float(config.stall_load_per_joint.get(j, config.stall_load)) for j in joints
        ]

    def reset(self) -> None:
        self._prev_cmd = None
        self._stall_since = [None] * len(self.joints)

    def apply(self, cmd: np.ndarray, present: np.ndarray, load: np.ndarray | None, now: float) -> np.ndarray:
        """Return the command actually safe to write. Order matters:

        stall first (it decides a joint must not be pushed at all), then lead
        (bound the distance to the arm), then slew (bound the jump from the last
        command). Slew last so it also smooths whatever the other two changed.
        """
        if not self.cfg.enabled:
            return cmd
        out = np.asarray(cmd, dtype=np.float64).copy()
        self.stats.n_commands += 1

        # --- stall: stop pushing a joint that is saturated and not progressing
        if load is not None:
            err = np.abs(out - present)
            for i, name in enumerate(self.joints):
                saturated = abs(float(load[i])) >= self._stall_thresh[i]
                if not saturated:
                    self._stall_since[i] = None
                    continue
                if self._stall_since[i] is None:
                    self._stall_since[i] = now
                    self._stall_err0[i] = float(err[i])
                    continue
                progressed = self._stall_err0[i] - float(err[i]) > self.cfg.stall_progress_deg
                if progressed:
                    # still closing the gap; restart the window
                    self._stall_since[i] = now
                    self._stall_err0[i] = float(err[i])
                elif now - self._stall_since[i] >= self.cfg.stall_secs:
                    # Saturated and going nowhere: hold where the joint is
                    # instead of leaning harder on it.
                    out[i] = present[i]
                    self.stats.n_stall_frozen += 1
                    self.stats.stalled_joints[name] = self.stats.stalled_joints.get(name, 0) + 1

        # --- lead: bound how far the command may run ahead of the arm
        if self.cfg.max_lead_deg > 0:
            lead = out - present
            over = np.abs(lead) > self.cfg.max_lead_deg
            if over.any():
                excess = float(np.abs(lead[over]).max() - self.cfg.max_lead_deg)
                self.stats.n_lead_clipped += 1
                self.stats.max_lead_clip_deg = max(self.stats.max_lead_clip_deg, excess)
                out = present + np.clip(lead, -self.cfg.max_lead_deg, self.cfg.max_lead_deg)

        # --- slew: bound the discontinuity from the previous command
        if self.cfg.max_step_deg > 0 and self._prev_cmd is not None:
            step = out - self._prev_cmd
            over = np.abs(step) > self.cfg.max_step_deg
            if over.any():
                excess = float(np.abs(step[over]).max() - self.cfg.max_step_deg)
                self.stats.n_slew_clipped += 1
                self.stats.max_slew_clip_deg = max(self.stats.max_slew_clip_deg, excess)
                out = self._prev_cmd + np.clip(step, -self.cfg.max_step_deg, self.cfg.max_step_deg)

        self._prev_cmd = out.copy()
        return out
