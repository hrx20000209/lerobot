#!/usr/bin/env python
"""Re-anchor each arriving action chunk to the pose the arm is actually in.

The server computes a chunk as ``anchor + gain * (a_raw - anchor)``, where
``anchor`` is the proprioception captured at inference time. By the time the
chunk's first action executes, a median of 415 ms has passed and the arm has
moved on. The chunk therefore starts at a pose the arm has already left, and
executing it drags the arm backwards.

The size of that drag is ``velocity x latency``, which is why raising
ACTION_GAIN makes it worse without any latency changing. Measured over the task
window of two runs at identical fps and identical 415 ms observation age:

    metric (chunk boundaries)          gain=1     gain=2
    |delta command| p95                 4.71 deg  37.39 deg
    cosine(jump, recent motion)        -0.04      -0.64
    fraction of jumps pointing back     51%        71%

At gain=1 the boundary jump has no preferred direction. At gain=2 it points
squarely against the direction the arm was travelling, on 71% of boundaries --
the retraction that shows up on hardware as the arm hitching between chunks.

The fix is to shift the whole chunk so its first action starts where the arm
measurably is, then decay that shift to zero across the next ``blend_steps``
actions so the policy's absolute targets are still reached. The chunk's shape,
which is what the policy actually decided, is untouched.

Replaying the gain=2 trace through this:

    blend_steps    cosine    pointing back    mean |step|
    none (raw)     -0.64        71%            9.74 deg
    8              +0.53        42%            7.92 deg
    12             +0.35        35%            7.54 deg

The jump stops being a retraction and becomes a continuation, and the mean step
size falls rather than rises.

Capping the offset was tried and is worse (cosine back to -0.04 at a 20 deg
cap): a partial correction leaves part of the stale offset in place, which is
the very thing being corrected.

Note what this is not. It adds no error-proportional feedback and no lead. At
k=0 the command is moved *onto* the measurement, so |command - measured| is
strictly reduced at the boundary and never increased anywhere. It cannot excite
the lead-driven limit cycle that an execution-gap compensator can.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ReanchorConfig:
    enabled: bool = False
    # Actions over which the offset decays to zero. At fps=24 a chunk is 16
    # actions, so 8 is half a chunk (~330 ms). Shorter concentrates the whole
    # correction into fewer steps; longer leaves the arm off the policy's
    # absolute target for longer.
    blend_steps: int = 8
    # Safety stop, degrees. Not a tuning knob -- capping degrades the metric
    # this exists to improve. It only refuses corrections so large that the
    # measurement is more likely to be wrong than the chunk (a dropped frame, a
    # servo read error), in which case the chunk is used unmodified.
    reject_above_deg: float = 90.0
    # The gripper is re-anchored too by default. Its offset is small (its
    # travel is short) and leaving it out would make the gripper's timing
    # inconsistent with the arm's.
    skip_joints: tuple = ()


@dataclass
class ReanchorStats:
    n_chunks: int = 0
    n_rejected: int = 0
    n_actions_shifted: int = 0
    max_offset_deg: float = 0.0
    sum_abs_offset: float = 0.0

    def as_dict(self) -> dict:
        return {
            "n_chunks": self.n_chunks,
            "n_rejected": self.n_rejected,
            "n_actions_shifted": self.n_actions_shifted,
            "max_offset_deg": round(self.max_offset_deg, 2),
            "mean_offset_deg": round(self.sum_abs_offset / max(self.n_chunks, 1), 2),
        }


class ChunkReanchor:
    def __init__(self, joints: list[str], config: ReanchorConfig):
        self.joints = joints
        self.cfg = config
        self.stats = ReanchorStats()
        self.act = np.array([j not in config.skip_joints for j in joints], dtype=bool)
        self._src: int | None = None
        self._offset = np.zeros(len(joints))
        self._k = 0

    def reset(self) -> None:
        self._src = None
        self._offset[:] = 0.0
        self._k = 0

    def apply(self, cmd: np.ndarray, present: np.ndarray, source_timestep: int) -> np.ndarray:
        """``source_timestep`` identifies the observation the chunk came from;
        a change in it is what marks a chunk boundary."""
        if not self.cfg.enabled:
            return cmd
        if source_timestep is None:
            # No chunk identity means no way to tell a boundary from a
            # continuation. Holding the last offset would apply a stale shift
            # indefinitely, so pass the command through instead.
            return cmd
        cmd = np.asarray(cmd, dtype=np.float64)
        present = np.asarray(present, dtype=np.float64)

        if source_timestep != self._src:
            self._src = source_timestep
            self._k = 0
            off = present - cmd
            self.stats.n_chunks += 1
            if np.abs(off).max() > self.cfg.reject_above_deg:
                self.stats.n_rejected += 1
                self._offset[:] = 0.0
            else:
                off[~self.act] = 0.0
                self._offset = off
                self.stats.max_offset_deg = max(self.stats.max_offset_deg, float(np.abs(off).max()))
                self.stats.sum_abs_offset += float(np.abs(off).max())

        K = max(1, self.cfg.blend_steps)
        w = max(0.0, 1.0 - self._k / K)
        self._k += 1
        if w <= 0.0:
            return cmd
        self.stats.n_actions_shifted += 1
        return cmd + self._offset * w
