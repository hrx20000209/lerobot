#!/usr/bin/env python
"""Closed-loop guards between the policy's action and the motor bus.

The pipeline is otherwise pure feedforward: the policy emits a target, the
target is written, and nothing ever checks whether the arm got there. Every
hardware problem seen on 2026-08-05 came from that gap. Four guards, each aimed
at one measured failure, each independently switchable so its effect can be
attributed:

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

``fixed point``  opens the gripper when motion has stopped but commanding has not.
    The other three all bound a quantity that is *too large*. This one catches
    the opposite: on 2026-08-06 the policy converged on commanding the pose the
    arm was already in, so nothing moved, so the observation never changed, so
    the next chunk was identical -- 87 s in an absorbing state, holding a cube,
    with no threshold anywhere exceeded. Releasing changes the scene, which is
    the only input that can move the policy off the fixed point.

All four are off by default. Every intervention is counted and logged so a run
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
    # Per-joint override of max_lead_deg. A single limit cannot serve all six:
    # the body joints swing over ~200 deg while the gripper's whole range is
    # 0-100, so a 50 deg limit is inert on the gripper.
    #
    # Empty by default, and the gripper in particular is left alone on purpose.
    # It looked like the obvious fix for the overload latch -- the stuck run sat
    # 8.4 deg past the cube for 87 s -- until the same measurement was taken on
    # a run that worked: base24 held its cube at 7.3 deg of squeeze and placed
    # it successfully. The magnitude is not what distinguishes the failure, the
    # duration is, so clipping the magnitude only weakens a grasp that was fine.
    # Replaying base24 confirms it: a 3 deg cap rewrites 92% of commands, 10 deg
    # still doubles the clipping. Duration is handled by ``stall`` and
    # ``fixed point`` instead. Entries here are for joints with a genuine
    # geometric limit; 0 disables the limit for that joint.
    max_lead_deg_per_joint: dict = field(default_factory=dict)
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

    # Spend this many seconds at the start of a run observing instead of acting,
    # then set the thresholds from what was observed. 0 keeps the fixed values.
    #
    # The thresholds above were read off one baseline run by hand, which does not
    # survive a change of checkpoint, fps, payload or gripper contents -- and
    # getting them wrong is not benign: a flat 95 fired on 36.9% of elbow_flex's
    # normal operation and over-constrained the policy into failing the task.
    # Deriving them from the run itself is the same idea as the guards
    # themselves: let the measurement set the parameter.
    autotune_secs: float = 0.0
    # Stall threshold = observed p99 load x this.
    autotune_stall_margin: float = 1.15
    # Lead limit = observed p99 tracking error x this, floored at 5 deg.
    autotune_lead_margin: float = 1.5

    # --- fixed point ---
    # The policy can converge on commanding the pose the arm is already in.
    # Nothing then moves, so the next observation is identical, so the next
    # chunk is identical: a closed loop with no noise has no way out. Observed
    # twice -- gain=2 run, second grasp, and again with the guards on, where
    # every joint held to within a degree for 87 s while the gripper strained
    # against the cube it was still holding.
    #
    # No stall or lead threshold catches this. Loads are not saturated (the
    # gripper read 100 against a 230 threshold) and the command is not running
    # ahead of the arm -- the command *is* where the arm is. That is the whole
    # signature: motion stops while the policy keeps issuing commands.
    #
    # Detection needs a second condition, because "not moving" is not by itself
    # an error. Requiring that the policy also be commanding the gripper shut
    # harder than it already is pins it to the case that matters: still holding
    # something, still squeezing, going nowhere.
    #
    # The escape is to open the gripper. That is not just damage control -- it
    # drops the cube, which changes the scene, which changes the observation,
    # which is the only thing that can move the policy off the fixed point.
    # "Not moving" is measured as net displacement over the window -- how far
    # the pose is from where it was W seconds ago -- not as the spread within
    # it. The distinction decides whether the test works at all: a stuck arm
    # still jitters a few degrees, so by peak-to-peak the longest stillness in
    # the stuck run was 6.1 s against 3.7 s of legitimate dwell before a place
    # in base24, which is not a margin anything can be set inside. By net
    # displacement over 8 s the two separate cleanly:
    #
    #   run                 min net displacement over any 8 s
    #   base24  (success)        4.14 deg
    #   gain2b  (success)       20.48 deg
    #   g2fb    (stuck)          1.05 deg   -- 45% of samples under 3 deg
    #
    # Hence 8 s / 3 deg, which fires 32.8 s into the stuck run and never in
    # either successful one. The arm is allowed to pause; it is not allowed to
    # end up where it started.
    # Window over which net displacement is measured, seconds. 0 disables.
    fixed_point_secs: float = 0.0
    # Net displacement below this, on every joint, counts as going nowhere.
    fixed_point_move_deg: float = 3.0
    # Also require the policy to be commanding the gripper at least this many
    # degrees tighter than measured -- evidence it is holding something.
    fixed_point_require_squeeze: bool = True
    fixed_point_squeeze_deg: float = 3.0
    # How far to open the gripper to break out, and for how long.
    fixed_point_release_deg: float = 30.0
    fixed_point_release_secs: float = 1.5


@dataclass
class FeedbackStats:
    n_commands: int = 0
    n_slew_clipped: int = 0
    n_lead_clipped: int = 0
    n_stall_frozen: int = 0
    max_slew_clip_deg: float = 0.0
    max_lead_clip_deg: float = 0.0
    stalled_joints: dict = field(default_factory=dict)
    n_fixed_point: int = 0
    n_release_commands: int = 0
    fixed_point_at: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n_commands": self.n_commands,
            "n_slew_clipped": self.n_slew_clipped,
            "n_lead_clipped": self.n_lead_clipped,
            "n_stall_frozen": self.n_stall_frozen,
            "max_slew_clip_deg": self.max_slew_clip_deg,
            "max_lead_clip_deg": self.max_lead_clip_deg,
            "stalled_joints": self.stalled_joints,
            "n_fixed_point": self.n_fixed_point,
            "n_release_commands": self.n_release_commands,
            "fixed_point_at": self.fixed_point_at,
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
        self._lead_limit = self._build_lead_limits()
        self._grip = next((i for i, j in enumerate(joints) if j.startswith("gripper")), None)
        # (time, present) samples spanning the fixed-point window
        self._fp_hist: list[tuple[float, np.ndarray]] = []
        self._fp_squeeze_since: float | None = None
        self._fp_release_until: float | None = None
        self._tune_t0: float | None = None
        self._tune_loads: list[list[float]] = []
        self._tune_errs: list[list[float]] = []
        self.autotuned: dict | None = None

    def _build_lead_limits(self) -> np.ndarray:
        return np.array(
            [self.cfg.max_lead_deg_per_joint.get(j, self.cfg.max_lead_deg) for j in self.joints],
            dtype=np.float64,
        )

    @property
    def tuning(self) -> bool:
        return self.cfg.autotune_secs > 0 and self.autotuned is None

    def _observe(self, err: np.ndarray, load: np.ndarray | None, now: float) -> None:
        if self._tune_t0 is None:
            self._tune_t0 = now
        if load is not None:
            self._tune_loads.append([abs(float(v)) for v in load])
        self._tune_errs.append([float(v) for v in err])

    def _finish_autotune(self) -> None:
        """Raise the thresholds to fit what this run showed -- never lower them.

        A startup window cannot see loads that only occur later: the gripper's
        p99 over a full run is ~216, but in the first seconds the arm has not
        grasped anything yet and it reads a fraction of that. Tuning downward on
        that evidence would set a threshold the grasp then trips constantly --
        which is precisely the failure that a flat threshold of 95 produced
        (36.9% of elbow_flex's normal operation flagged as a stall, 470 spurious
        freezes, task not completed).

        So the configured values act as a floor: observation can only widen the
        band. That keeps auto-tuning strictly safer than the priors it starts
        from, at the cost of not tightening a threshold that was set too loose.
        """
        if self._tune_loads:
            loads = np.asarray(self._tune_loads)
            p99 = np.percentile(loads, 99, axis=0)
            self._stall_thresh = [
                max(self._stall_thresh[i], float(p99[i]) * self.cfg.autotune_stall_margin)
                for i in range(len(self.joints))
            ]
        lead = self.cfg.max_lead_deg
        if self._tune_errs:
            errs = np.asarray(self._tune_errs)
            observed = float(np.percentile(errs, 99)) * self.cfg.autotune_lead_margin
            lead = max(lead, observed, 5.0)
            self.cfg.max_lead_deg = lead
        # Per-joint entries are deliberate physical limits, not priors to be
        # widened: the gripper's 3 deg exists to keep it out of overload, and a
        # startup window that never grasped anything is no evidence against it.
        # Rebuilding picks up the new default for the joints that use it and
        # leaves the explicit entries alone.
        self._lead_limit = self._build_lead_limits()
        self.autotuned = {
            "stall_thresh": {j: round(t, 1) for j, t in zip(self.joints, self._stall_thresh)},
            "max_lead_deg": round(lead, 2),
            "lead_limit": {j: round(v, 2) for j, v in zip(self.joints, self._lead_limit)},
            "samples": len(self._tune_errs),
        }
        self._tune_loads.clear()
        self._tune_errs.clear()

    def reset(self) -> None:
        self._prev_cmd = None
        self._stall_since = [None] * len(self.joints)
        self._fp_hist.clear()
        self._fp_squeeze_since = None
        self._fp_release_until = None

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

        # Observation window: watch, do not act. Passing the command through
        # unmodified is what makes the sample representative of the unguarded
        # system -- guarding while measuring would tune against its own effect.
        if self.tuning:
            self._observe(np.abs(out - present), load, now)
            if self._tune_t0 is not None and now - self._tune_t0 >= self.cfg.autotune_secs:
                self._finish_autotune()
            self._prev_cmd = out.copy()
            return out

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
        active = self._lead_limit > 0
        if active.any():
            lead = out - present
            over = active & (np.abs(lead) > self._lead_limit)
            if over.any():
                excess = float((np.abs(lead) - self._lead_limit)[over].max())
                self.stats.n_lead_clipped += 1
                self.stats.max_lead_clip_deg = max(self.stats.max_lead_clip_deg, excess)
                bound = np.where(active, self._lead_limit, np.inf)
                out = present + np.clip(lead, -bound, bound)

        # --- slew: bound the discontinuity from the previous command
        if self.cfg.max_step_deg > 0 and self._prev_cmd is not None:
            step = out - self._prev_cmd
            over = np.abs(step) > self.cfg.max_step_deg
            if over.any():
                excess = float(np.abs(step[over]).max() - self.cfg.max_step_deg)
                self.stats.n_slew_clipped += 1
                self.stats.max_slew_clip_deg = max(self.stats.max_slew_clip_deg, excess)
                out = self._prev_cmd + np.clip(step, -self.cfg.max_step_deg, self.cfg.max_step_deg)

        # --- fixed point: break the loop the other three guards cannot see.
        # Applied last and exempt from the slew limit on purpose: a 30 deg
        # opening under a 10 deg/step cap would take three commands to arrive,
        # and the point of the escape is that it be decisive.
        out = self._fixed_point(cmd, out, present, now)

        self._prev_cmd = out.copy()
        return out

    def _fixed_point(self, cmd: np.ndarray, out: np.ndarray, present: np.ndarray, now: float) -> np.ndarray:
        if self.cfg.fixed_point_secs <= 0 or self._grip is None:
            return out
        gi = self._grip

        if self._fp_release_until is not None:
            if now < self._fp_release_until:
                out[gi] = present[gi] + self.cfg.fixed_point_release_deg
                self.stats.n_release_commands += 1
                return out
            self._fp_release_until = None

        # Squeeze: the policy wants the gripper tighter than it managed to get,
        # which around a rigid object means it is leaning on a stall. Measured
        # against the policy's own request, not the guarded command, so the
        # stall guard freezing the joint does not hide the intent.
        squeezing = cmd[gi] < present[gi] - self.cfg.fixed_point_squeeze_deg
        if squeezing:
            if self._fp_squeeze_since is None:
                self._fp_squeeze_since = now
        else:
            self._fp_squeeze_since = None

        self._fp_hist.append((now, present.copy()))
        while len(self._fp_hist) >= 2 and now - self._fp_hist[1][0] >= self.cfg.fixed_point_secs:
            self._fp_hist.pop(0)
        if now - self._fp_hist[0][0] < self.cfg.fixed_point_secs:
            return out

        if np.abs(present - self._fp_hist[0][1]).max() >= self.cfg.fixed_point_move_deg:
            return out
        if self.cfg.fixed_point_require_squeeze and (
            self._fp_squeeze_since is None
            or now - self._fp_squeeze_since < self.cfg.fixed_point_secs
        ):
            return out

        self.stats.n_fixed_point += 1
        self.stats.fixed_point_at.append(round(now, 3))
        self._fp_release_until = now + self.cfg.fixed_point_release_secs
        # Clear the window so the release itself is not measured as more
        # stillness, and so the next detection needs a fresh full window.
        self._fp_hist.clear()
        self._fp_squeeze_since = None
        out[gi] = present[gi] + self.cfg.fixed_point_release_deg
        self.stats.n_release_commands += 1
        return out
