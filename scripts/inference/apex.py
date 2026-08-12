#!/usr/bin/env python
"""APEX: an adaptive execution layer between the policy and the motor bus.

Implements Zhao, Jiang, An, Jia and Yang, "APEX: Adaptive Policy Execution for
Precise Manipulation" (arXiv:2606.16504), Eqs. (5), (7) and (8).

The premise matches this deployment exactly. Cosmos emits joint-position
targets; the Feetech servo runs its own position loop whose gains we cannot
read or set. The policy supplies no velocity or acceleration reference, so that
loop is asked to track a signal it has no feed-forward for, and it lags. The
gap is measurable here: at gain=2 the command ran a median 4.8 deg and a p95 of
40 deg ahead of the arm, which means the trajectory the arm actually flew was
not the one the policy planned, and the gripper opened and closed somewhere
other than where it was supposed to.

This is the opposite of the clamp-style guards in ``action_feedback.py``. Those
bound the command to keep the hardware safe, which by construction cannot make
the arm track better -- clipping a command the arm was already failing to reach
only moves the target closer to where it already is. APEX instead reconstructs
the missing higher-order reference and *adds* lead, with the coefficients that
would require knowing K_p, K_d and J learned online from the tracking error.

Per joint, at each control step:

    e_a = a_pi - a                                    tracking error
    y  += (-K1 * (y + e_a)) * dt                      filter state, Eq. (5)
    add = K1 * K2 * (y + e_a)                         reference acceleration
    ad  = -K2 * y                                     reference velocity
    ab += ad * dt                                     reference position
    s   = a_dot - ad                                  velocity error
    w  += -Gw  * s * (J * add + alpha * e_a) * dt     Eq. (8)
    wd += -Gwd * s * ad * dt
    J  += -GJ  * s * add * dt
    a_r = ab + w * (J * add + alpha * e_a) + wd * ad   Eq. (7)

Two departures from the paper, both because this runs on hardware that has
already latched its overload protection three times in one day:

* a warm-up window during which the raw policy command passes through. At t=0
  the adaptive weights are at their priors and ``ab`` has not yet caught up to
  ``a_pi``, so the layer's own output is worse than doing nothing.
* a cap on |a_r - a_pi|. Adaptation laws of this form are only guaranteed
  bounded under persistent excitation, and a stationary arm holding a cube
  provides none. The cap is a bound on how wrong the layer is allowed to be,
  not a tracking limit -- it is applied to the correction, never to the policy.
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ApexConfig:
    enabled: bool = False
    # Reconstruction filter gains, Eq. (5). K1 sets how fast the filter state
    # chases the tracking error, K2 how fast the reference position follows.
    # Both are in 1/s; at fps=24 the control step is 42 ms, so gains far above
    # ~40 integrate a step larger than the signal and ring.
    k1: float = 12.0
    k2: float = 12.0
    # Feedback gain anchoring the reconstructed reference to the policy's, Eq. (6).
    alpha: float = 3.0
    # Learning rates for w, w_d and J, Eq. (8).
    gamma_w: float = 2.0e-4
    gamma_wd: float = 2.0e-4
    gamma_j: float = 2.0e-5
    # Priors. w plays the role of 1/K_p, so a small positive value starts the
    # layer with a little lead rather than none and lets adaptation raise it.
    w0: float = 0.004
    wd0: float = 0.0
    j0: float = 1.0
    # Pass the policy command through unchanged for this long at the start.
    warmup_secs: float = 1.5
    # Then ramp the correction in over this long, so the first corrected
    # command is not a step change on a moving arm.
    ramp_secs: float = 1.0
    # Hard bound on |a_r - a_pi|, degrees.
    max_correction_deg: float = 12.0
    # Bound on w, w_d and J. These are *projections* onto the physically
    # meaningful set, not just safety rails: w stands for 1/K_p, w_d for
    # K_d/K_p and J for an inertia, so all three are positive by construction.
    # Eq. (8) as written has nothing keeping them there, and unconstrained it
    # drives J to -20 on this data -- a negative inertia, which inverts the
    # sign of the acceleration feed-forward and turns the correction into
    # sustained chatter.
    max_w: float = 0.04
    max_wd: float = 0.04
    max_j: float = 20.0
    min_w: float = 0.0
    min_wd: float = 0.0
    min_j: float = 0.05
    # Adaptation needs persistent excitation to converge, and this task spends
    # much of its time with the arm parked. Below this tracking error there is
    # nothing to learn from, so the estimates are held rather than integrated
    # against noise.
    dead_zone_deg: float = 0.5
    # sigma-modification: leak the estimates back toward their priors at this
    # rate (1/s). Bounds the drift that the dead zone does not catch.
    leak: float = 0.05
    # Rate limit on the correction itself, deg/s. The magnitude cap alone still
    # permits a jump from +cap to -cap between two 42 ms steps.
    max_correction_rate_deg_s: float = 40.0
    # Anti-windup on the reconstructed reference. Eq. (5) makes ``ab`` a pure
    # integrator of the tracking error: at steady state y = -e_a, so ab ramps
    # for as long as e_a is non-zero. The paper's loop stops the ramp by
    # driving a to a_pi, but any persistent steady-state error -- gravity droop
    # on a geared servo, stiction, or the correction cap itself -- leaves e_a
    # pinned away from zero and ab winds up without bound. Unbounded, that is
    # not a subtle degradation: on hardware it pinned the correction at its cap
    # on 98% of steps, which turned an adaptive layer into a constant 6 deg
    # bias on every joint and made the arm miss the cube entirely. Holding the
    # reference within a bounded distance of the policy command is what keeps
    # the correction adaptive instead of saturated.
    # 1 deg, not a loose safety bound. Decomposing the correction showed that
    # at 8 deg the reference offset |ab - a_pi| supplied ~6 deg of it while all
    # three adaptive terms together contributed under 0.3 deg -- the layer was
    # pure windup wearing APEX's name. Holding ab within a degree of a_pi makes
    # it the smooth reconstruction the paper intends and leaves the correction
    # to the feed-forward terms, which is the only configuration in which any
    # of this is actually adaptive.
    max_ref_gap_deg: float = 1.0
    # Joints the layer acts on. The gripper is excluded by default: its travel
    # ends against a rigid object rather than at a setpoint, so the "tracking
    # error" that APEX is built to cancel is, for the gripper, the object. The
    # layer would read a blocked finger as a lag and lean harder on it, which
    # is exactly how the servo latched before.
    skip_joints: tuple = ("gripper.pos",)
    # Smoothing on the finite-differenced joint velocity, in [0, 1). The
    # measured position is quantised, so raw differences at 24 Hz are noisy and
    # s would be mostly noise.
    vel_smooth: float = 0.6


@dataclass
class ApexStats:
    n_steps: int = 0
    n_corrected: int = 0
    n_capped: int = 0
    max_correction_deg: float = 0.0
    sum_abs_correction: float = 0.0
    sum_abs_err_before: float = 0.0
    final_w: list = field(default_factory=list)
    final_wd: list = field(default_factory=list)
    final_j: list = field(default_factory=list)

    def as_dict(self) -> dict:
        n = max(self.n_corrected, 1)
        return {
            "n_steps": self.n_steps,
            "n_corrected": self.n_corrected,
            "n_capped": self.n_capped,
            "max_correction_deg": round(self.max_correction_deg, 3),
            "mean_abs_correction_deg": round(self.sum_abs_correction / n, 3),
            "mean_abs_tracking_err_deg": round(self.sum_abs_err_before / max(self.n_steps, 1), 3),
            "final_w": [round(v, 5) for v in self.final_w],
            "final_wd": [round(v, 5) for v in self.final_wd],
            "final_j": [round(v, 3) for v in self.final_j],
        }


class Apex:
    """One instance per run; ``step`` is called once per executed action."""

    def __init__(self, joints: list[str], config: ApexConfig):
        self.joints = joints
        self.cfg = config
        self.stats = ApexStats()
        n = len(joints)
        self.act = np.array([j not in config.skip_joints for j in joints], dtype=bool)

        self._y = np.zeros(n)
        self._ab: np.ndarray | None = None   # reconstructed reference position
        self._ad = np.zeros(n)               # reconstructed reference velocity
        self._w = np.full(n, config.w0)
        self._wd = np.full(n, config.wd0)
        self._j = np.full(n, config.j0)

        self._t0: float | None = None
        self._prev_t: float | None = None
        self._prev_a: np.ndarray | None = None
        self._vel = np.zeros(n)
        self._prev_corr = np.zeros(n)

    def reset(self) -> None:
        self._ab = None
        self._prev_t = None
        self._prev_a = None
        self._y[:] = 0.0
        self._ad[:] = 0.0
        self._vel[:] = 0.0
        self._prev_corr[:] = 0.0

    def _blend(self, now: float) -> float:
        """0 during warm-up, ramping to 1 over ramp_secs."""
        if self._t0 is None:
            return 0.0
        el = now - self._t0 - self.cfg.warmup_secs
        if el <= 0:
            return 0.0
        if self.cfg.ramp_secs <= 0:
            return 1.0
        return float(min(1.0, el / self.cfg.ramp_secs))

    def step(self, a_pi: np.ndarray, a: np.ndarray, now: float) -> np.ndarray:
        """Return the command to write, given the policy's and the measurement."""
        if not self.cfg.enabled:
            return a_pi
        a_pi = np.asarray(a_pi, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)
        c = self.cfg

        if self._t0 is None:
            self._t0 = now
        if self._ab is None:
            # Start the reference on the arm, not on the policy: the first
            # command of a run can be far from the current pose, and seeding ab
            # there would inject that whole jump into the reference.
            self._ab = a.copy()
        if self._prev_t is None or now <= self._prev_t:
            self._prev_t, self._prev_a = now, a.copy()
            self.stats.n_steps += 1
            return a_pi
        dt = min(now - self._prev_t, 0.25)  # a stalled queue must not integrate a huge step

        # measured joint velocity, smoothed
        raw_v = (a - self._prev_a) / dt
        self._vel = c.vel_smooth * self._vel + (1.0 - c.vel_smooth) * raw_v
        self._prev_t, self._prev_a = now, a.copy()

        e_a = a_pi - a
        self.stats.n_steps += 1
        self.stats.sum_abs_err_before += float(np.abs(e_a[self.act]).mean()) if self.act.any() else 0.0

        # --- Eq. (5): adaptive passive filter reconstructing a_bar and derivatives
        self._y += (-c.k1 * (self._y + e_a)) * dt
        add = c.k1 * c.k2 * (self._y + e_a)      # reference acceleration
        self._ad = -c.k2 * self._y               # reference velocity
        self._ab = self._ab + self._ad * dt      # reference position
        if c.max_ref_gap_deg > 0:
            g = c.max_ref_gap_deg
            clamped = a_pi + np.clip(self._ab - a_pi, -g, g)
            # Back off the filter state on the joints that hit the bound, so it
            # stops integrating in the direction it cannot go -- clamping ab
            # alone would leave y winding up behind it.
            stuck = clamped != self._ab
            self._y[stuck] *= 0.5
            self._ab = clamped

        # --- Eq. (8): test-time adaptation of w, w_d, J from the velocity error,
        # gated by a dead zone and pulled back toward the priors by a leak.
        s = self._vel - self._ad
        regressor = self._j * add + c.alpha * e_a
        excited = np.abs(e_a) >= c.dead_zone_deg
        gate = np.where(excited, 1.0, 0.0)
        self._w += (gate * -c.gamma_w * s * regressor - c.leak * (self._w - c.w0)) * dt
        self._wd += (gate * -c.gamma_wd * s * self._ad - c.leak * (self._wd - c.wd0)) * dt
        self._j += (gate * -c.gamma_j * s * add - c.leak * (self._j - c.j0)) * dt
        np.clip(self._w, c.min_w, c.max_w, out=self._w)
        np.clip(self._wd, c.min_wd, c.max_wd, out=self._wd)
        np.clip(self._j, c.min_j, c.max_j, out=self._j)

        # --- Eq. (7): corrected command
        a_r = self._ab + self._w * (self._j * add + c.alpha * e_a) + self._wd * self._ad

        # Express the result as a correction on the policy command, so warm-up,
        # ramping and the cap all act on the thing this layer is responsible for
        # and never on the policy's intent itself.
        corr = (a_r - a_pi) * self._blend(now)
        corr[~self.act] = 0.0
        capped = np.abs(corr) > c.max_correction_deg
        if capped.any():
            self.stats.n_capped += 1
            corr = np.clip(corr, -c.max_correction_deg, c.max_correction_deg)
        if c.max_correction_rate_deg_s > 0:
            room = c.max_correction_rate_deg_s * dt
            corr = self._prev_corr + np.clip(corr - self._prev_corr, -room, room)
        self._prev_corr = corr.copy()

        m = float(np.abs(corr).max())
        if m > 0:
            self.stats.n_corrected += 1
            self.stats.sum_abs_correction += float(np.abs(corr[self.act]).mean())
            self.stats.max_correction_deg = max(self.stats.max_correction_deg, m)
        return a_pi + corr

    def finalize(self) -> None:
        self.stats.final_w = [float(v) for v in self._w]
        self.stats.final_wd = [float(v) for v in self._wd]
        self.stats.final_j = [float(v) for v in self._j]
