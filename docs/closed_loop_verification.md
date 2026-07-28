# Closed-loop verification at step_20000 — root cause found

Follow-up to `diagnosis.md` §5 open question: *does the closed-loop `select_action` collapse
persist at step_20000, or was it just the undertrained step_2000?*

**Answer: it persists at step_20000. It is a real autoregressive action-feedback problem, not undertraining.**

## Experiment

`examples/lingbot_va_so101/verify_closed_loop.py` — same held-out episode (95), same checkpoint
(`full_20k/step_20000`), same 320 frames, `attn_mode=torch`. Episode decoded once into RAM so every
mode is GPU-bound (the earlier 4×-video-decode was the bottleneck). Four feedback modes differ **only**
in what is written into the KV cache as the *executed-action history* between chunks; observations are
always the real episode frames.

| mode | executed-action history fed back | mean_mse (deg²) | motion amplitude |
|---|---|---|---|
| teacher_forced | ground-truth past actions | **167** | **85%** |
| state_feedback | measured proprioception `state[t+1]` | **194** | **81%** |
| self_feedback | the model's own previous predictions | 899 | 31% |
| select_action | (real deploy API — model's own predictions) | 1061 | 28% |

Plots: `outputs/eval/action_plots/step_20000/ep95_{mode}.png`.

## Interpretation

- **teacher_forced ≈ state_feedback (≈81–85% amplitude, tracks all 6 joints).** The per-chunk model is
  healthy: observation/video conditioning, action decoding, normalization, channel mapping and quantiles
  are all correct. This confirms `diagnosis.md`'s claim that the basic alignment is right.
- **self_feedback collapses to 31%, select_action to 28%.** The *only* thing changed vs state_feedback is
  that the model's own (slightly-off) predicted actions are fed back as history instead of the real ones.
  Those small errors push the next chunk's conditioning off-distribution and compound → the model stops
  committing to large motions and drifts at ~⅓ amplitude. Classic **exposure bias / autoregressive
  action-feedback drift**.
- `select_action` (28%) ≈ `self_feedback` (31%) → the collapse is driven by the action-history feedback,
  not by `select_action`'s internal obs-keyframe buffering.

Key enabling fact (already noted in `scripts/inference/diagnose_lingbot_va_actions.py`): on a
position-controlled SO-101, commanded `action[t] ≈ measured state[t+1]` (r = 0.93–0.997), so measured
proprioception is a faithful stand-in for the executed action — and a real robot always has it.

## Actionable fixes (in priority order)

1. **Inference-side, no retraining (recommended first):** at deployment feed the **measured joint state**
   back as the executed-action history instead of the model's own predictions. `state_feedback` already
   yields mse 194 / amp 81% at the *existing* step_20000 checkpoint. Implementation: the deploy/eval loop
   must overwrite `policy._executed_actions` with the proprioception observed while the previous chunk
   executed (currently `select_action` feeds back `self._executed_actions` = the model's predicted chunk).
   This is a small, localized change to the deployment loop — not a retrain.
2. **Training-side (what the prior run `selfcond_5k` was attempting):** self-conditioning — train with the
   model's own no-grad predictions as action history so it learns to be robust to its own drift. Slower;
   pursue only if the inference-side fix is insufficient (e.g. the missed final gripper-close).
3. Independently, the earlier `diagnosis.md` recommendations still stand (LoRA r≥64 + unfreeze
   `action_embedder`/`action_proj_out`) to sharpen per-chunk accuracy — but they are **not** the cause of
   the closed-loop collapse.

## Residual imperfections even in state_feedback

- The final **gripper close** (frame ~195, GT drops ~32→4) is not followed — gripper stays ~15. Worth a
  closer look (gripper channel 28 / its quantiles / task-phase timing).
- Chunk-boundary jitter and a bad first chunk (frames 0–~15), consistent with the teacher_forced plot.

---

# UPDATE — state-feedback implemented in the real `select_action()` API and verified

`examples/lingbot_va_so101/state_feedback.py` — `attach_state_feedback(policy, q01, q99)` monkeypatches
`select_action` (same instance-level style as `attach_text_embed_cache`; core lerobot untouched) to buffer
`batch["observation.state"]` in lockstep with the obs keyframes and overwrite `policy._executed_actions`
from those measured states (normalized with the training quantiles, scattered into `used_action_channel_ids`)
right before each chunk refill. The deploy loop just has to put `observation.state` in every batch.

Verified through the **actual public deployment API** (`select_action`, one real frame per tick), full
508-frame ep95, step_20000 (`verify_closed_loop.py --modes select_action select_action_statefb`):

| `select_action` mode | mean_mse (deg²) | amplitude |
|---|---|---|
| plain (model's own predictions) | 1424 | 54% |
| **+ measured-state feedback** | **85** | **91%** |

Per-joint the 5 arm joints track GT tightly across the whole episode (e.g. elbow_flex mse 3915→172,
shoulder_lift 2780→170). Runtime shape sanity confirms correct feedback tensors (`N=12→f=3`, `N=16→f=4`,
`[1,30,F,4,1]`). Plot: `outputs/eval/action_plots/step_20000/ep95_select_action_statefb.png`.

**Conclusion:** the closed-loop collapse is fixed for the arm at the existing checkpoint, no retraining —
a ~17× MSE reduction, through the real deployment path.

## Remaining task-critical issue: the gripper

Even with state feedback, the **gripper does not fully close during the grasp**: GT closes ~32→4
(frames ~195–340) but the prediction only drops to ~15 and holds, with a spurious spike near frame 330
(gripper mse 136→53, pred_std 0.8→6.2 vs gt 10.3 — much improved but still damped). On a real robot a
gripper that stops at ~15° instead of ~4° likely **fails to grasp the cube → task fails**. This is now the
prime suspect for "can't complete the task," and it is gripper-specific (the arm is essentially solved).

Next-step hypotheses to check (Stage 3/4 style, before more training):
- Per-dim **action-loss breakdown** during training — is the gripper (channel 28) loss actually dropping,
  or is it swamped by the 5 arm channels (equal-weight MSE)? If swamped, up-weight the gripper channel.
- Gripper is near-**bimodal** (open ~32 / closed ~4, a step); flow-matching may smooth the hard close.
  Check whether stronger adaptation (LoRA r≥64, unfreeze `action_embedder`/`action_proj_out`) sharpens it.
- Sanity-check gripper **quantile normalization** (q01 0.71 / q99 49.5) and channel-28 convention.

---

# UPDATE #2 — GRIPPER ROOT CAUSE: a train/deploy feedback mismatch (not capacity, not loss weight)

Ran the gripper-weighted retrain (`train_gripper_weighted.py`, run `gripper_w5_r64_heads`: gripper loss
weight 5.0, LoRA r=64 all-linear + `action_embedder`/`action_proj_out` fully trained via PEFT
`modules_to_save`, 2 LR groups) to 15k steps, then dissected the gripper with per-feedback-mode evals.

**1. The gripper physically STALLS on the cube during the grasp.** ep95, frames 195–345:
`gripper action` mean 4.9 (min 4.4) but `gripper state` mean 15.5 (min 15.0) → `|action−state| ≈ 10.6`
during grasp vs ~0.2 everywhere else. The commanded close is ~4; the motor jams at ~15 against the cube.

**2. The gripper output just tracks whatever action-history it is fed** (step_10000, gripper pred_std,
gt_std = 10.4):

| feedback mode | gripper mse | gripper pred_std | note |
|---|---|---|---|
| teacher_forced (GT action ~4) | **10.1** | **9.97** | near-perfect — model CAN close |
| state_feedback (measured state ~15) | 47 | 6.15 | under-closes to ~15 (the stall value) |
| self_feedback (own action) | 92 | 2.58 | collapses |
| hybrid (arm=state, gripper=own action) | 102 | 2.09 | collapses |

**3. Loss weighting did NOT fix it.** Gripper deploy amplitude (state_feedback pred_std) stayed ~6.15
from step_2000 → step_10000 (baseline was 6.19), even at 5× weight with fully-trained action heads. This
is expected once (2) is understood: teacher_forced already nails the gripper (mse 10), so it is **not** a
capacity/learning problem — up-weighting a loss that is already low on the wrong-feedback path can't help.

**Root cause.** Training conditions the action stream on the **ground-truth action** (~4 during grasp);
deployment can only supply the **measured state** (~15, stalled). For the 5 arm joints `action ≈ state`
(r 0.93–0.997) so there is no mismatch and state-feedback works. For the gripper the two differ by ~10
during the grasp, so the model — which leans heavily on action-history for the barely-visible gripper —
outputs the stalled ~15 and never applies grip force. It is a **train/deploy distribution mismatch on the
gripper channel**, full stop.

## Recommended fixes (needs a design decision — deferred to the user)

1. **Training-side (the principled fix):** make training match the state-feedback deployment — condition
   the action history on **measured state** (state[t+1]) while keeping the **action** as the target, so the
   model learns "given gripper-state-history ~15 (stalling), keep commanding ~4." This is what the existing
   `selfcond.py` / the `selfcond_5k` run were reaching toward. Combine with the state-feedback arm path.
2. **Deploy-side (cheap, but changes gripper control semantics):** the model gets the gripper **timing**
   right (it drops at the grasp frame and rises at release under state-feedback), only the magnitude is
   short. Threshold the gripper command on that timing → command a firm close during grasp. Common for
   near-binary grippers in IL, no retraining, but should be signed off since it overrides the learned
   gripper magnitude.

# UPDATE #3 — GRIPPER FIXED via state-conditioned training (option 1)

Implemented the training-side fix: condition the action-**history** on measured **state** while keeping
the **action** as the flow-matching target. Code: `state_cond.py` (`training_loss_state_cond` overwrites
`action_dict["latent"]` with normalized measured state after `_add_noise_stream`; targets untouched) and
`train_state_cond.py` (loads `observation.state` on the action timesteps; same gripper-weight 5 + LoRA r=64
+ fully-trained action heads). Run `state_cond_w5_r64_heads`, 15k steps.

Now training matches the state-feedback deployment: the model learns "given gripper-state-history ~15
(stalling), keep commanding ~4." Result (ep95, `select_action_statefb`, the real deployment path):

| checkpoint | overall mse | overall amp | **gripper mse** | gripper pred_std (gt 10.3) |
|---|---|---|---|---|
| baseline step_20000 (no state-cond) | 85 | 91% | 53.0 | 6.19 (stuck at ~15) |
| gripper-weighting only, step_15000 | 114 | 89% | 46.6 | 6.15 (stuck at ~15) |
| **state_cond step_5000** | 88 | 88% | 50.1 | 7.48 (closing, noisy) |
| **state_cond step_10000** | 85 | 93% | 28.0 | 7.70 (closes to ~5) |
| **state_cond step_15000 (final)** | 89 | 92% | **25.4** | **8.23** |

Gripper mse **halved (53 → 25)**, improving monotonically with steps; the arm is unchanged (state≈action).
Visually, the final gripper drops sharply to ~4–5 at the grasp transition (reaching GT), holds low (~7–11)
through the grasp with mild oscillation, and releases on time — vs the flat ~15 (no grip) of every prior
run. Plots: `outputs/eval/action_plots/state_cond_w5_r64_heads_step_{5000,10000,15000}/`.

**This is the recommended checkpoint.** Deployment: put `observation.state` in every `select_action`
batch and attach the state-feedback path (`attach_state_feedback(policy, q01, q99)`); training now matches
it for all channels including the gripper. Remaining polish (optional): mild gripper hold-phase oscillation
(~7–11 vs GT 4) — could tighten with a slightly higher gripper weight or a touch more training.

## Status of the earlier `gripper_w5_r64_heads` run (superseded)

Arm is fully solved at every checkpoint (state_feedback mse ~90, amp ~91%, matching/slightly beating the
un-weighted step_20000 baseline). The run is a good **arm** checkpoint; it does **not** by itself fix the
gripper (that needs fix #1 or #2 above). New/changed code: `train_gripper_weighted.py`,
`state_feedback.py` (+ `hybrid_gripper_channel` option), `verify_closed_loop.py` (per-run out dirs,
`select_action_hybrid` mode), and backward-compatible core edits (config `used_action_channel_weights`;
`_flow_matching_loss` weighted-mean + `_last_per_ch_action_loss`).
