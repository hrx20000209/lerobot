# LingBot-VA SO-101 post-training — final report (as of 2026-07-28)

Full detail: `docs/diagnosis.md` (Stage-1) and `docs/closed_loop_verification.md` (closed-loop + gripper).

## What was actually wrong (and what wasn't)

The prior assumption was "data/interface not aligned." **That was not the main problem.** Evidence:
teacher-forced eval at step_20000 already tracked ground truth in physical units — impossible if
normalization / channel mapping / quantiles / cameras / task text were wrong. The real failures were:

1. **Closed-loop action-feedback (arm) — SOLVED.** `select_action` fed the model's own predicted actions
   back as the executed-action history; small errors compounded and the rollout collapsed to ~30%
   amplitude. Feeding **measured joint state** back instead (valid: `action[t] ≈ state[t+1]`, r 0.93–0.997)
   fixes it with **no retraining**: through the real `select_action()` API on ep95, `mean_mse 1424 → 85`,
   `amplitude 54% → 91%`. Impl: `examples/lingbot_va_so101/state_feedback.py`.

2. **Gripper (task-critical) — SOLVED via state-conditioned training.** The gripper stalls on the cube
   during grasp (commanded ~4 vs measured state ~15), so the state-feedback that fixes the arm fed the
   gripper the wrong history. Root cause = **train/deploy feedback mismatch** (training conditioned the
   action-history on GT *action*; deployment supplies measured *state*). Fix (chosen option 1): condition
   the action-history on measured **state** while keeping **action** as the target, so training matches the
   state-feedback deployment. Run `state_cond_w5_r64_heads` (15k). Gripper mse **53 → 25** (halved,
   monotonic), and it now closes to ~4–5 at the grasp instead of stalling at ~15; arm unchanged
   (mse ~89, amp ~92%). Details: `closed_loop_verification.md` UPDATE #3.
   - Weighting-only (`gripper_w5_r64_heads`) did NOT work — the gripper isn't a capacity/weight problem.

## Current best checkpoint

- **`state_cond_w5_r64_heads/step_15000`** — arm mse ~89 / amp ~92% and the gripper now grasps, under the
  state-feedback deployment path. **Deployment: put `observation.state` in every `select_action` batch and
  attach `attach_state_feedback(policy, q01, q99)`; training now matches this for all channels.**

## Reproduce

```bash
CONDA=~/anaconda3/envs/lerobot/bin/python
cd ~/Projects/lerobot/examples/lingbot_va_so101

# Closed-loop eval (state-feedback = arm fix). Modes: select_action, select_action_statefb,
# select_action_hybrid, teacher_forced, state_feedback, self_feedback
CUDA_VISIBLE_DEVICES=<free_gpu> $CONDA verify_closed_loop.py \
  --checkpoint /data/rxhuang/lingbot_va_runs/full_20k/checkpoints/step_20000 \
  --episode 95 --modes select_action select_action_statefb

# Gripper-weighted retrain (arm-good; gripper needs fix (a)/(b) above)
CUDA_VISIBLE_DEVICES=<free_gpu> $CONDA train_gripper_weighted.py \
  --run_id gripper_w5_r64_heads --num_steps 15000
```

Plots: `outputs/eval/action_plots/<run>_<step>/ep95_<mode>.png`.

## Key facts / gotchas

- Env: `~/anaconda3/envs/lerobot/bin/python` (lerobot 0.5.2 editable). GPUs are 24 GB → **LoRA only**.
- Training needs `attn_mode=flex`; eval/sampling needs `attn_mode=torch` (separate process/GPU).
- Actions must be quantile-normalized externally (built-in normalize is IDENTITY) — `common.normalize_action`.
- Eval decodes the episode once into RAM (`InMemoryEpisode`) — av1 random-access decode was the bottleneck.
