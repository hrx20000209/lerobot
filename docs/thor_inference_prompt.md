# Prompt for Claude Code on `thor` — real-time SO-101 inference with fine-tuned LingBot-VA

Copy everything below the line into Claude Code on `thor`.

---

You are writing the **real-robot inference/deployment code** for a fine-tuned **LingBot-VA** policy
(`policy.type=lingbot_va`, a ~5B Wan2.2 video-action world model) controlling an **SO-101** single arm
(6 DoF: 5 arm joints + gripper). A working, *verified-correct* offline inference path already exists on the
source workstation — your job is to port it into a live robot control loop, without breaking the one
non-obvious thing that makes it work (state feedback). **Read the reference files first; do not re-derive the
call sequence from scratch.**

## What already works (source of truth — read these)
On the source machine (rsync the `lerobot` repo, editable install):
- `examples/lingbot_va_so101/verify_closed_loop.py` → `run_select_action()` — the exact, proven loop that
  drives the policy one frame per tick and produces correct actions. **Port THIS.**
- `examples/lingbot_va_so101/state_feedback.py` → `attach_state_feedback(policy, q01, q99)` — the critical
  patch. Study it.
- `examples/lingbot_va_so101/common.py` → `build_config`, `normalize_action`, `denormalize_action`,
  `attach_text_embed_cache`, `make_policy` usage.
- `src/lerobot/policies/lingbot_va/modeling_lingbot_va.py` → `select_action`, `predict_action_chunk`,
  `reset`.
- Existing deploy scaffolding to reuse for robot I/O and async: `scripts/deploy_so101_lingbo_va.py`,
  `scripts/serve_lingbo_va.py`, `examples/lingbot_va_so101/async_repro_test.py` (PolicyServer +
  `set_observation_history`).
- Background/why: `docs/closed_loop_verification.md` (esp. UPDATE #1/#2/#3) and `docs/final_report.md`.

## Assets to obtain on thor (rsync from the workstation)
1. Fine-tuned checkpoint (LoRA adapter + fully-trained action heads):
   `/data/rxhuang/lingbot_va_runs/state_cond_w5_r64_heads/checkpoints/step_15000/`  (~639 MB)
2. Base LeRobot policy `lerobot/lingbot_va_base` (5B transformer):
   `/data/hf_cache/hub/models--lerobot--lingbot_va_base/`  (~10 GB) — or re-download from HF Hub.
3. Frozen Wan VAE + UMT5 + tokenizer (this is `WAN_PRETRAINED_PATH`):
   `/home/rxhuang/Projects/models/lingbot-va-base/{vae,text_encoder,tokenizer}`  (~17 GB)
4. The `lerobot` repo (editable install; the `lingbot_va` policy code must match the checkpoint) +
   `examples/lingbot_va_so101/`.

Env: Python 3.12, torch ≥ 2.7 (cu12x), `pip install -e ".[lingbot_va]"`, plus `peft`. On Jetson Thor the
5B transformer (bf16 ~10 GB) + VAE (~2.8 GB) go on GPU; keep UMT5 (~11 GB) on CPU
(`text_encoder_device="cpu"`). Thor's 128 GB unified memory fits this.

## How to load the policy (mirror verify_closed_loop.py)
```python
config = build_config(attn_mode="torch")          # MUST be "torch" for inference; "flex" is TRAIN-ONLY and crashes select_action
# set config.wan_pretrained_path -> local frozen dir; obs_cam_keys already = [front, right, wrist]
policy = make_policy(config, ds_meta=ds_meta)      # loads base lerobot/lingbot_va_base + frozen VAE/UMT5
attach_text_embed_cache(policy)                    # UMT5 runs on CPU ~150 s; cache it (task string is constant)
from peft import PeftModel
policy = PeftModel.from_pretrained(policy, CKPT_STEP_15000, is_trainable=False).get_base_model()
policy.eval()
attach_state_feedback(policy, q01, q99)            # <-- REQUIRED. See below.
```
`ds_meta` only supplies `stats["action"]["q01"/"q99"]`; you can pass those quantiles directly (values below)
instead of shipping the dataset.

## THE CRITICAL REQUIREMENT — state feedback (do not skip)
The default `select_action` feeds the model's OWN predicted actions back into its KV cache; on this setup
that makes the arm collapse to ~30 % motion and the gripper never grips (measured + documented). The fix,
`attach_state_feedback`, instead feeds the **measured joint state** back as the executed-action history.
This was verified end-to-end and is what the checkpoint was trained to match. Therefore, in the control loop:

- Call `attach_state_feedback(policy, q01, q99)` after loading.
- Put the **raw measured joint state** (6 values, physical **degrees**, NOT normalized) in **every**
  `select_action` batch under key `"observation.state"`, shape `[1, 6]`. `attach_state_feedback` normalizes
  it internally. Omitting it silently reverts to the broken path.

## The control loop (per tick, ~30 Hz to match training fps=30)
```python
policy.reset()                                     # at the START of each episode/rollout
task = "go to red cube. take the red cube. go to box. put the red cube in box."
for t in range(...):
    batch = {
      "observation.images.front": front_img,       # float tensor [1,3,H,W] in [0,1]; any H,W (policy resizes to 128x128)
      "observation.images.right": right_img,
      "observation.images.wrist": wrist_img,
      "observation.state": joint_state_deg,         # [1,6] RAW degrees: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper].pos
      "task": [task],
    }
    action_norm = policy.select_action(batch)       # [1,6] NORMALIZED in [-1,1]
    action_deg  = denormalize_action(action_norm, q01, q99)   # -> physical joint DEGREES
    send_to_motors(action_deg)                       # 6 joint position targets (deg). Gripper too -> command the value, let it stall/grip.
```
- 6 action dims, in order: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper` (`.pos`, degrees).
  Actions are **absolute joint-position targets** (not deltas). The gripper command is a position; commanding
  the full close (~4°) even when the jaw stalls on the cube at ~15° is what applies grip force — send the
  denormalized value straight through, do NOT clamp it to the measured state.
- Cameras: order matters = `[front, right, wrist]`. front = external/head view. Images in `[0,1]`.
- `select_action` internally batches actions into chunks of 16 and maintains the KV cache; you just call it
  every tick with a fresh observation.

## Latency / async (the main real-robot challenge)
Each chunk refill runs 20 video + 50 action flow-matching denoising steps of a 5B model → **seconds**, and
plain `select_action` blocks on that every 16 ticks. For smooth 30 Hz control you must predict the next
chunk while executing the current one. Reuse the repo's async `PolicyServer` path
(`predict_action_chunk` + `set_observation_history` + `observation_history_size`), as in
`async_repro_test.py` / `serve_lingbo_va.py`.
**Important:** the `attach_state_feedback` patch currently overrides history inside `select_action`. For the
async driver you must apply the SAME override in that driver — i.e. before each `predict_action_chunk`
refill, set `policy._executed_actions` from the measured states observed while the previous chunk executed
(normalized + scattered to `used_action_channel_ids`, exactly as `state_feedback._states_to_executed` does).
If real-time is too tight, first prove correctness synchronously (below), then optimize
(fewer action denoise steps, smaller guidance, torch.compile, fp8/quant on Thor) — but re-verify after any
change to denoise steps/guidance.

## Fixed config values (from the checkpoint / training)
- `used_action_channel_ids = [14, 15, 16, 17, 18, 28]` (5 arm joints → left-arm joint channels; gripper → 28).
- `obs_cam_keys = ["observation.images.front", "observation.images.right", "observation.images.wrist"]`,
  `camera_layout="width_concat"`, `height=width=128`.
- denoising: video steps=20, action steps=50, `guidance_scale=5.0`, `action_guidance_scale=1.0`.
- fps=30, chunk_size=16, action_per_frame=4, frame_chunk_size=4.
- **q01** = `[-40.3146, -104.9285, -44.1590, 59.5478, 0.5715, 8.9597]`
- **q99** = `[  6.3816,   44.5878,  91.8237, 96.6554, 72.5608, 35.7176]`
  (order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper; degrees)
  normalize: `a_norm = 2*(a - q01)/(q99 - q01) - 1`; denormalize is the inverse.

## Verify BEFORE touching the robot
Record (or copy) one held-out episode and run `verify_closed_loop.py`
(`--modes select_action_statefb --checkpoint <step_15000>`) on thor. You should reproduce roughly:
overall mse ≈ 89, amplitude ≈ 92 %, **gripper mse ≈ 25** and the gripper dipping to ~4–5 at the grasp
(plot). If you don't, the port (state feedback, normalization, camera order, or attn_mode) is wrong — fix
that before running hardware.

## Safety
Enforce SO-101 joint limits and rate limits on the commanded degrees; wrap the gripper close with a force
cap; add an e-stop. Start at reduced speed. The policy assumes the same camera mounting/framing and the same
task language as training ("go to red cube. take the red cube. go to box. put the red cube in box.").

Deliver: a runnable `deploy_thor_so101_lingbot_va.py` (sync version first, then async), reusing lerobot's
SO-101 robot + camera classes for I/O, with the state-feedback path integrated and the pre-hardware
verification step wired in.
