# Stage 1 — Read-only Diagnosis: LingBot-VA post-training on `three_cubes_1` (SO-101)

Date: 2026-07-27. All facts below were read from actual installed source / files, not memory.
Environment, checkpoints, dataset, policy source, and **all prior training artifacts** were inspected.

---

## 0. Environment inventory

| Item | Finding |
|---|---|
| Correct env | **`lerobot` conda env** → `~/anaconda3/envs/lerobot/bin/python` |
| lerobot | 0.5.2, **editable** install of `~/Projects/lerobot/src/lerobot` |
| torch | 2.10.0+cu128, CUDA available |
| GPUs | **8× RTX 4090 D, 24 GB each** (NOT 80 GB). Only **GPU 6** currently free (~0.4 GB, 0%); the rest are busy. |
| `flex_attention` | present in torch 2.10 (required for training `attn_mode=flex`) |
| Checkpoints (local) | `~/Projects/models/lingbot-va-base` (transformer 9.5 GB + **VAE 2.8 GB + UMT5 11 GB + tokenizer**), and `~/Projects/models/lingbot-va-posttrain-robotwin`. **No LIBERO checkpoint.** |
| Frozen Wan VAE+UMT5 | present under `lingbot-va-base/{vae,text_encoder,tokenizer}` → `_ensure_frozen_modules()` loads them via `subfolder=`. **Loads fine.** |
| Note | These `~/Projects/models/*` dirs are upstream **Wan-format** dumps (no `policy_postprocessor.json`). The LeRobot **policy** weights come from HF repo `lerobot/lingbot_va_base` (used as `pretrained_path`). |

**GPU implication:** full fine-tuning of the ~5B transformer does not fit on 24 GB (80–90 GB of optimizer state alone). **LoRA is the only realistic path** on this hardware — the §5.2 FSDP-full-shard option in the task brief does not apply here.

---

## 1. Key policy config (`configuration_lingbot_va.py`, LIBERO defaults)

- `action_dim=30`, `used_action_channel_ids` default `range(7)`.
- `action_per_frame=4`, `frame_chunk_size=4` → **`chunk_size = 16`** single-step actions per autoregressive chunk. `attn_window=30`.
- `height=width=128` (dataset is 480×640 → resized inside the policy).
- `obs_cam_keys` default `["...image", "...image2"]` (2 cams), `camera_layout="width_concat"`.
- Denoising: video `num_inference_steps=20`, `action_num_inference_steps=50`; `guidance_scale=5.0`, `action_guidance_scale=1.0`; `snr_shift=5.0`, `action_snr_shift=0.05`.
- `attn_mode` default `"torch"` (inference). **Training requires `flex`** — `training_loss_from_streams()` raises otherwise.
- **Normalization mapping is `IDENTITY` for VISUAL/STATE/ACTION.** Actions are NOT normalized by the built-in pipeline; the postprocessor unnormalizes `[-1,1]→physical` via `QUANTILES` from the checkpoint's `policy_postprocessor.json`.
- **Loss** (`_flow_matching_loss`): `loss = latent_loss + action_loss` (equal weight, no scaling); each timestep-weighted; `action_loss` is masked to `used_action_channel_ids` only.
- `get_optim_params()` = transformer params with `requires_grad` (VAE/UMT5 frozen, outside the module). With PEFT → adapter params only.
- Inference (`select_action`): first obs conditions chunk 0; subsequent obs are buffered as keyframes (`_keyframe_stride`) and, when the action queue drains, fed back into the KV cache with the **model's own executed actions** before predicting the next chunk.

**Critical pipeline gap:** `_build_training_streams()` takes `batch["action"]` and only *scatters* it into the 30-d space — **it never normalizes**. So training actions must be pre-normalized to ≈`[-1,1]` externally, or the flow-matching target scale is wrong. The prior code does exactly this (see §3).

---

## 2. Dataset `/data/rxhuang/three_cubes_1`

- `codebase_version=v3.0`, `robot_type=so_follower`, `fps=30`, **100 episodes / 51 387 frames**, **1 task**.
- Episode length min/median/max = **507 / 508 / 538** (very uniform, ~17 s each).
- `action` = **6 dims** = `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper].pos`.
- `observation.state` = same 6 dims. `mean|action − state|` per dim = `[1.0, 3.2, 4.1, 1.3, 1.4, 2.0]°` → **action is the absolute joint-position target**, leading measured state by a few frames.
- 3 cameras, all 480×640 av1: `observation.images.front`, `...right`, `...wrist`.
- Per-joint stats (degrees), from `scripts/inspect_dataset.py` → `outputs/diagnosis/action_stats.png`:

| joint | min | max | mean | std | q01 | q99 |
|---|---|---|---|---|---|---|
| shoulder_pan | -45.0 | 16.6 | -14.2 | 15.9 | -41.8 | 10.9 |
| shoulder_lift | -105.6 | 56.7 | -18.9 | 54.1 | -105.1 | 53.1 |
| elbow_flex | -63.9 | 96.3 | 15.2 | 47.9 | -60.7 | 93.6 |
| wrist_flex | 43.3 | 100.5 | 76.0 | 12.2 | 51.4 | 99.3 |
| wrist_roll | -9.2 | 95.6 | 41.2 | 26.1 | -5.1 | 79.2 |
| gripper | 0.2 | 50.9 | 23.5 | 10.9 | 0.7 | 49.5 |

- Episode curves (`outputs/diagnosis/action_curves_ep{0,50}.png`): **smooth, no jumps/NaN**; a short ~25-frame idle segment at the start of each episode (minor trim candidate).
- Task text (`meta/tasks.parquet`): **"go to red cube. take the red cube. go to box. put the red cube in box."** — non-empty, English, imperative. Good for UMT5+CFG.

---

## 3. Prior work found (this is where the "previous failures" live)

There are **two distinct prior attempts**:

**(A) Cosmos/wan_va side (different codebase).** Logs in `three_cubes_1/action_focused_train_logs/` are emitted by `cosmos_policy/...` — a *separate* repo (matches memory note "Cosmos Policy collapse caveat is a different codebase"). Channel map `[0,1,2,3,4,28]` (arm joints → **left-arm EEF** channels) per `so101_front_wrist_lingbot_norm_stats.json`. Sweep A–F over `full_dit`/`partial8`/`partial16`, `action` vs `joint`, `x10` loss weight; reached total loss ~0.025 at 10k. **front+wrist** cameras; pre-extracted latents in `three_cubes_1/latents/` are from here.

**(B) LeRobot-native `lingbot_va` (current, this repo).** `examples/lingbot_va_so101/` + runs in `/data/rxhuang/lingbot_va_runs/{smoke_500, full_20k, selfcond_5k}`.
- `common.py`: channel map **`[14,15,16,17,18,28]`** (arm joints → **left-arm JOINT** channels 14–18 + left gripper 28); `obs_cam_keys=[front, right, wrist]` (**3 cams**); `camera_layout=width_concat`; base = HF `lerobot/lingbot_va_base`; Wan path = local base dir.
- Actions ARE quantile-normalized manually: `batch["action"] = normalize_action(a, q01, q99)` with `q01/q99` from `ds_meta.stats["action"]` → `2*(a-q01)/(q99-q01) - 1`. This closes the "IDENTITY normalize" gap noted in §1.
- LoRA via `wrap_with_peft(target_modules="all-linear", r=16)`. **`action_embedder`/`action_proj_out` are NOT separately unfrozen** (task §5.1 recommends they should be), and **r=16** (task recommends 64–128).
- `full_20k`: `action_loss` fell from ~0.066 to ~0.01–0.08. Teacher-forced eval plot (`action_compare_step20000_ep95.png`, real obs + **GT** past actions) **tracks GT well, mean_mse≈201 deg²**.
- The only `select_action()` closed-loop test on record (`select_action_sync_test_ep95.png`) **collapses to nearly flat, mse≈1184** — but it loaded **`step_2000`**, not `step_20000`, so it conflates under-training with feedback mode.
- `selfcond.py` + `selfcond_5k` run: an attempt to train with the model's own predictions as action-history conditioning → i.e. they had already identified the **train/deploy action-history distribution gap** as the suspected culprit.

---

## 4. Answers to the 12 diagnosis questions

1. **Joint or EEF?** → **Joint space** — 6 raw joint positions in degrees. (Feature names + `action≈state`.)
2. **Delta or absolute; rotation/units?** → **Absolute joint targets, degrees.** Not delta. (Rotation representation N/A — no EEF pose.)
3. **Gripper range / convention?** → Continuous open width ~**0.2–50.9** (deg-like); q01/q99 `[0.71, 49.48]` → min-max normalized to `[-1,1]`. Not 0/1 or ±1. Self-consistent because we fine-tune from **base** with our own quantiles into channel 28 — we do NOT inherit a released gripper convention.
4. **Map to 30-d layout?** → Current (B): `[14,15,16,17,18]` = 5 arm joints into left-arm **joint** channels + `28` = gripper. (Earlier (A) used EEF channels `[0–4]`.) Uses 5 of the 7 left-arm joint slots (SO-101 has exactly 5 arm DOF) — consistent.
5. **Camera count/order?** → cam0=`front` (external/head ✓). **RISK:** `common.py` passes **3 cams `[front, right, wrist]`**; `right` is a 2nd *external* view, not a wrist cam, so "cam0 head, rest wrist" is violated. Pre-extracted latents + the norm-stats filename use only **front+wrist (2)**. Base LIBERO default is **2** cams width-concatenated → **3 cams changes latent width vs the pretrained layout.** ⚠️ **Must verify what the base checkpoint expects and standardize on 2 (front+wrist).**
6. **Resolution / layout?** → Dataset 480×640 → resized to **128×128** inside policy; `width_concat`. OK once cam count is fixed.
7. **fps match?** → Dataset **30 fps**. Pretrained (LIBERO) fps **not yet verified** ⚠️ — if different, per-step physical displacement differs and `action_per_frame`/keyframe stride semantics shift. Verify before trusting closed-loop timing.
8. **Can it supply 16 actions + per-cam windows?** → **Yes** (episodes ~508 frames; `delta_timestamps` in `common.py` handle it).
9. **Language instruction?** → "go to red cube. take the red cube. go to box. put the red cube in box." ✓ non-empty, English, imperative.
10. **Which checkpoint to start from?** → **base** (`lerobot/lingbot_va_base`) — correct for a **joint-channel** fine-tune; libero/robotwin are EEF-pose-specialized on channels we don't use. Agree with the existing choice.
11. **Enough data?** → 100 eps (95 train / 5 held-out `[95–99]`), single task, uniform length — reasonable for **LoRA**. Cube-position diversity not yet quantified; no obviously-bad trajectories seen in ep0/ep50.
12. **Previous failed config?** → See §3. Two attempts; the current one **does normalize correctly and the action head learns** (teacher-forced tracks). Open failure = closed-loop `select_action`, tested only at step_2000, plus possibly under-powered LoRA.

---

## 5. Refined hypothesis (differs from the brief's default)

The task brief's default assumption is "data/interface misalignment." **The evidence partly contradicts this:** at step 20000, the teacher-forced action-compare plot tracks ground truth in physical units across all 6 joints (mse≈201 deg²). That is only possible if action normalization, channel mapping, quantiles, task text, and camera encoding are **basically correct** — a real alignment bug would corrupt the teacher-forced path too.

The most likely remaining root causes, in priority order:

1. **Closed-loop / action-history distribution gap** — training conditions on *ground-truth* past actions; deployment (`select_action`) conditions on the *model's own* predictions + fed-back keyframes. The only closed-loop test collapsed, but at step_2000. **Needs a clean apples-to-apples re-test at step_20000: `open_loop` vs `teacher_forcing`** (exactly the §6 deliverable). This is the #1 thing to nail down.
2. **Camera-count mismatch (3 vs 2)** — `[front,right,wrist]` vs the base model's likely 2-cam width_concat layout. Cheap to fix; could distort the conditioning latent.
3. **Under-powered adaptation** — LoRA `r=16`, `all-linear`, with `action_embedder`/`action_proj_out` left frozen. Task §5.1 recommends r=64–128 + unfreezing those two modules.
4. **fps assumption unverified** — pretrained vs 30 fps.

## 6. Recommended next steps (pending your OK — Stage 1 stops here per brief §8)

- **Verify base checkpoint camera count / expected `obs_cam_keys`** and pretrained fps (read `lerobot/lingbot_va_base` config).
- Before any new training, run the §4 sanity checks — especially the **clean step_20000 `select_action` (open_loop) vs teacher_forcing** comparison — to localize the failure to the closed-loop path (or not).
- Standardize on **front+wrist (2 cams)** unless the base expects 3.
- Then a LoRA run with **r≥64 + unfrozen `action_embedder`/`action_proj_out`**, reusing the (correct) normalization.

**Not yet verified (flagged honestly):** base-model camera count, pretrained fps, and whether the closed-loop collapse persists at step_20000.
