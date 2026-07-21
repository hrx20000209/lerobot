# LingBoVA on SO101 — training, validation, and deployment

This README documents the fixes/additions made to the LeRobot-native LingBoVA port for SO101
training, plus every semantic divergence found between this port and the upstream
[`Robbyant/lingbot-va`](https://github.com/Robbyant/lingbot-va) reference implementation.

## Quick start

```bash
# 1. Compute train-only-scoped normalization stats (episodes 0-94; 95-99 held out for validation)
python scripts/compute_lingbo_va_train_stats.py \
  --repo-id hrx2000/Three_Cubes_1 --root /data/rxhuang/three_cubes_1 --revision v0.1.0 \
  --train-episodes 0-94 \
  --output /data/rxhuang/three_cubes_1/lingbo_va_train_action_stats_ep0-94.json

# 2. Train (LoRA by default; see scripts/train_lingbo_va.sh --help for TRAIN_MODE=full/action_head_only)
bash scripts/train_lingbo_va.sh

# 3. Offline verification gate before touching a real robot
python scripts/eval_lingbo_va_offline.py \
  --checkpoint-dir output_lerobot_train/three_cubes/lingbo_va/checkpoints/020000/pretrained_model \
  --repo-id hrx2000/Three_Cubes_1 --root /data/rxhuang/three_cubes_1 --revision v0.1.0 \
  --episode-ids 95,97 --mode all \
  --output-dir output_lerobot_train/three_cubes/lingbo_va/offline_eval/020000

# 4a. Server-client deployment
python scripts/serve_lingbo_va.py --checkpoint-dir .../pretrained_model --port 29056
python scripts/deploy_so101_lingbo_va.py --checkpoint-dir .../pretrained_model \
  --robot-port /dev/ttyACM1 --task "..." --dry-run   # drop --dry-run once verified
```

## What was fixed (Task 1: training config)

1. **`train_action_head_only=true` + no LoRA froze the entire transformer** — only the
   action/video I/O heads ever unfroze, none of the 30 `WanTransformerBlock`s did. This is the
   root cause of the reported mean-collapse bug. Fixed by defaulting `scripts/train_lingbo_va.sh`
   to `TRAIN_MODE=lora` (rank=64, alpha=128, targets `to_q/to_k/to_v/to_out.0/ffn.net.0.proj/
   ffn.net.2` in every block + action/video heads fully trainable at a separate lr), with
   `TRAIN_MODE=full` (full fine-tune, untested on a single 24GB GPU — see script header) and
   `TRAIN_MODE=action_head_only` (legacy/debug, reproduces the original bug for A/B) as
   alternatives. A new `train_last_n_blocks` knob (mirrors upstream's `train_mode="action_last_n"`)
   is also available as a third middle-ground option, not wired into the script by default.
2. **Startup self-check**: `LingBoVAPolicy._report_trainable_params` (called once from
   `_apply_training_freeze`) logs total/trainable param counts and trainable module names, and
   **raises `RuntimeError`** if trainable% < 0.5%.
3. **`attn_mode` train/inference correctness**: `forward()` now asserts `train_attn_mode == "flex"`
   (training with `torch`/`flashattn` silently disables the model's causal training mask — a
   correctness bug, not a perf regression, since `FlexAttnFunc.init_mask` is only built inside
   `WanTransformer3DModel.forward_train`). `_ensure_local_runtime(for_training=False)`'s
   previously-dead code path is now reachable via the weight-sharing shim
   (`eval_utils.build_weight_sharing_server`), which force-overrides `attn_op` to
   `inference_attn_mode` (torch/flashattn only) regardless of what's baked into a checkpoint's
   `transformer/config.json` (every checkpoint saved by training bakes in `attn_mode="flex"`,
   matching upstream's `save_checkpoint` behavior — this requires no manual config.json editing).
4. **Train/val split + train-only-scoped norm stats**: episodes 0-94 train
   (`--dataset.episodes`), 95-99 held out (`--policy.val_episodes`, used only by the validation
   callback). `scripts/compute_lingbo_va_train_stats.py` computes q01/q99/min/max/mean/std from
   raw per-frame actions restricted to the train episodes (LeRobot's generic
   `LeRobotDatasetMetadata.stats` is always global, independent of any `episodes=` filter, so it
   cannot be reused for this). The resulting JSON is loaded via the new
   `--policy.action_stats_path` field and gets baked into every checkpoint's
   `lingbo_va_action_stats.json` automatically (`_save_pretrained`/`_save_action_stats`), so
   deploy-time `from_pretrained` always loads the exact same stats with no extra wiring.

## Task 2: training-time validation

Implemented in `eval_utils.py`, wired into `LingBoVAPolicy.run_lingbo_va_validation`, dispatched
from `lerobot_train.py` via a **duck-typed hook** (`hasattr(policy, "run_lingbo_va_validation")`)
that is a true no-op for every other policy in the repo:

- Per-step loss/grad-norm/lr with EMA(0.99) smoothing → `{output_dir}/train_metrics.csv` + wandb.
- Every `val_freq` (default 500) steps: fixed-timestep ([0.25, 0.5, 0.75]) validation loss over 32
  fixed samples with a pinned RNG seed (12345), reusing `policy.forward()`'s existing loss
  computation via a new optional `fixed_timesteps` kwarg (additive, does not affect any other
  policy).
- Same cadence: teacher-forced action-curve eval at 8 fixed offsets in episode 95 — single-shot
  conditional generation (NOT chained/autoregressive) compared to GT, producing per-joint MAE,
  direction-consistency, `std(pred)/std(GT)` ratio, and a `[MEAN COLLAPSE WARNING]` log line if
  the ratio stays below 0.2 for 3 consecutive rounds.
- Every `open_loop_freq` (default 2000) steps and on every checkpoint save: full-episode
  open-loop stride-requery eval on episodes 95 and 97 — chains the model's **own predicted
  actions** through the real KV-cache protocol (reset → infer → commit → infer → ...), matching
  true deployment semantics (camera frames come from the real recorded episode at each new
  offset; the action history conditioning the model is entirely its own prior predictions).
- All of the above write plots to `{output_dir}/eval_curves/step_{N}/` and a running
  `{output_dir}/validation_summary.md` table.

Both the training-time hook and `scripts/eval_lingbo_va_offline.py` (Task 3.3) call the exact same
`eval_utils.py` functions — no duplicated eval logic.

## Task 3: inference & deployment

- **Fixed a real, previously-undiscovered bug**: `_predict_action_chunk_server`/`_local` never
  sent `compute_kv_cache`, so `frame_st_id` never advanced server-side — every
  `predict_action_chunk` call silently "replanned from scratch" instead of streaming. Fixed via a
  new public `LingBoVAPolicy.commit_executed_action(executed_action_chunk, obs)` method that the
  caller (rollout loop) must invoke once per chunk boundary with the actually-executed
  (post safety-clip) actions.
- `scripts/serve_lingbo_va.py`: thin wrapper around the vendored `wan_va.wan_va_server.VA_Server`
  — reuses `LingBoVAPolicy._make_va_job_config()` so the server always reflects whatever the
  checkpoint was actually trained with (camera keys, channel mapping, norm stats), rather than a
  possibly-stale vendored config registry entry.
- `LingboVaKvCacheStrategy` (`src/lerobot/rollout/strategies/lingbo_va_kv_cache.py`, registered as
  `--strategy.type=lingbo_va_kv_cache`) drives the correct reset → infer → execute → commit
  protocol on top of LeRobot's real-robot deployment entrypoint (`lerobot-rollout`, **not**
  `lerobot-eval` which is simulation-only). It reuses `SOFollower`'s built-in
  `max_relative_target` for per-step delta safety clipping, and additionally clips every
  predicted action to the checkpoint's train-only-scoped `[q01, q99]` absolute range.
  `scripts/deploy_so101_lingbo_va.py` is a thin CLI that builds the equivalent
  `lerobot-rollout` command (supports `--dry-run`, `--print-only`).
- `scripts/eval_lingbo_va_offline.py`: standalone CLI running the same teacher-forced/open-loop/
  val-loss primitives against any checkpoint + episode, as the last gate before real-robot use.

## Upstream vs. port semantic divergences (for review)

1. **Channel mapping bug**: the training script used `used_action_channel_ids=[0,1,2,3,4,5]`
   (gripper at a generic joint slot); fixed to `[0,1,2,3,4,28]`, matching upstream's universal
   single-arm-gripper convention (`va_robotwin_cfg.py`, `va_demo_cfg.py`, `va_franka_cfg.py`, and
   this repo owner's own already-validated `va_three_cubes_cfg.py`).
2. **LoRA is a genuinely new capability**, not ported from upstream — upstream
   (`Robbyant/lingbot-va`) has zero LoRA code anywhere; its only training modes are `train_mode=
   "full"` (full fine-tune, lr 1e-5) or `"action_last_n"` (freeze all but heads + last N blocks).
   `train_last_n_blocks` mirrors the latter as an additive option.
3. **`cfg_prob` (CFG text dropout) was completely missing** in the port; now implemented inline
   in `_build_training_input` (upstream applies it in the dataset's `__getitem__`, since this port
   has no offline-cached-latent dataset stage — same probability-based swap-to-empty-embedding
   semantics, different call site).
4. **`video_loss_weight`/`action_loss_weight` were hardcoded to implicit 1.0/1.0**
   (`loss = latent_loss + action_loss`); now configurable. `scripts/train_lingbo_va.sh` defaults
   to `0.1/1.0` (this repo owner's own already-validated SO101 setting); the dataclass default
   stays `1.0/1.0` to match upstream's un-overridden default for other robots/configs.
5. **`attn_mode` train/inference split was structurally broken** (dead code path, see above) —
   now fixed.
6. **Norm-stat clamp bounds**: port clamps normalized actions to `[-5, 5]`; upstream clamps to
   `[-1.5, 1.5]`. Left unchanged (strictly more permissive, not implicated in the mean-collapse
   bug) — flagged here as a known, intentionally-unresolved minor divergence.
7. **`compute_kv_cache` was never invoked** — see Task 3 above; a real bug found and fixed during
   this work, not a known upstream-vs-port divergence per se.
8. **`state_key`/`observation.state` remains declared but never consumed** anywhere in
   `modeling_lingbo_va.py` (the model is purely vision+language conditioned) — pre-existing,
   out of scope here, flagged so it isn't mistaken for a new bug.

## Known limitations / not yet validated on real hardware

- No actual GPU training run has been executed as part of this change (no long-running training
  job was launched in this session) — the trainable-param self-check, loss-movement trend, and
  MAE-improvement acceptance criteria in the task description need to be verified by actually
  running `scripts/train_lingbo_va.sh` to completion (or at least a few thousand steps) and
  inspecting `train_metrics.csv` / `validation_summary.md` / `eval_curves/`.
- `LingboVaKvCacheStrategy` and `scripts/deploy_so101_lingbo_va.py` have not been exercised
  against real SO101 hardware or a live `VA_Server` — always start with `--dry-run` /
  `--print-only` per their own docstrings.
- `TRAIN_MODE=full` is explicitly documented as untested on a single 24GB GPU.
