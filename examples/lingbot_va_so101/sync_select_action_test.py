#!/usr/bin/env python
"""Test LeRobot's official synchronous inference API -- select_action() -- for lingbot_va.

This is deliberately different from rollout_eval.py's teacher_forced_rollout(): that script
calls predict_action_chunk() directly and manually pokes internal state (_obs_buffer,
_executed_actions) to test chunk-level prediction quality. This script instead drives the
policy through its actual public deployment API, select_action(), one real observation per
call -- exactly how lerobot-eval / a real robot client would call it every control tick.
Checks whether select_action's own internal bookkeeping (action queue, _obs_buffer
accumulation via _keyframe_stride, _exec_step/_prev_j) works correctly end-to-end, using
real historical frames from a held-out episode as a stand-in for live observations
(open-loop / teacher-forced at the per-frame level, not per-chunk).
"""

import time

import torch

from common import ACTION_NAMES, RUNS_ROOT, action_quantiles, attach_text_embed_cache, build_config, denormalize_action, make_ds_meta
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from plotting import plot_action_compare

CHECKPOINT_DIR = "/data/rxhuang/lingbot_va_runs/full_20k/checkpoints/step_2000"
DATASET_ROOT = "/data/rxhuang/three_cubes_1"
EPISODE = 95
N_STEPS_TO_TEST = 160  # 10 chunks' worth of real control ticks


def main():
    config = build_config(attn_mode="torch")
    ds_meta = make_ds_meta()
    q01, q99 = action_quantiles(ds_meta)

    policy = make_policy(config, ds_meta=ds_meta)
    attach_text_embed_cache(policy)

    from peft import PeftModel

    policy = PeftModel.from_pretrained(policy, CHECKPOINT_DIR, is_trainable=False).get_base_model()
    policy.eval()

    raw_ds = LeRobotDataset("three_cubes_1", root=DATASET_ROOT, episodes=[EPISODE])
    cam_keys = config.obs_cam_keys
    task = raw_ds[0]["task"]

    def frame_batch(i):
        s = raw_ds[i]
        return {k: s[k].unsqueeze(0).to(config.device) for k in cam_keys} | {"task": [task]}

    policy.reset()
    print(f"n_action_steps (chunk_size) = {config.n_action_steps}, keyframe_stride = {policy._keyframe_stride}")

    pred_norm = []
    step_times = []
    frame_st_id_history = []
    obs_buffer_len_history = []
    crashed_at = None

    for t in range(N_STEPS_TO_TEST):
        t0 = time.perf_counter()
        try:
            with torch.no_grad():
                action = policy.select_action(frame_batch(t))  # [n_used], normalized
        except Exception as exc:  # noqa: BLE001
            crashed_at = (t, f"{type(exc).__name__}: {exc}")
            print(f"CRASHED at real timestep {t}: {crashed_at[1]}")
            break
        step_times.append(time.perf_counter() - t0)
        pred_norm.append(action.detach().cpu().reshape(-1))  # select_action returns [1, n_used]; drop batch dim
        frame_st_id_history.append(policy._frame_st_id)
        obs_buffer_len_history.append(len(policy._obs_buffer))

        if t % 16 == 0:
            print(
                f"t={t}: frame_st_id={policy._frame_st_id} obs_buffer_len={len(policy._obs_buffer)} "
                f"action_queue_len={len(policy._action_queue)} step_time={step_times[-1] * 1000:.1f}ms "
                f"action[:3]={action.reshape(-1)[:3].tolist()}"
            )

    if crashed_at is None:
        pred_norm = torch.stack(pred_norm)  # [T, n_used]
        gt_raw = torch.stack([raw_ds[t]["action"] for t in range(len(pred_norm))])
        pred_raw = denormalize_action(pred_norm, q01, q99)
        mse = ((pred_raw - gt_raw) ** 2).mean(dim=0)
        mean_mse = mse.mean().item()
        print(f"\nper-dim MSE: {dict(zip(ACTION_NAMES, mse.tolist(), strict=False))}")
        print(f"mean MSE: {mean_mse:.3f}")
        print(f"frame_st_id progression (every 16 steps): {frame_st_id_history[::16]}")
        print(f"mean step_time: {sum(step_times) / len(step_times) * 1000:.1f}ms, max: {max(step_times) * 1000:.1f}ms")

        out_dir = RUNS_ROOT / "full_20k" / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_path = out_dir / "select_action_sync_test_ep95.png"
        plot_action_compare(pred_raw, gt_raw, plot_path, title=f"select_action() sync test, mean_mse={mean_mse:.3f}")
        print(f"plot saved to {plot_path}")
    else:
        print(f"\nTest crashed at timestep {crashed_at[0]}: {crashed_at[1]}")


if __name__ == "__main__":
    main()
