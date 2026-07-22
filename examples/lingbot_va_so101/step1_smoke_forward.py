#!/usr/bin/env python
"""Step 1 smoke test: prove the three_cubes_1 dataset can feed lingbot_va's forward().

Not a training run -- just isolates the data-interface question (shapes align, no
NaN, loss computes) from the training-quality question. Uses a randomly-initialized
transformer (the frozen VAE/UMT5 still load from the local lingbot-va-base dir) so it
does not require pulling the ~10GB LeRobot-format checkpoint from the Hub first.
"""
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy
from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig

DATASET_ROOT = "/data/rxhuang/three_cubes_1"
WAN_PRETRAINED_PATH = "/home/rxhuang/Projects/models/lingbot-va-base"

# 5 arm joints + gripper, in the dataset's own column order (shoulder_pan, shoulder_lift,
# elbow_flex, wrist_flex, wrist_roll, gripper) -> channels 14-18 ("left-arm joints,
# unused by released checkpoints") + 28 ("left gripper"). Joint-space fine-tune of the
# base (non-EEF-specialized) checkpoint, per user's explicit choice over FK-to-EEF.
USED_ACTION_CHANNEL_IDS = [14, 15, 16, 17, 18, 28]
OBS_CAM_KEYS = ["observation.images.front", "observation.images.right", "observation.images.wrist"]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    config = LingBotVAConfig(
        obs_cam_keys=OBS_CAM_KEYS,
        camera_layout="width_concat",
        used_action_channel_ids=USED_ACTION_CHANNEL_IDS,
        attn_mode="flex",  # forward()/training_loss_from_streams require this
        wan_pretrained_path=WAN_PRETRAINED_PATH,
        text_encoder_device="cpu",
        device=device,
    )

    # Build delta_timestamps from the config's own delta-index properties (mirrors how
    # lerobot_train.py would wire n_obs_steps/chunk history for any policy).
    fps = 30
    obs_idx = config.observation_delta_indices
    act_idx = config.action_delta_indices
    print(f"observation_delta_indices ({len(obs_idx)}): {obs_idx}")
    print(f"action_delta_indices ({len(act_idx)}): {act_idx}")

    delta_timestamps = {k: [i / fps for i in obs_idx] for k in OBS_CAM_KEYS}
    delta_timestamps["action"] = [i / fps for i in act_idx]

    dataset = LeRobotDataset(
        "three_cubes_1",
        root=DATASET_ROOT,
        delta_timestamps=delta_timestamps,
    )
    print(f"dataset: {dataset.meta.total_episodes} episodes, {dataset.meta.total_frames} frames")

    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    for k in OBS_CAM_KEYS + ["action"]:
        print(f"batch[{k!r}].shape = {tuple(batch[k].shape)}, dtype={batch[k].dtype}")
    print(f"batch['task'] = {batch['task']}")

    # Mirrors lerobot_train.py: derive input/output_features + dataset_stats from ds_meta
    # (populates cfg.input_features/output_features, required by validate_features()).
    ds_meta = LeRobotDatasetMetadata("three_cubes_1", root=DATASET_ROOT)
    policy = make_policy(config, ds_meta=ds_meta)
    policy.train()

    # Move tensors to device; keep 'task' (list[str]) as-is.
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)

    loss, metrics = policy.forward(batch)
    print(f"loss = {loss.item()}")
    print(f"metrics = {metrics}")
    assert torch.isfinite(loss).all(), "loss is NaN/Inf!"
    print("SMOKE TEST PASSED: forward pass ran, loss is finite.")

    loss.backward()
    grad_norms = [p.grad.norm().item() for p in policy.get_optim_params() if p.grad is not None]
    print(f"num params with grad: {len(grad_norms)}, max grad norm: {max(grad_norms) if grad_norms else None}")


if __name__ == "__main__":
    main()
