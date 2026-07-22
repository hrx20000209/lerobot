"""Teacher-forced open-loop action-prediction eval for one held-out episode.

"Open loop" here means: no env / no executing predicted actions anywhere. At every
chunk boundary we feed the *real* dataset frames + *real* dataset actions back into the
KV cache (teacher forcing), then ask the model to predict the next chunk, and compare
that prediction against the real next-chunk actions. This isolates "can the model
predict the right action given true history" from closed-loop drift/compounding-error
questions (which belong to Step 3's sync/async inference review, not here).
"""

import torch

from common import (
    ACTION_NAMES,
    DATASET_ROOT,
    USED_ACTION_CHANNEL_IDS,
    denormalize_action,
    normalize_action,
)
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _scatter_to_action_dim(action_norm_used: torch.Tensor, action_dim: int, used_ids: list[int]) -> torch.Tensor:
    """[*, len(used_ids)] (normalized, used-channel order) -> [*, action_dim] (zero elsewhere)."""
    shape = list(action_norm_used.shape[:-1]) + [action_dim]
    full = action_norm_used.new_zeros(shape)
    idx = torch.as_tensor(used_ids, device=action_norm_used.device)
    full[..., idx] = action_norm_used
    return full


@torch.no_grad()
def teacher_forced_rollout(policy, config, ds_meta, episode_index: int, q01, q99, device="cuda"):
    """Returns (pred_actions, gt_actions): both [T, len(used_action_channel_ids)] in physical units."""
    fps = 30
    raw_ds = LeRobotDataset("three_cubes_1", root=DATASET_ROOT, episodes=[episode_index])

    frame_chunk_size = config.frame_chunk_size
    action_per_frame = config.action_per_frame
    raw_frames_per_chunk = frame_chunk_size * 4  # VAE temporal downsample = 4
    n_frames = len(raw_ds)
    if n_frames < 1 + raw_frames_per_chunk:
        raise ValueError(f"episode {episode_index} too short ({n_frames} frames) for even 2 chunks")

    task = raw_ds[0]["task"]
    cam_keys = config.obs_cam_keys

    def frame_batch(i):
        s = raw_ds[i]
        return {k: s[k].unsqueeze(0).to(device) for k in cam_keys} | {"task": [task]}

    def gt_action_window(start, length):
        """[length, n_used] raw physical-unit ground-truth actions."""
        return torch.stack([raw_ds[start + j]["action"] for j in range(length)]).to(device)

    policy.reset()
    pred_chunks = []
    gt_chunks = []

    # First chunk: frame 0 conditions; predict_action_chunk returns (frame_chunk_size-1)*action_per_frame
    # steps of action (frame 0's own action is dropped -- it's the conditioning frame, see
    # modeling_lingbot_va.py predict_action_chunk's `is_first` branch).
    actions = policy.predict_action_chunk(frame_batch(0))  # [1, n_steps, n_used], normalized
    pred_chunks.append(actions[0])
    gt_chunks.append(gt_action_window(1, actions.shape[1]))

    # Frame 0 is the conditioning frame (not itself predicted): the first chunk covers
    # frames [0, 1+n_first) where n_first = (frame_chunk_size-1)*action_per_frame < a full
    # chunk's raw-frame span. Every later chunk (not "is_first") covers a full
    # raw_frames_per_chunk span with no dropped frame. Track this with a running counter
    # instead of assuming fixed-size chunk_start = c*raw_frames_per_chunk boundaries (that
    # assumption caused a real bug here: it produced a window_len of 19 instead of 16 on
    # the second chunk, since the first chunk only consumes 13 raw frames, not 16).
    consumed = 1 + actions.shape[1]  # frame 0 (conditioning) + the steps just predicted

    while consumed + raw_frames_per_chunk <= n_frames:
        # Feed the REAL observed frames + REAL executed actions for the window just finished
        # (teacher forcing) back into the KV cache, mirroring what an online client would do
        # with its actually-executed actions -- except ours are ground truth, not the model's.
        window_start = consumed
        window_len = raw_frames_per_chunk
        obs_buffer = [
            {k: raw_ds[window_start + j][k].unsqueeze(0).to(device) for k in cam_keys}
            for j in range(window_len)
        ]
        gt_used = gt_action_window(window_start, window_len)  # [window_len, n_used] raw units
        gt_used_norm = normalize_action(gt_used, q01, q99)
        # [window_len, n_used] -> [1, action_dim, F=window_len/action_per_frame, action_per_frame, 1]
        f = window_len // action_per_frame
        gt_full = _scatter_to_action_dim(gt_used_norm, 30, USED_ACTION_CHANNEL_IDS)  # [window_len, 30]
        gt_full = gt_full.view(f, action_per_frame, 30).permute(2, 0, 1).unsqueeze(0).unsqueeze(-1)

        policy._obs_buffer = obs_buffer
        policy._executed_actions = gt_full.to(policy.dtype)
        actions = policy.predict_action_chunk(None)  # [1, chunk_size, n_used], normalized
        pred_chunks.append(actions[0])
        gt_chunks.append(gt_action_window(window_start, actions.shape[1]))
        consumed = window_start + actions.shape[1]

    pred_norm = torch.cat(pred_chunks, dim=0)  # [T, n_used], normalized [-1, 1]-ish
    gt_raw = torch.cat(gt_chunks, dim=0)  # [T, n_used], raw physical units
    pred_raw = denormalize_action(pred_norm, q01, q99)
    return pred_raw.cpu(), gt_raw.cpu()


def per_dim_mse(pred_raw: torch.Tensor, gt_raw: torch.Tensor) -> dict:
    mse = ((pred_raw - gt_raw) ** 2).mean(dim=0)
    return {name: mse[i].item() for i, name in enumerate(ACTION_NAMES)}
