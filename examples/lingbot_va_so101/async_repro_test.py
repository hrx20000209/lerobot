#!/usr/bin/env python
"""Repro test: does lingbot_va work through the async PolicyServer's observation->action path?

PolicyServer._get_action_chunk(observation) (policy_server.py:482-488) is what both the
classic and continuous async servers actually call on every GetActions/inference tick:

    def _get_action_chunk(self, observation):
        chunk = self.policy.predict_action_chunk(observation)
        ...

This calls LingBotVAPolicy.predict_action_chunk() directly, per observation, the same way
the async server would (bypassing select_action() entirely -- the async server has no
concept of select_action's per-substep _obs_buffer bookkeeping). This script mocks that
exact call pattern with real held-out frames (episode 95) standing in for the robot
client's observation stream, and prints, per chunk: predicted action, self._frame_st_id
(KV cache advancement), and similarity to the true episode motion -- to check whether the
policy is actually reacting to the new observations or not.

Does not touch lerobot source; only calls the public predict_action_chunk() interface the
same way policy_server.py does.
"""

import torch

from common import USED_ACTION_CHANNEL_IDS, attach_text_embed_cache, build_config, denormalize_action
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy

CHECKPOINT_DIR = "/data/rxhuang/lingbot_va_runs/full_20k/checkpoints/step_2000"
DATASET_ROOT = "/data/rxhuang/three_cubes_1"
EPISODE = 95
N_CHUNKS_TO_TEST = 6


def main():
    config = build_config(attn_mode="torch")
    from common import make_ds_meta

    ds_meta = make_ds_meta()
    policy = make_policy(config, ds_meta=ds_meta)
    attach_text_embed_cache(policy)

    from peft import PeftModel

    policy = PeftModel.from_pretrained(policy, CHECKPOINT_DIR, is_trainable=False).get_base_model()
    policy.eval()

    raw_ds = LeRobotDataset("three_cubes_1", root=DATASET_ROOT, episodes=[EPISODE])
    cam_keys = config.obs_cam_keys
    task = raw_ds[0]["task"]
    raw_frames_per_chunk = config.frame_chunk_size * 4  # 16

    def frame_batch(i):
        s = raw_ds[i]
        return {k: s[k].unsqueeze(0).to(config.device) for k in cam_keys} | {"task": [task]}

    policy.reset()
    print(f"initial: policy._frame_st_id = {policy._frame_st_id}, "
          f"policy._first_chunk = {policy._first_chunk}")

    prev_action = None
    for c in range(N_CHUNKS_TO_TEST):
        # This mirrors exactly what PolicyServer._get_action_chunk / _predict_action_chunk
        # does: it calls predict_action_chunk(observation) with the CURRENT single-frame
        # observation and nothing else -- it never touches _obs_buffer (that bookkeeping
        # only happens inside select_action, which the async server never calls).
        obs_frame_idx = c * raw_frames_per_chunk
        batch = frame_batch(obs_frame_idx)
        with torch.no_grad():
            actions = policy.predict_action_chunk(batch)  # [1, n_steps, n_used], normalized

        gt_start = obs_frame_idx if c == 0 else obs_frame_idx  # ground truth for reference only
        gt_len = min(actions.shape[1], len(raw_ds) - gt_start)
        gt = torch.stack([raw_ds[gt_start + j]["action"] for j in range(gt_len)])

        action_similarity_to_prev = None
        if prev_action is not None:
            n = min(prev_action.shape[0], actions.shape[1])
            action_similarity_to_prev = torch.nn.functional.cosine_similarity(
                prev_action[:n].flatten().unsqueeze(0), actions[0, :n].flatten().unsqueeze(0)
            ).item()

        print(
            f"chunk {c}: obs_frame_idx={obs_frame_idx} | "
            f"policy._frame_st_id={policy._frame_st_id} | "
            f"policy._obs_buffer len={len(policy._obs_buffer)} | "
            f"pred[0,:3]={actions[0, 0, :3].tolist()} | "
            f"cosine_sim_to_prev_chunk={action_similarity_to_prev}"
        )
        prev_action = actions[0].detach().clone()

    print(
        "\nIf policy._frame_st_id stays at 0 across all chunks and cosine_sim_to_prev_chunk "
        "stays very high (~1.0) despite feeding DIFFERENT real observations each time, the "
        "policy is not reacting to new observations at all -- it is repeatedly re-predicting "
        "chunk 0 (mod sampling noise), blind to everything the async server sends it after "
        "the first call."
    )


if __name__ == "__main__":
    main()
