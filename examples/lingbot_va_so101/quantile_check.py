import time

import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_policy
from lerobot.policies.lingbot_va.configuration_lingbot_va import LingBotVAConfig

DATASET_ROOT = "/data/rxhuang/three_cubes_1"
OBS_CAM_KEYS = ["observation.images.front", "observation.images.right", "observation.images.wrist"]
USED = [14, 15, 16, 17, 18, 28]

config = LingBotVAConfig(
    obs_cam_keys=OBS_CAM_KEYS, camera_layout="width_concat",
    used_action_channel_ids=USED, attn_mode="flex",
    wan_pretrained_path="/home/rxhuang/Projects/models/lingbot-va-base",
    text_encoder_device="cpu", device="cuda",
)
fps = 30
dt = {k: [i / fps for i in config.observation_delta_indices] for k in OBS_CAM_KEYS}
dt["action"] = [i / fps for i in config.action_delta_indices]
dataset = LeRobotDataset("three_cubes_1", root=DATASET_ROOT, delta_timestamps=dt)
ds_meta = LeRobotDatasetMetadata("three_cubes_1", root=DATASET_ROOT)

q01 = torch.tensor(ds_meta.stats["action"]["q01"])
q99 = torch.tensor(ds_meta.stats["action"]["q99"])
print("q01", q01, "q99", q99, flush=True)

loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
batch = next(iter(loader))
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        batch[k] = v.to("cuda")

raw_action = batch["action"].clone()
norm_action = 2 * (raw_action.cpu() - q01) / (q99 - q01) - 1
print("raw action range", raw_action.min().item(), raw_action.max().item(), flush=True)
print("normalized action range", norm_action.min().item(), norm_action.max().item(), flush=True)
batch["action"] = norm_action.to("cuda")

policy = make_policy(config, ds_meta=ds_meta)
print("policy built", flush=True)
policy.train()
t0 = time.time()
loss, metrics = policy.forward(batch)
print(f"forward took {time.time() - t0:.1f}s", flush=True)
print("WITH quantile-normalized actions: loss=", loss.item(), metrics, flush=True)
