import time

import torch

from common import attach_text_embed_cache, build_config, make_dataset, make_ds_meta
from lerobot.policies.factory import make_policy

config = build_config()
ds_meta = make_ds_meta()
dataset = make_dataset(config, episodes=list(range(5)))
loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)
batch = next(iter(loader))
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        batch[k] = v.to("cuda")

policy = make_policy(config, ds_meta=ds_meta)
attach_text_embed_cache(policy)
policy.train()
policy._ensure_frozen_modules()

t0 = time.time()
task = batch.get("task")
if isinstance(task, str):
    task = [task]
text_emb = policy._get_t5_prompt_embeds(list(task), config.max_sequence_length)
torch.cuda.synchronize()
print(f"text encode: {time.time() - t0:.1f}s")

t0 = time.time()
latents = policy._encode_training_latents(batch)
torch.cuda.synchronize()
print(f"VAE encode latents: {time.time() - t0:.1f}s, shape={latents.shape}")

t0 = time.time()
loss, metrics = policy.forward(batch)
torch.cuda.synchronize()
print(f"full forward (incl re-encoding text+vae): {time.time() - t0:.1f}s, metrics={metrics}")

t0 = time.time()
loss.backward()
torch.cuda.synchronize()
print(f"backward: {time.time() - t0:.1f}s")

# second forward: text embed should now be cached (same task string every batch here)
t0 = time.time()
loss2, metrics2 = policy.forward(batch)
torch.cuda.synchronize()
print(f"second forward (cached text embed): {time.time() - t0:.1f}s, metrics={metrics2}")
