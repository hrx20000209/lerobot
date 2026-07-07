#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/hrx/miniconda3/envs/lerobot/bin/python}"
REPO_ID="${REPO_ID:-Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
CACHE_DIR="${CACHE_DIR:-/home/hrx/Projects/models/giga-world-policy}"
MAX_WORKERS="${MAX_WORKERS:-4}"

export REPO_ID
export CACHE_DIR
export MAX_WORKERS
export HF_HOME="${HF_HOME:-/home/hrx/.cache/huggingface}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"

exec "${PYTHON_BIN}" - <<'PY'
import os

from huggingface_hub import snapshot_download

repo_id = os.environ["REPO_ID"]
cache_dir = os.environ["CACHE_DIR"]
max_workers = int(os.environ["MAX_WORKERS"])

patterns = [
    "model_index.json",
    "scheduler/*",
    "tokenizer/*",
    "text_encoder/*",
    "transformer/*",
    "vae/*",
]

print(f"Downloading {repo_id} into cache_dir={cache_dir}", flush=True)
path = snapshot_download(
    repo_id=repo_id,
    cache_dir=cache_dir,
    allow_patterns=patterns,
    max_workers=max_workers,
)
print(f"Done: {path}", flush=True)
PY
