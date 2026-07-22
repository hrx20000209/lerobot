#!/usr/bin/env python
"""Drive several real LingBot-VA chunks through the async gRPC stack, no robot required.

Modelled on ``smoke_giga_world_async.py``, but deliberately requests **more than one**
chunk. One chunk proves nothing here: LingBot-VA is autoregressive, and the interesting
failure is on the *second* and later chunks, where ``PolicyServer`` calls
``predict_action_chunk(observation)`` again without ever having called ``select_action()``
(see ``examples/lingbot_va_so101/async_repro_test.py``). This script feeds a different real
dataset frame per chunk and reports ``_frame_st_id`` plus chunk-to-chunk cosine similarity,
so a policy that has gone blind to new observations is visible rather than silent:

    frame_st_id stuck at 0 + cosine ~1.0  ->  blind, re-predicting chunk 0
    frame_st_id advancing + cosine < 1.0  ->  consuming the observation stream

Usage:
    python scripts/inference/smoke_lingbot_va_async.py
    python scripts/inference/smoke_lingbot_va_async.py --chunks 6 --episode 95
"""

from __future__ import annotations

import argparse
import json
import pickle  # nosec: local smoke test only
import time
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np
import torch

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import send_bytes_in_chunks
from lerobot.utils.constants import OBS_STATE

CAMERAS = ("front", "right", "wrist")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run several LingBot-VA chunks through the async gRPC stack.")
    p.add_argument("--checkpoint", type=Path, default=Path("/home/hrx/Projects/models/three_cubes_1/lingbot_va_async"))
    p.add_argument("--dataset-root", type=Path, default=Path("/home/hrx/Datasets/three_cubes_1"))
    p.add_argument("--repo-id", default="three_cubes_1")
    p.add_argument("--video-backend", default="pyav", choices=("pyav", "torchcodec"))
    p.add_argument("--episode", type=int, default=95, help="Held-out episode (train used 0-94).")
    p.add_argument("--chunks", type=int, default=4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--port", type=int, default=18081)
    p.add_argument("--server-address", default=None, help="Use an already-running server instead.")
    p.add_argument("--actions-per-chunk", type=int, default=16)
    p.add_argument("--timeout-s", type=float, default=900.0)
    return p.parse_args()


def image_to_uint8_hwc(image) -> np.ndarray:
    array = image.detach().cpu().numpy() if hasattr(image, "detach") else np.asarray(image)
    if array.shape[0] in {1, 3, 4}:
        array = np.moveaxis(array, 0, -1)
    if array.dtype.kind == "f":
        array = np.clip(array, 0.0, 1.0) * 255.0
    return np.asarray(array, dtype=np.uint8)


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not (checkpoint / "config.json").exists():
        raise SystemExit(
            f"No config.json in {checkpoint}. Build it with "
            "scripts/inference/build_lingbot_va_three_cubes_checkpoint.py"
        )

    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        episodes=[args.episode],
        video_backend=args.video_backend,
        return_uint8=False,
    )
    lerobot_features = {OBS_STATE: dataset.meta.features[OBS_STATE]} | {
        f"observation.images.{c}": dataset.meta.features[f"observation.images.{c}"] for c in CAMERAS
    }
    state_names = list(dataset.meta.features[OBS_STATE]["names"])
    task = str(dataset[0].get("task", ""))

    # One frame per chunk, spaced by the frames a chunk actually consumes
    # (frame_chunk_size x the VAE's x4 temporal downsample = 16 dataset frames).
    stride = 16

    policy_server = None
    server = None
    if args.server_address is None:
        config = PolicyServerConfig(host="127.0.0.1", port=args.port, obs_queue_timeout=30.0)
        policy_server = PolicyServer(config)
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
        services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
        server.add_insecure_port(f"{config.host}:{config.port}")
        server.start()
        server_address = f"{config.host}:{config.port}"
    else:
        server_address = args.server_address

    channel = grpc.insecure_channel(server_address)
    stub = services_pb2_grpc.AsyncInferenceStub(channel)
    results = []
    try:
        stub.Ready(services_pb2.Empty(), timeout=args.timeout_s)
        remote_config = RemotePolicyConfig(
            policy_type="lingbot_va",
            pretrained_name_or_path=str(checkpoint),
            lerobot_features=lerobot_features,
            actions_per_chunk=args.actions_per_chunk,
            device=args.device,
            task=task,
        )
        t0 = time.perf_counter()
        stub.SendPolicyInstructions(
            services_pb2.PolicySetup(data=pickle.dumps(remote_config)), timeout=args.timeout_s
        )
        print(f"policy loaded in {time.perf_counter() - t0:.1f}s", flush=True)

        # Instrument the KV-feedback hand-off so "the policy got real, distinct keyframes" is
        # asserted rather than inferred from the action values.
        pushed: list[dict] = []
        if policy_server is not None:
            inner = getattr(policy_server.policy, "get_base_model", lambda: policy_server.policy)()
            original_set = inner.set_observation_history

            def counting_set(batches, _orig=original_set, _log=pushed):
                cam0 = f"observation.images.{CAMERAS[0]}"
                sigs = {float(b[cam0].float().mean()) for b in batches if cam0 in b}
                _log.append({"n": len(batches), "distinct": len(sigs)})
                return _orig(batches)

            inner.set_observation_history = counting_set

        def raw_obs_at(frame_idx: int) -> dict:
            sample = dataset[min(frame_idx, len(dataset) - 1)]
            obs = {
                name: float(v)
                for name, v in zip(state_names, np.asarray(sample[OBS_STATE], dtype=np.float32), strict=True)
            }
            obs |= {cam: image_to_uint8_hwc(sample[f"observation.images.{cam}"]) for cam in CAMERAS}
            obs["task"] = task
            return obs

        prev = None
        timestep = 0
        for c in range(args.chunks):
            # Stream the frames observed "while the previous chunk executed", the way a real
            # robot client does -- one per executed action. Sending a single frame per chunk
            # would leave the policy's KV feedback with nothing but the current frame and
            # would not exercise PolicyServer._push_observation_history at all.
            for j in range(stride):
                observation = TimedObservation(
                    observation=raw_obs_at(c * stride + j),
                    timestamp=time.time(),
                    timestep=timestep,
                    must_go=True,
                )
                timestep += 1
                stub.SendObservations(
                    send_bytes_in_chunks(
                        pickle.dumps(observation), services_pb2.Observation, log_prefix="", silent=True
                    ),
                    timeout=args.timeout_s,
                )
            t = time.perf_counter()
            response = stub.GetActions(services_pb2.Empty(), timeout=args.timeout_s)
            dt = time.perf_counter() - t
            actions = pickle.loads(response.data)  # nosec: local in-process server
            if not actions:
                raise RuntimeError(f"chunk {c}: server returned no actions")
            arr = np.stack([item.get_action().numpy() for item in actions])

            cos = None
            if prev is not None:
                n = min(len(prev), len(arr))
                cos = float(
                    torch.nn.functional.cosine_similarity(
                        torch.tensor(prev[:n].ravel()).unsqueeze(0),
                        torch.tensor(arr[:n].ravel()).unsqueeze(0),
                    )
                )
            frame_st_id = None
            if policy_server is not None:
                inner = getattr(policy_server.policy, "get_base_model", lambda: policy_server.policy)()
                frame_st_id = getattr(inner, "_frame_st_id", None)

            kf = pushed[-1] if pushed else None
            kf_desc = "n/a" if kf is None else f"{kf['n']} ({kf['distinct']} distinct)"
            cos_desc = "n/a" if cos is None else f"{cos:.4f}"
            print(
                f"chunk {c}: {dt:6.2f}s | {arr.shape[0]:2d} actions | frame_st_id={frame_st_id} | "
                f"keyframes={kf_desc} | cos_to_prev={cos_desc} | finite={bool(np.isfinite(arr).all())}",
                flush=True,
            )
            results.append(
                {
                    "chunk": c,
                    "seconds": round(dt, 3),
                    "n_actions": int(arr.shape[0]),
                    "frame_st_id": frame_st_id,
                    "cosine_to_prev": cos,
                    "keyframes_pushed": kf,
                    "action_min": arr.min(axis=0).round(3).tolist(),
                    "action_max": arr.max(axis=0).round(3).tolist(),
                }
            )
            prev = arr

        ids = [r["frame_st_id"] for r in results if r["frame_st_id"] is not None]
        blind = len(ids) > 1 and len(set(ids)) == 1
        print(json.dumps({"checkpoint": str(checkpoint), "chunks": results, "blind": blind}, indent=2))
        if blind:
            raise SystemExit("FAIL: _frame_st_id never advanced -- policy is blind to new observations")
    finally:
        channel.close()
        if policy_server is not None:
            policy_server.stop()
        if server is not None:
            server.stop(grace=0).wait()


if __name__ == "__main__":
    main()
