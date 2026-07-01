#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import pickle  # nosec: local smoke test only
import time
from concurrent import futures
from pathlib import Path

import grpc
import numpy as np

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import RemotePolicyConfig, TimedObservation
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import send_bytes_in_chunks
from lerobot.utils.constants import OBS_STATE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one real GigaWorld request through the async gRPC stack.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("/data/rxhuang/three_cubes_1"))
    parser.add_argument("--repo-id", default="hrx2000/Three_Cubes_1")
    parser.add_argument("--revision", default="v0.1.0")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument(
        "--server-address",
        default=None,
        help="Connect to an already-running policy server instead of starting one in-process.",
    )
    parser.add_argument("--actions-per-chunk", type=int, default=16)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    return parser.parse_args()


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
    dataset = LeRobotDataset(
        args.repo_id,
        root=args.dataset_root,
        episodes=[args.episode],
        revision=args.revision,
        video_backend="torchcodec",
        return_uint8=False,
    )
    sample = dataset[args.frame]
    state = np.asarray(sample[OBS_STATE], dtype=np.float32)
    names = list(dataset.meta.features[OBS_STATE]["names"])
    lerobot_features = {
        OBS_STATE: dataset.meta.features[OBS_STATE],
        "observation.images.front": dataset.meta.features["observation.images.front"],
        "observation.images.wrist": dataset.meta.features["observation.images.wrist"],
    }
    raw_observation = {name: float(value) for name, value in zip(names, state, strict=True)}
    raw_observation.update(
        {
            "front": image_to_uint8_hwc(sample["observation.images.front"]),
            "wrist": image_to_uint8_hwc(sample["observation.images.wrist"]),
            "task": str(sample.get("task", "")),
        }
    )

    policy_server = None
    server = None
    if args.server_address is None:
        config = PolicyServerConfig(host="127.0.0.1", port=args.port, obs_queue_timeout=10.0)
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
    try:
        stub.Ready(services_pb2.Empty(), timeout=args.timeout_s)
        remote_config = RemotePolicyConfig(
            policy_type="giga_world",
            pretrained_name_or_path=str(checkpoint),
            lerobot_features=lerobot_features,
            actions_per_chunk=args.actions_per_chunk,
            device=args.device,
            task=str(sample.get("task", "")),
        )
        stub.SendPolicyInstructions(
            services_pb2.PolicySetup(data=pickle.dumps(remote_config)), timeout=args.timeout_s
        )
        observation = TimedObservation(
            observation=raw_observation,
            timestamp=time.time(),
            timestep=0,
            must_go=True,
        )
        chunks = send_bytes_in_chunks(
            pickle.dumps(observation), services_pb2.Observation, log_prefix="async smoke", silent=True
        )
        stub.SendObservations(chunks, timeout=args.timeout_s)
        response = stub.GetActions(services_pb2.Empty(), timeout=args.timeout_s)
        actions = pickle.loads(response.data)  # nosec: response is from the local in-process server
        if len(actions) != args.actions_per_chunk:
            raise RuntimeError(f"Expected {args.actions_per_chunk} actions, received {len(actions)}")
        action_array = np.stack([item.get_action().numpy() for item in actions])
        result = {
            "checkpoint": str(checkpoint),
            "actions_shape": list(action_array.shape),
            "finite": bool(np.isfinite(action_array).all()),
            "action_min": action_array.min(axis=0).tolist(),
            "action_max": action_array.max(axis=0).tolist(),
        }
        print(json.dumps(result, indent=2))
    finally:
        channel.close()
        if policy_server is not None:
            policy_server.stop()
        if server is not None:
            server.stop(grace=0).wait()


if __name__ == "__main__":
    main()
