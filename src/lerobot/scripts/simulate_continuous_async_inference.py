# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Dry-run simulation for continuous async inference."""

from __future__ import annotations

import argparse
import queue
import random
import threading
import time
from pathlib import Path

import torch

from lerobot.async_inference.continuous import (
    AsyncActionQueueManager,
    ContinuousAsyncConfig,
    ContinuousObservationPublisher,
    InferenceResult,
    StructuredEventLogger,
    monotonic_now,
    wall_now,
)
from lerobot.async_inference.helpers import TimedObservation
from lerobot.scripts.plot_async_timeline import plot_timeline


class SimulatedContinuousServer:
    def __init__(
        self,
        event_logger: StructuredEventLogger,
        control_fps: float,
        chunk_size: int,
        action_dim: int,
        latency_mean: float,
        latency_jitter: float,
    ) -> None:
        self.event_logger = event_logger
        self.control_fps = control_fps
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.latency_mean = latency_mean
        self.latency_jitter = latency_jitter
        self.shutdown_event = threading.Event()
        self._condition = threading.Condition()
        self._latest_obs: TimedObservation | None = None
        self._latest_recv_ts: float | None = None
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self.results: queue.Queue[InferenceResult] = queue.Queue()
        self._inference_id = 0
        self.dropped = 0

    def start(self) -> None:
        self._worker.start()

    def stop(self) -> None:
        self.shutdown_event.set()
        with self._condition:
            self._condition.notify_all()
        self._worker.join(timeout=2)

    def receive_observation(self, obs: TimedObservation) -> bool:
        recv_ts = monotonic_now()
        obs_id = obs.metadata["obs_id"]
        self.event_logger.record(
            "observation_received_by_server",
            obs_id=obs_id,
            server_obs_recv_ts=recv_ts,
            obs_capture_start_ts=obs.metadata.get("obs_capture_start_ts"),
            obs_capture_end_ts=obs.metadata.get("obs_capture_end_ts"),
            obs_send_ts=obs.metadata.get("obs_send_ts"),
        )
        with self._condition:
            if self._latest_obs is not None:
                self.dropped += 1
                self.event_logger.record(
                    "observation_dropped",
                    obs_id=self._latest_obs.metadata.get("obs_id"),
                    drop_reason="server_latest_only_overwrite",
                    dropped_observation_count=self.dropped,
                )
            self._latest_obs = obs
            self._latest_recv_ts = recv_ts
            self._condition.notify()
        return True

    def _take_latest(self) -> tuple[TimedObservation, float] | None:
        with self._condition:
            while self._latest_obs is None and not self.shutdown_event.is_set():
                self._condition.wait(timeout=0.1)
            if self._latest_obs is None:
                return None
            obs = self._latest_obs
            recv_ts = self._latest_recv_ts or monotonic_now()
            self._latest_obs = None
            self._latest_recv_ts = None
            return obs, recv_ts

    def _loop(self) -> None:
        while not self.shutdown_event.is_set():
            taken = self._take_latest()
            if taken is None:
                continue
            obs, recv_ts = taken
            obs_id = int(obs.metadata["obs_id"])
            inference_id = self._inference_id
            self._inference_id += 1
            start_ts = monotonic_now()
            self.event_logger.record(
                "inference_started",
                obs_id=obs_id,
                inference_id=inference_id,
                source_obs_id=obs_id,
                inference_start_ts=start_ts,
                server_queue_delay=start_ts - recv_ts,
            )
            latency = max(0.0, random.gauss(self.latency_mean, self.latency_jitter))
            time.sleep(latency)
            end_ts = monotonic_now()
            base = torch.linspace(0.0, 1.0, self.chunk_size).unsqueeze(1)
            phase = obs_id / max(1.0, self.control_fps)
            action_chunk = [
                torch.sin(torch.full((self.action_dim,), phase + float(base[i])))
                for i in range(self.chunk_size)
            ]
            self.event_logger.record(
                "inference_finished",
                inference_id=inference_id,
                source_obs_id=obs_id,
                inference_start_ts=start_ts,
                inference_end_ts=end_ts,
                action_chunk_size=self.chunk_size,
                model_latency=end_ts - start_ts,
                server_queue_delay=start_ts - recv_ts,
            )
            result = InferenceResult(
                inference_id=inference_id,
                source_obs_id=obs_id,
                source_obs_capture_ts=float(obs.metadata["obs_capture_end_ts"]),
                source_obs_capture_wall_time=obs.metadata.get("wall_time_ts"),
                inference_start_ts=start_ts,
                inference_end_ts=end_ts,
                generated_action_chunk=action_chunk,
                action_chunk_size=self.chunk_size,
                model_latency=end_ts - start_ts,
                server_queue_delay=start_ts - recv_ts,
            )
            self.results.put(result)
            self.event_logger.record(
                "action_chunk_sent",
                inference_id=inference_id,
                source_obs_id=obs_id,
                action_chunk_send_ts=monotonic_now(),
                action_chunk_size=self.chunk_size,
            )


def simulate(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "timeline.jsonl"
    plot_path = output_dir / "timeline.png"
    summary_path = output_dir / "summary.json"
    for path in (log_path, plot_path, summary_path, plot_path.with_suffix(".pdf")):
        if path.exists():
            path.unlink()
    logger = StructuredEventLogger(log_path, enabled=True)
    config = ContinuousAsyncConfig(
        continuous_obs_fps=args.observation_fps,
        control_fps=args.control_fps,
        aggregation_fn=args.aggregation_fn,
        stale_inference_max_age=args.stale_inference_max_age,
        blend_horizon=args.blend_horizon,
        blend_alpha=args.blend_alpha,
        max_joint_delta=args.max_joint_delta,
        max_gripper_delta=args.max_gripper_delta,
        max_joint_abs_range=args.max_joint_abs_range,
    )
    manager = AsyncActionQueueManager(config, logger)
    server = SimulatedContinuousServer(
        logger,
        control_fps=args.control_fps,
        chunk_size=args.chunk_size,
        action_dim=args.action_dim,
        latency_mean=args.inference_latency_mean,
        latency_jitter=args.inference_latency_jitter,
    )

    def capture(obs_id: int) -> TimedObservation:
        return TimedObservation(
            timestamp=wall_now(),
            timestep=obs_id,
            observation={"observation.state": torch.zeros(args.action_dim), "task": "dry-run"},
            must_go=True,
            metadata={"obs_id": obs_id},
        )

    publisher = ContinuousObservationPublisher(
        fps=args.observation_fps,
        capture_fn=capture,
        send_fn=server.receive_observation,
        event_logger=logger,
        max_pending_observations=1,
    )
    shutdown = threading.Event()

    def receiver_loop() -> None:
        while not shutdown.is_set():
            try:
                result = server.results.get(timeout=0.05)
            except queue.Empty:
                continue
            receive_ts = monotonic_now()
            logger.record(
                "action_chunk_received_by_client",
                inference_id=result.inference_id,
                source_obs_id=result.source_obs_id,
                action_chunk_size=result.action_chunk_size,
                obs_age_at_receive=receive_ts - result.source_obs_capture_ts,
                inference_latency=result.model_latency,
            )
            manager.apply_inference_result(result)

    def executor_loop() -> None:
        dt = 1.0 / args.control_fps
        while not shutdown.is_set():
            start = monotonic_now()
            item = manager.pop_next_action()
            if item is not None:
                manager.mark_action_execution_started(item)
                time.sleep(min(0.001, dt))
                manager.finish_action_execution(item, performed_action={"shadow_mode": True})
            time.sleep(max(0.0, dt - (monotonic_now() - start)))

    server.start()
    publisher.start()
    receiver = threading.Thread(target=receiver_loop, daemon=True)
    executor = threading.Thread(target=executor_loop, daemon=True)
    receiver.start()
    executor.start()
    time.sleep(args.duration)
    shutdown.set()
    publisher.stop()
    server.stop()
    receiver.join(timeout=2)
    executor.join(timeout=2)
    logger.close()
    plot_timeline(
        log_path=log_path,
        output_path=plot_path,
        window_start=args.window_start,
        window_duration=args.window_duration or args.duration,
        summary_path=summary_path,
    )
    return log_path, plot_path, summary_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--observation_fps", type=float, default=30.0)
    parser.add_argument("--control_fps", type=float, default=30.0)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--action_dim", type=int, default=6)
    parser.add_argument("--inference_latency_mean", type=float, default=0.6)
    parser.add_argument("--inference_latency_jitter", type=float, default=0.1)
    parser.add_argument(
        "--aggregation_fn",
        choices=["replace_remaining", "splice_by_timestamp", "smooth_blend_remaining", "conservative_update"],
        default="splice_by_timestamp",
    )
    parser.add_argument("--stale_inference_max_age", type=float, default=2.0)
    parser.add_argument("--blend_horizon", type=int, default=5)
    parser.add_argument("--blend_alpha", type=float, default=0.5)
    parser.add_argument("--max_joint_delta", type=float, default=None)
    parser.add_argument("--max_gripper_delta", type=float, default=None)
    parser.add_argument("--max_joint_abs_range", type=float, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/continuous_async_debug")
    parser.add_argument("--window_start", type=float, default=0.0)
    parser.add_argument("--window_duration", type=float, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    log_path, plot_path, summary_path = simulate(args)
    print(f"Wrote timeline log: {log_path}")
    print(f"Wrote timeline plot: {plot_path}")
    print(f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
