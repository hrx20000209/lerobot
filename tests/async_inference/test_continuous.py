# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _result(
    *,
    inference_id: int = 1,
    source_obs_id: int = 1,
    source_obs_capture_ts: float = 100.0,
    values: list[float] | None = None,
):
    from lerobot.async_inference.continuous import InferenceResult

    values = values or [1.0, 2.0, 3.0]
    actions = [torch.full((2,), value) for value in values]
    return InferenceResult(
        inference_id=inference_id,
        source_obs_id=source_obs_id,
        source_obs_capture_ts=source_obs_capture_ts,
        source_obs_capture_wall_time=None,
        inference_start_ts=source_obs_capture_ts + 0.1,
        inference_end_ts=source_obs_capture_ts + 0.2,
        generated_action_chunk=actions,
        action_chunk_size=len(actions),
        model_latency=0.1,
        server_queue_delay=0.0,
    )


def test_replace_remaining_only_replaces_pending(monkeypatch):
    from lerobot.async_inference import continuous
    from lerobot.async_inference.continuous import AsyncActionQueueManager, ContinuousAsyncConfig

    ts = 100.0
    monkeypatch.setattr(continuous, "monotonic_now", lambda: ts)
    manager = AsyncActionQueueManager(
        ContinuousAsyncConfig(aggregation_fn="replace_remaining", stale_inference_max_age=10.0)
    )
    manager.apply_inference_result(_result(inference_id=1, source_obs_capture_ts=99.9, values=[1, 2, 3]))
    executing = manager.pop_next_action()
    assert executing is not None
    assert executing.status == "executing"

    manager.apply_inference_result(_result(inference_id=2, source_obs_capture_ts=99.9, values=[10, 11]))

    assert executing.status == "executing"
    assert [item.source_inference_id for item in manager.snapshot()] == [2, 2]
    assert [item.status for item in manager.snapshot()] == ["pending", "pending"]


def test_splice_by_timestamp_discards_when_result_too_old(monkeypatch):
    from lerobot.async_inference import continuous
    from lerobot.async_inference.continuous import AsyncActionQueueManager, ContinuousAsyncConfig

    monkeypatch.setattr(continuous, "monotonic_now", lambda: 110.0)
    manager = AsyncActionQueueManager(
        ContinuousAsyncConfig(
            aggregation_fn="splice_by_timestamp",
            control_fps=30.0,
            stale_inference_max_age=20.0,
        )
    )
    stats = manager.apply_inference_result(_result(source_obs_capture_ts=100.0, values=[1, 2, 3]))

    assert stats.applied is False
    assert stats.discard_reason == "splice_index_out_of_chunk"
    assert manager.qsize() == 0


def test_conservative_update_rejects_large_delta(monkeypatch):
    from lerobot.async_inference import continuous
    from lerobot.async_inference.continuous import AsyncActionQueueManager, ContinuousAsyncConfig

    monkeypatch.setattr(continuous, "monotonic_now", lambda: 100.0)
    manager = AsyncActionQueueManager(
        ContinuousAsyncConfig(
            aggregation_fn="replace_remaining",
            stale_inference_max_age=10.0,
        )
    )
    manager.apply_inference_result(_result(inference_id=1, source_obs_capture_ts=99.9, values=[0, 0, 0]))
    manager.config.aggregation_fn = "conservative_update"
    manager.config.max_joint_delta = 0.5
    stats = manager.apply_inference_result(_result(inference_id=2, source_obs_capture_ts=99.9, values=[5, 5]))

    assert stats.applied is False
    assert stats.discard_reason.startswith("max_joint_delta_exceeded")
    assert [item.source_inference_id for item in manager.snapshot()] == [1, 1, 1]
