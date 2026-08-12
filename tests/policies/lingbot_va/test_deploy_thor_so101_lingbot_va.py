from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "inference" / "deploy_thor_so101_lingbot_va.py"
SPEC = importlib.util.spec_from_file_location("deploy_thor_so101_lingbot_va", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
deploy = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = deploy
SPEC.loader.exec_module(deploy)


class FakeChunkPolicy:
    def __init__(self) -> None:
        self.config = SimpleNamespace(action_per_frame=4)
        self.observation_history_size = 16
        self._obs_buffer = []
        self.state_count = 0

    def set_observation_history(self, batches) -> None:
        self._obs_buffer = list(batches)
        self._obs_buffer.extend([self._obs_buffer[-1]] * (16 - len(self._obs_buffer)))

    def _states_to_executed(self, states):
        self.state_count = len(states)
        frames = len(states) // 4
        return torch.zeros(1, 30, frames, 4, 1)

    def predict_action_chunk(self, batch):
        assert batch is None
        assert len(self._obs_buffer) == self._executed_actions.shape[2] * 4
        return torch.zeros(1, 16, 6)


def test_first_async_refill_preserves_twelve_step_feedback() -> None:
    policy = FakeChunkPolicy()
    observations = [{"frame": index} for index in range(12)]
    states = [torch.full((1, 6), float(index)) for index in range(12)]

    chunk = deploy.predict_next_chunk_with_state_feedback(policy, observations, states, 12)

    assert chunk.shape == (1, 16, 6)
    assert len(policy._obs_buffer) == 12
    assert policy.state_count == 12
    assert policy._executed_actions.shape == (1, 30, 3, 4, 1)


def test_early_async_prefetch_pads_images_and_states_together() -> None:
    policy = FakeChunkPolicy()
    observations = [{"frame": index} for index in range(8)]
    states = [torch.full((1, 6), float(index)) for index in range(8)]

    deploy.predict_next_chunk_with_state_feedback(policy, observations, states, 16)

    assert len(policy._obs_buffer) == 16
    assert policy.state_count == 16
    assert policy._obs_buffer[-1] is observations[-1]
    assert policy._executed_actions.shape == (1, 30, 4, 4, 1)


def test_gripper_rate_limit_uses_previous_command_not_stalled_state() -> None:
    limiter = deploy.SafetyLimiter(
        joint_min=np.full(6, -100.0, dtype=np.float32),
        joint_max=np.full(6, 100.0, dtype=np.float32),
        max_arm_step=2.0,
        max_gripper_command_step=4.0,
    )
    measured = np.array([0, 0, 0, 0, 0, 15], dtype=np.float32)
    requested = np.array([10, 0, 0, 0, 0, 4], dtype=np.float32)

    first = limiter.apply(requested, measured)
    second = limiter.apply(requested, measured)
    third = limiter.apply(requested, measured)

    assert first[0] == 2
    assert [first[5], second[5], third[5]] == [11, 7, 4]


def test_arm_outside_envelope_returns_at_rate_limit_without_boundary_jump() -> None:
    limiter = deploy.SafetyLimiter(
        joint_min=np.full(6, -100.0, dtype=np.float32),
        joint_max=np.full(6, 100.0, dtype=np.float32),
        max_arm_step=2.0,
        max_gripper_command_step=4.0,
    )
    measured = np.array([140, -140, 0, 0, 0, 20], dtype=np.float32)
    requested = np.zeros(6, dtype=np.float32)

    safe = limiter.apply(requested, measured)

    np.testing.assert_array_equal(safe[:2], np.array([138, -138], dtype=np.float32))


def test_official_client_action_then_observation_history_has_all_twelve_results() -> None:
    """RobotClient pops an action before capturing each observation."""
    boundary = deploy.OfficialRefillBoundary(margin=0)
    history = []

    # The first captured result sees queue size 11 after action 0 was popped.
    for queue_size in range(11, -1, -1):
        observation = f"result_{11 - queue_size}"
        history.append(observation)
        begin_history, enqueue = boundary.update(
            queue_size,
            must_go=queue_size == 0,
            has_processed_observation=True,
        )
        if begin_history:
            # This mirrors the server boundary transition: discard pre-arrival idle
            # frames but retain this current result frame.
            history = [observation]

    assert enqueue
    assert history == [f"result_{index}" for index in range(12)]
