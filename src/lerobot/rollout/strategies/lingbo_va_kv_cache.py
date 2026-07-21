# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rollout strategy for LingBoVA: drives the reset -> infer chunk -> execute -> compute_kv_cache
protocol explicitly, since the generic `BaseStrategy`/`SyncInferenceEngine`/`send_next_action`
path has no concept of "chunk boundary" or "actually-executed (post safety-clip) action" — both
of which LingBoVA's KV-cache commit (`LingBoVAPolicy.commit_executed_action`) requires.

Only single-camera-tick (no action interpolation) is supported: `--strategy.type=lingbo_va_kv_cache`
requires `--interpolation_multiplier=1`, since the model's `n_action_steps`/`action_per_frame`
granularity is defined in the policy's own timestep space, not an interpolated control-loop space.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import torch

from lerobot.policies.utils import make_robot_action, prepare_observation_for_inference
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.robot_utils import precise_sleep

from ..configs import RolloutStrategyConfig
from ..context import RolloutContext
from .core import RolloutStrategy

logger = logging.getLogger(__name__)


@RolloutStrategyConfig.register_subclass("lingbo_va_kv_cache")
@dataclass
class LingboVaKvCacheStrategyConfig(RolloutStrategyConfig):
    """Autonomous LingBoVA rollout that correctly drives the KV-cache streaming protocol."""

    # Only print/plot predicted actions; never call robot.send_action(). Full sense -> infer
    # loop still runs so end-to-end latency can be measured.
    dry_run: bool = False
    # Per-joint absolute-range safety clip, as a fraction of the checkpoint's train-only-scoped
    # [q01, q99] range to pad on each side before clipping predicted actions (0 = clip exactly to
    # [q01, q99]). Per-step delta clipping is handled by the robot itself
    # (`--robot.max_relative_target`), not duplicated here.
    action_range_pad_fraction: float = 0.0


class LingboVaKvCacheStrategy(RolloutStrategy):
    """Reset once, then repeat: infer one chunk -> execute n_action_steps on the robot ->
    commit the actually-executed actions to the KV cache -> infer next chunk."""

    def setup(self, ctx: RolloutContext) -> None:
        self._init_engine(ctx)
        policy = ctx.policy.policy
        if not hasattr(policy, "commit_executed_action"):
            raise TypeError(
                "LingboVaKvCacheStrategy requires a LingBoVAPolicy (missing commit_executed_action); "
                f"got {type(policy).__name__}."
            )
        if ctx.runtime.cfg.interpolation_multiplier != 1:
            raise ValueError(
                "LingboVaKvCacheStrategy does not support action interpolation "
                "(interpolation_multiplier must be 1) — the model's action-chunk granularity is "
                "defined in its own timestep space, not an interpolated control-loop space."
            )
        n_action_steps = policy.config.n_action_steps
        action_per_frame = policy.config.action_per_frame
        if n_action_steps % action_per_frame != 0:
            raise ValueError(
                f"policy.config.n_action_steps={n_action_steps} must be a multiple of "
                f"action_per_frame={action_per_frame} for commit_executed_action to align with "
                "the server's KV-cache latent-frame granularity."
            )
        self._policy = policy
        self._executed_actions: list[list[float]] = []
        self._last_obs_batch: dict | None = None
        q01 = policy._action_stats["q01"].tolist()
        q99 = policy._action_stats["q99"].tolist()
        pad = self.config.action_range_pad_fraction
        self._joint_lo = [lo - pad * (hi - lo) for lo, hi in zip(q01, q99, strict=False)]
        self._joint_hi = [hi + pad * (hi - lo) for lo, hi in zip(q01, q99, strict=False)]
        logger.info(
            "LingboVaKvCacheStrategy ready (dry_run=%s, n_action_steps=%d)",
            self.config.dry_run,
            n_action_steps,
        )

    def run(self, ctx: RolloutContext) -> None:
        cfg = ctx.runtime.cfg
        robot = ctx.hardware.robot_wrapper
        policy = self._policy
        features = ctx.data.dataset_features
        ordered_action_keys = ctx.data.ordered_action_keys
        device = torch.device(policy.config.device or "cpu")
        task = cfg.task
        robot_type = robot.robot_type
        control_interval = 1.0 / cfg.fps

        start_time = time.perf_counter()
        self._engine.resume()
        logger.info("LingboVaKvCache strategy control loop started")

        while not ctx.runtime.shutdown_event.is_set():
            loop_start = time.perf_counter()
            if cfg.duration > 0 and (time.perf_counter() - start_time) >= cfg.duration:
                logger.info("Duration limit reached (%.0fs)", cfg.duration)
                break

            obs_raw = robot.get_observation()
            obs_processed = self._process_observation_and_notify(ctx.processors, obs_raw)
            if self._handle_warmup(cfg.use_torch_compile, loop_start, control_interval):
                continue

            obs_frame = build_dataset_frame(features, obs_processed, prefix=OBS_STR)
            observation = prepare_observation_for_inference(obs_frame, device, task, robot_type)
            observation = ctx.policy.preprocessor(observation)
            self._last_obs_batch = observation

            about_to_start_new_chunk = len(policy._action_queue) == 0
            with torch.inference_mode():
                action = policy.select_action(observation)
                action = ctx.policy.postprocessor(action)
            action_tensor = action.squeeze(0).cpu()
            action_dict = make_robot_action(action_tensor, features)
            ordered_action = [action_dict[k] for k in ordered_action_keys]

            # Absolute-range safety clip to the checkpoint's train-only-scoped [q01, q99] (per
            # joint). Per-step delta clipping is the robot's own job (--robot.max_relative_target).
            for i, key in enumerate(ordered_action_keys):
                clipped = min(max(action_dict[key], self._joint_lo[i]), self._joint_hi[i])
                if clipped != action_dict[key]:
                    logger.warning(
                        "Clipped predicted action[%s]=%.3f to [%.3f, %.3f] (train-data range).",
                        key,
                        action_dict[key],
                        self._joint_lo[i],
                        self._joint_hi[i],
                    )
                    action_dict[key] = clipped
                    ordered_action[i] = clipped

            if self.config.dry_run:
                logger.info("[dry-run] predicted action: %s", action_dict)
                executed_action_dict = action_dict
            else:
                processed = ctx.processors.robot_action_processor((action_dict, obs_raw))
                executed_action_dict = robot.send_action(processed)
            self._log_telemetry(obs_processed, executed_action_dict, ctx.runtime)

            executed_ordered = [
                float(executed_action_dict.get(k, ordered_action[i])) for i, k in enumerate(ordered_action_keys)
            ]
            self._executed_actions.append(executed_ordered)

            if len(self._executed_actions) >= policy.config.n_action_steps:
                executed_tensor = torch.tensor(self._executed_actions[: policy.config.n_action_steps])
                policy.commit_executed_action(executed_tensor, self._last_obs_batch)
                self._executed_actions.clear()
                logger.debug("Committed %d executed actions to KV cache (chunk boundary).", len(executed_tensor))
            elif about_to_start_new_chunk and self._executed_actions:
                logger.warning(
                    "Action queue emptied after only %d/%d executed steps; commit_executed_action "
                    "was NOT called (partial chunk). This should not happen in normal operation.",
                    len(self._executed_actions),
                    policy.config.n_action_steps,
                )

            dt = time.perf_counter() - loop_start
            if (sleep_t := control_interval - dt) > 0:
                precise_sleep(sleep_t)
            else:
                logger.warning(
                    "LingboVaKvCache loop running slower (%.1f Hz) than target FPS (%.1f Hz).",
                    1 / dt if dt > 0 else float("inf"),
                    cfg.fps,
                )

    def teardown(self, ctx: RolloutContext) -> None:
        self._teardown_hardware(ctx.hardware, return_to_initial_position=ctx.runtime.cfg.return_to_initial_position)
        logger.info("LingboVaKvCache strategy teardown complete")
