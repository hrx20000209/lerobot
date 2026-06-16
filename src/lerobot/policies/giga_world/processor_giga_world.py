#!/usr/bin/env python

from __future__ import annotations

from typing import Any

from lerobot.processor import (
    IdentityProcessorStep,
    PolicyAction,
    PolicyProcessorPipeline,
    policy_action_to_transition,
    transition_to_policy_action,
)
from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME

from .configuration_giga_world import GigaWorldConfig


def make_giga_world_pre_post_processors(
    config: GigaWorldConfig,
    dataset_stats: dict[str, dict[str, Any]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    del config, dataset_stats
    return (
        PolicyProcessorPipeline[dict[str, Any], dict[str, Any]](
            steps=[IdentityProcessorStep()],
            name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        ),
        PolicyProcessorPipeline[PolicyAction, PolicyAction](
            steps=[IdentityProcessorStep()],
            name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
            to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        ),
    )
