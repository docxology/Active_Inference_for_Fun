"""
Utility functions and helpers for Active Inference experiments.

This module contains compatibility helpers, episode runners, plotting utilities,
and other shared functionality used across the package.
"""

from .compatibility import (
    infer_states_compat,
    sample_action_compat,
    reset_agent_compat,
    infer_policies_compat,
)
from .episode_runners import EpisodeRunner, RandomEpisodeRunner, AIFEpisodeRunner
from .plotting import moving_average
from .parsers import parse_pos

__all__ = [
    "infer_states_compat",
    "sample_action_compat",
    "reset_agent_compat",
    "infer_policies_compat",
    "EpisodeRunner",
    "RandomEpisodeRunner",
    "AIFEpisodeRunner",
    "moving_average",
    "parse_pos",
]
