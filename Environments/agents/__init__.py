"""
Active Inference agent factories.

This module provides factory functions for constructing Active Inference
agents tailored to specific environments and experimental conditions.
"""

from .factory import build_gridworld_agent
from .nav3_factory import build_trimodal_nav_agent

__all__ = ["build_gridworld_agent", "build_trimodal_nav_agent"]
