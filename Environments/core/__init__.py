"""
Core environment implementations for Active Inference experiments.

This module contains the fundamental grid world environments used
throughout the Active Inference for Fun package.
"""

from .gridworld_env import GridWorld
from .oriented_trimodal_grid import OrientedTriModalGrid

__all__ = ["GridWorld", "OrientedTriModalGrid"]
