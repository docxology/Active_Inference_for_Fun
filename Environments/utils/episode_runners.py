"""
Episode runners for executing agent-environment interactions.

This module provides reusable classes for running episodes with different
types of agents (random, Active Inference) and environments, with support
for rendering, logging, and various output formats.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .compatibility import (
    infer_policies_compat,
    infer_states_compat,
    reset_agent_compat,
    sample_action_compat,
)

logger = logging.getLogger(__name__)


class EpisodeResult:
    """Container for episode execution results."""

    def __init__(
        self,
        total_return: float,
        steps: int,
        outcome: str,
        episode_idx: Optional[int] = None,
    ):
        self.total_return = total_return
        self.steps = steps
        self.outcome = outcome
        self.episode_idx = episode_idx

    def to_dict(self) -> Dict[str, Union[float, int, str]]:
        """Convert to dictionary format for compatibility."""
        return {
            "return": self.total_return,
            "steps": self.steps,
            "outcome": self.outcome,
        }

    def to_tuple(self) -> Tuple[float, int]:
        """Convert to (return, steps) tuple for simple cases."""
        return (self.total_return, self.steps)

    def __repr__(self) -> str:
        return f"EpisodeResult(return={self.total_return:.3f}, steps={self.steps}, outcome='{self.outcome}')"


class EpisodeRunner(ABC):
    """
    Abstract base class for episode runners.

    Provides common functionality for running episodes with different agent types
    and optional rendering/logging features.
    """

    def __init__(
        self,
        env: Any,
        rng: np.random.Generator,
        max_steps: Optional[int] = None,
        render: bool = False,
        render_fps: float = 12.0,
        episode_idx: Optional[int] = None,
        overlay_callback: Optional[callable] = None,
    ):
        """
        Initialize episode runner.

        Args:
            env: Gymnasium environment instance
            rng: Random number generator
            max_steps: Maximum steps per episode (if None, uses env's max_steps)
            render: Whether to render during execution
            render_fps: Frames per second for rendering
            episode_idx: Episode index for logging/overlays
            overlay_callback: Function to call for overlay updates during rendering
        """
        self.env = env
        self.rng = rng
        self.max_steps = max_steps or getattr(env, 'max_steps', 1000)
        self.render = render
        self.render_fps = render_fps
        self.episode_idx = episode_idx
        self.overlay_callback = overlay_callback

        # Validate environment has required methods
        required_methods = ['reset', 'step']
        for method in required_methods:
            if not hasattr(env, method) or not callable(getattr(env, method)):
                raise ValueError(f"Environment must have {method} method")

    def _reset_environment(self) -> Any:
        """Reset environment and return initial observation."""
        seed = int(self.rng.integers(0, 2**31 - 1))
        obs, info = self.env.reset(seed=seed)
        logger.debug(f"Environment reset with seed {seed}")
        return obs

    def _determine_outcome(self, terminated: bool, truncated: bool) -> str:
        """Determine episode outcome based on termination conditions."""
        if truncated:
            return "timeout"
        elif terminated:
            # Try to determine specific terminal outcome from environment state
            if hasattr(self.env, 'pos') and hasattr(self.env, 'reward_pos') and hasattr(self.env, 'punish_pos'):
                if self.env.pos == self.env.reward_pos:
                    return "reward"
                elif self.env.pos == self.env.punish_pos:
                    return "punish"
            return "terminal"
        else:
            return "unknown"

    def _update_overlay(self, step: int, total_return: float) -> None:
        """Update rendering overlay if callback provided."""
        if self.overlay_callback and callable(self.overlay_callback):
            agent_type = self.__class__.__name__.replace('EpisodeRunner', '').lower()
            overlay_text = f"{agent_type.upper()} | ep {self.episode_idx} | step {step} | R={total_return:.2f}"
            self.overlay_callback(self.env, overlay_text)

    def _render_step(self) -> None:
        """Render environment if rendering is enabled."""
        if self.render:
            self.env.render()
            if self.render_fps > 0:
                time.sleep(1.0 / self.render_fps)

    @abstractmethod
    def run_episode(self) -> EpisodeResult:
        """Run a single episode. Must be implemented by subclasses."""
        pass


class RandomEpisodeRunner(EpisodeRunner):
    """Episode runner for random policy agents."""

    def run_episode(self) -> EpisodeResult:
        """Run episode with random action selection."""
        logger.info(f"Starting random episode {self.episode_idx or 'N/A'}")

        obs = self._reset_environment()
        total_return = 0.0
        steps = 0

        while steps < self.max_steps:
            # Select random action
            action = int(self.rng.integers(0, self.env.action_space.n))

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_return += reward
            steps += 1

            # Handle rendering and overlays
            self._update_overlay(steps, total_return)
            self._render_step()

            # Check termination
            if terminated or truncated:
                break

        outcome = self._determine_outcome(terminated, truncated)
        result = EpisodeResult(total_return, steps, outcome, self.episode_idx)

        logger.info(f"Random episode {self.episode_idx or 'N/A'} completed: {result}")
        return result


class AIFEpisodeRunner(EpisodeRunner):
    """Episode runner for Active Inference agents."""

    def __init__(
        self,
        env: Any,
        agent: Any,
        rng: np.random.Generator,
        sophisticated: bool = False,
        controls: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize AIF episode runner.

        Args:
            env: Gymnasium environment instance
            agent: pymdp agent instance
            rng: Random number generator
            sophisticated: Whether to use sophisticated inference
            controls: Optional controls dict from agent factory
            **kwargs: Additional arguments for base class
        """
        super().__init__(env, rng, **kwargs)
        self.agent = agent
        self.sophisticated = sophisticated
        self.controls = controls

    def run_episode(self) -> EpisodeResult:
        """Run episode with Active Inference agent."""
        logger.info(f"Starting AIF episode {self.episode_idx or 'N/A'} (sophisticated={self.sophisticated})")

        # Reset agent beliefs
        reset_agent_compat(self.agent)

        obs = self._reset_environment()
        total_return = 0.0
        steps = 0

        while steps < self.max_steps:
            # Active Inference cycle
            infer_states_compat(self.agent, obs)
            infer_policies_compat(self.agent, self.sophisticated, self.controls)
            action = sample_action_compat(self.agent)

            # Step environment
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_return += reward
            steps += 1

            # Handle rendering and overlays
            self._update_overlay(steps, total_return)
            self._render_step()

            # Check termination
            if terminated or truncated:
                break

        outcome = self._determine_outcome(terminated, truncated)
        result = EpisodeResult(total_return, steps, outcome, self.episode_idx)

        logger.info(f"AIF episode {self.episode_idx or 'N/A'} completed: {result}")
        return result
