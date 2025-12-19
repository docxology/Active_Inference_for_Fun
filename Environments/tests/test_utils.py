"""
General utility tests.
"""

import pytest
import numpy as np


class TestPackageImports:
    """Test that package imports work correctly."""

    def test_core_imports(self):
        """Test that core modules can be imported."""
        from ..core import GridWorld, OrientedTriModalGrid

        # Should be able to create instances
        env1 = GridWorld(n_rows=3, n_cols=3, reward_pos=(2, 2), punish_pos=(0, 2))
        env2 = OrientedTriModalGrid(n_rows=4, n_cols=4, reward_pos=(3, 3), punish_pos=(0, 3))

        assert env1 is not None
        assert env2 is not None

    def test_agents_imports(self):
        """Test that agent modules can be imported."""
        from ..agents import build_gridworld_agent, build_trimodal_nav_agent

        assert callable(build_gridworld_agent)
        assert callable(build_trimodal_nav_agent)

    def test_utils_imports(self):
        """Test that utility modules can be imported."""
        from ..utils import (
            infer_states_compat,
            sample_action_compat,
            reset_agent_compat,
            infer_policies_compat,
            EpisodeRunner,
            RandomEpisodeRunner,
            AIFEpisodeRunner,
            moving_average,
            parse_pos,
        )

        # Should all be importable
        assert callable(infer_states_compat)
        assert callable(sample_action_compat)
        assert callable(reset_agent_compat)
        assert callable(infer_policies_compat)
        assert EpisodeRunner is not None
        assert RandomEpisodeRunner is not None
        assert AIFEpisodeRunner is not None
        assert callable(moving_average)
        assert callable(parse_pos)


class TestNumpyCompatibility:
    """Test numpy compatibility across functions."""

    def test_numpy_array_handling(self):
        """Test that functions handle numpy arrays properly."""
        from ..utils.compatibility import sample_action_compat
        from unittest.mock import Mock

        agent = Mock()

        # Test with numpy scalar
        agent.sample_action.return_value = np.int32(2)
        result = sample_action_compat(agent)
        assert result == 2
        assert isinstance(result, int)

        # Test with numpy array
        agent.sample_action.return_value = np.array([3])
        result = sample_action_compat(agent)
        assert result == 3
        assert isinstance(result, int)

    def test_moving_average_with_numpy(self):
        """Test moving average with numpy arrays."""
        from ..utils.plotting import moving_average

        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = moving_average(x, w=3)

        assert isinstance(result, np.ndarray)
        assert len(result) == len(x)
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        assert result[2] == 2.0  # (1+2+3)/3


class TestEnvironmentIntegration:
    """Test integration between components."""

    def test_episode_runner_with_environment(self, simple_gridworld, rng):
        """Test that episode runners work with real environments."""
        from ..utils.episode_runners import RandomEpisodeRunner

        runner = RandomEpisodeRunner(
            env=simple_gridworld,
            rng=rng,
            max_steps=5  # Short for testing
        )

        result = runner.run_episode()

        assert result.steps <= 5
        assert isinstance(result.total_return, float)
        assert result.outcome in ["reward", "punish", "timeout", "terminal"]

    def test_parser_integration(self):
        """Test that parsers work together."""
        from ..utils.parsers import create_gridworld_parser, parse_gridworld_args

        parser = create_gridworld_parser("Integration test")
        args = parser.parse_args([
            "--rows", "4", "--cols", "4",
            "--reward-pos", "3,3", "--punish-pos", "0,3",
            "--max-steps", "50"
        ])

        result = parse_gridworld_args(args)

        assert result['n_rows'] == 4
        assert result['n_cols'] == 4
        assert result['reward_pos'] == (3, 3)
        assert result['punish_pos'] == (0, 3)
        assert result['max_steps'] == 50


class TestConstantsAndTypes:
    """Test constants and type definitions."""

    def test_environment_constants(self):
        """Test that environment constants are properly defined."""
        from ..core.gridworld_env import Action, Pos
        from typing import Tuple

        # Should be type aliases
        assert Action == int
        # Check that Pos is a tuple type (either old or new syntax)
        assert Pos == Tuple[int, int] or str(Pos).endswith('tuple[int, int]')

    def test_oriented_constants(self):
        """Test oriented grid constants."""
        from ..core.oriented_trimodal_grid import (
            ORIENTS, ORI2IDX, FWD, TURN_L, TURN_R,
            CLASS_EMPTY, CLASS_EDGE, CLASS_RED, CLASS_GREEN,
            M2_EDGE, M2_RED, M2_GREEN
        )

        assert len(ORIENTS) == 4
        assert ORI2IDX["N"] == 0
        assert ORI2IDX["E"] == 1
        assert ORI2IDX["S"] == 2
        assert ORI2IDX["W"] == 3

        assert FWD == 0
        assert TURN_L == 1
        assert TURN_R == 2

        assert CLASS_EMPTY == 0
        assert CLASS_EDGE == 1
        assert CLASS_RED == 2
        assert CLASS_GREEN == 3

        assert M2_EDGE == 0
        assert M2_RED == 1
        assert M2_GREEN == 2
