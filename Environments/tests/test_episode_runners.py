"""
Tests for episode runner classes.
"""

import pytest
from unittest.mock import Mock, patch

from ..utils.episode_runners import (
    EpisodeResult,
    EpisodeRunner,
    RandomEpisodeRunner,
    AIFEpisodeRunner,
)


class TestEpisodeResult:
    """Test EpisodeResult class."""

    def test_episode_result_creation(self):
        """Test basic EpisodeResult creation."""
        result = EpisodeResult(
            total_return=5.5,
            steps=10,
            outcome="reward",
            episode_idx=3
        )

        assert result.total_return == 5.5
        assert result.steps == 10
        assert result.outcome == "reward"
        assert result.episode_idx == 3

    def test_episode_result_to_dict(self):
        """Test conversion to dictionary."""
        result = EpisodeResult(1.0, 5, "reward")
        result_dict = result.to_dict()

        expected = {"return": 1.0, "steps": 5, "outcome": "reward"}
        assert result_dict == expected

    def test_episode_result_to_tuple(self):
        """Test conversion to tuple."""
        result = EpisodeResult(2.5, 8, "timeout")
        result_tuple = result.to_tuple()

        assert result_tuple == (2.5, 8)

    def test_episode_result_repr(self):
        """Test string representation."""
        result = EpisodeResult(1.5, 6, "punish", episode_idx=2)
        repr_str = repr(result)

        assert "EpisodeResult" in repr_str
        assert "return=1.5" in repr_str
        assert "steps=6" in repr_str
        assert "outcome='punish'" in repr_str


class TestEpisodeRunner:
    """Test base EpisodeRunner class."""

    def test_episode_runner_initialization(self, simple_gridworld, rng):
        """Test EpisodeRunner initialization."""
        runner = EpisodeRunner(
            env=simple_gridworld,
            rng=rng,
            render=True,
            render_fps=10,
            episode_idx=5
        )

        assert runner.env is simple_gridworld
        assert runner.rng is rng
        assert runner.render is True
        assert runner.render_fps == 10
        assert runner.episode_idx == 5

    def test_episode_runner_invalid_env(self, rng):
        """Test error when env lacks required methods."""
        invalid_env = Mock()
        # Mock without required methods

        with pytest.raises(ValueError, match="must have.*method"):
            EpisodeRunner(invalid_env, rng)

    def test_determine_outcome_terminated(self, simple_gridworld, rng):
        """Test outcome determination for terminated episodes."""
        runner = EpisodeRunner(simple_gridworld, rng)

        # Manually set agent position to reward
        simple_gridworld.pos = simple_gridworld.reward_pos

        outcome = runner._determine_outcome(terminated=True, truncated=False)
        assert outcome == "reward"

        # Test punish
        simple_gridworld.pos = simple_gridworld.punish_pos
        outcome = runner._determine_outcome(terminated=True, truncated=False)
        assert outcome == "punish"

        # Test other terminal (edge case)
        simple_gridworld.pos = (0, 0)
        outcome = runner._determine_outcome(terminated=True, truncated=False)
        assert outcome == "terminal"

    def test_determine_outcome_truncated(self, simple_gridworld, rng):
        """Test outcome determination for truncated episodes."""
        runner = EpisodeRunner(simple_gridworld, rng)

        outcome = runner._determine_outcome(terminated=False, truncated=True)
        assert outcome == "timeout"

    def test_determine_outcome_unknown(self, simple_gridworld, rng):
        """Test outcome determination for unknown cases."""
        runner = EpisodeRunner(simple_gridworld, rng)

        outcome = runner._determine_outcome(terminated=False, truncated=False)
        assert outcome == "unknown"


class TestRandomEpisodeRunner:
    """Test RandomEpisodeRunner class."""

    def test_random_episode_run(self, simple_gridworld, rng):
        """Test running a random episode."""
        runner = RandomEpisodeRunner(
            env=simple_gridworld,
            rng=rng,
            max_steps=10,  # Short episode for testing
            episode_idx=1
        )

        result = runner.run_episode()

        assert isinstance(result, EpisodeResult)
        assert result.episode_idx == 1
        assert result.steps <= 10
        assert isinstance(result.total_return, float)
        assert result.outcome in ["reward", "punish", "timeout", "terminal"]

    def test_random_episode_with_rendering(self, simple_gridworld, rng):
        """Test random episode with rendering enabled."""
        with patch.object(simple_gridworld, 'render') as mock_render:
            runner = RandomEpisodeRunner(
                env=simple_gridworld,
                rng=rng,
                render=True,
                render_fps=60,  # Fast for testing
                max_steps=5
            )

            result = runner.run_episode()

            # Should have called render multiple times
            assert mock_render.call_count >= result.steps

    def test_random_episode_overlay(self, simple_gridworld, rng):
        """Test overlay callback functionality."""
        overlay_calls = []

        def mock_overlay(env, text):
            overlay_calls.append(text)

        runner = RandomEpisodeRunner(
            env=simple_gridworld,
            rng=rng,
            overlay_callback=mock_overlay,
            max_steps=3
        )

        result = runner.run_episode()

        # Should have called overlay for each step
        assert len(overlay_calls) == result.steps
        for call in overlay_calls:
            assert "RANDOM" in call
            assert "step" in call


class TestAIFEpisodeRunner:
    """Test AIFEpisodeRunner class."""

    def test_aif_episode_runner_initialization(self, simple_gridworld, mock_agent, rng):
        """Test AIFEpisodeRunner initialization."""
        controls = {"infer_policies": Mock()}

        runner = AIFEpisodeRunner(
            env=simple_gridworld,
            agent=mock_agent,
            rng=rng,
            sophisticated=True,
            controls=controls,
            max_steps=10
        )

        assert runner.agent is mock_agent
        assert runner.sophisticated is True
        assert runner.controls is controls

    def test_aif_episode_run(self, simple_gridworld, mock_agent, rng):
        """Test running an AIF episode."""
        # Mock agent methods
        mock_agent.sample_action.side_effect = [1, 2, 0, 1]  # Sequence of actions

        runner = AIFEpisodeRunner(
            env=simple_gridworld,
            agent=mock_agent,
            rng=rng,
            max_steps=4,  # Short episode
            episode_idx=2
        )

        result = runner.run_episode()

        assert isinstance(result, EpisodeResult)
        assert result.episode_idx == 2
        assert result.steps <= 4

        # Check that agent methods were called
        assert mock_agent.reset.called
        assert mock_agent.infer_states.call_count >= result.steps
        assert mock_agent.infer_policies.call_count >= result.steps
        assert mock_agent.sample_action.call_count == result.steps

    def test_aif_episode_with_controls_wrapper(self, simple_gridworld, mock_agent, rng):
        """Test AIF episode using factory controls wrapper."""
        controls = {"infer_policies": Mock()}

        runner = AIFEpisodeRunner(
            env=simple_gridworld,
            agent=mock_agent,
            rng=rng,
            controls=controls,
            max_steps=3
        )

        result = runner.run_episode()

        # Should use controls wrapper instead of direct agent method
        controls["infer_policies"].assert_called()
        mock_agent.infer_policies.assert_not_called()

    def test_aif_episode_overlay(self, simple_gridworld, mock_agent, rng):
        """Test overlay functionality for AIF episodes."""
        overlay_calls = []

        def mock_overlay(env, text):
            overlay_calls.append(text)

        mock_agent.sample_action.side_effect = [0, 0, 0]  # Actions

        runner = AIFEpisodeRunner(
            env=simple_gridworld,
            agent=mock_agent,
            rng=rng,
            overlay_callback=mock_overlay,
            max_steps=3
        )

        result = runner.run_episode()

        # Should have called overlay for each step
        assert len(overlay_calls) == result.steps
        for call in overlay_calls:
            assert "AIF" in call
            assert "step" in call

    def test_aif_episode_error_handling(self, simple_gridworld, rng):
        """Test error handling in AIF episodes."""
        failing_agent = Mock()
        failing_agent.reset.side_effect = Exception("Reset failed")

        runner = AIFEpisodeRunner(
            env=simple_gridworld,
            agent=failing_agent,
            rng=rng,
            max_steps=1
        )

        # Should handle agent errors gracefully (soft failures)
        result = runner.run_episode()
        assert isinstance(result, EpisodeResult)
