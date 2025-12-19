"""
Comprehensive tests for GridWorld agent factory.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from ..agents.factory import build_gridworld_agent


class TestBuildGridworldAgent:
    """Test build_gridworld_agent function."""

    def test_basic_agent_creation(self, mock_pymdp_agent):
        """Test basic agent creation with valid parameters."""
        with patch('active_inference.environments.agents.factory.Agent', mock_pymdp_agent):
            agent, model, controls = build_gridworld_agent(
                n_rows=3,
                n_cols=4,
                reward_pos=(2, 3),
                punish_pos=(0, 3)
            )

        assert agent is not None
        assert isinstance(model, dict)
        assert isinstance(controls, dict)

        # Check model components
        assert 'A' in model
        assert 'B' in model
        assert 'C' in model
        assert 'D' in model

        # Check matrix shapes
        assert model['A'].shape == (12, 12)  # O x S (12 obs, 12 states)
        assert model['B'].shape == (12, 12, 4)  # S x S x U (4 actions)
        assert model['C'].shape == (12,)  # O preferences
        assert model['D'].shape == (12,)  # S prior

    def test_agent_with_start_position(self, mock_pymdp_agent):
        """Test agent creation with specific start position."""
        with patch('active_inference.environments.agents.factory.Agent', mock_pymdp_agent):
            agent, model, controls = build_gridworld_agent(
                n_rows=3,
                n_cols=4,
                reward_pos=(2, 3),
                punish_pos=(0, 3),
                start_pos=(1, 1)
            )

        # Check that D has a single 1 at start position
        expected_start_idx = 1 * 4 + 1  # (1, 1) -> index 5
        assert model['D'][expected_start_idx] == 1.0
        assert np.sum(model['D']) == 1.0  # Only one position has probability 1

    def test_agent_with_random_start(self, mock_pymdp_agent):
        """Test agent creation with uniform random start prior."""
        with patch('active_inference.environments.agents.factory.Agent', mock_pymdp_agent):
            agent, model, controls = build_gridworld_agent(
                n_rows=3,
                n_cols=4,
                reward_pos=(2, 3),
                punish_pos=(0, 3),
                start_pos=None  # Random start
            )

        # Check that D is uniform over non-terminal states
        n_states = 12
        terminal_indices = [3, 11]  # punish at (0,3), reward at (2,3)
        non_terminal_states = n_states - len(terminal_indices)

        expected_prob = 1.0 / non_terminal_states
        for i in range(n_states):
            if i in terminal_indices:
                assert model['D'][i] == 0.0
            else:
                assert abs(model['D'][i] - expected_prob) < 1e-6

    def test_preferences_setup(self, mock_pymdp_agent):
        """Test that preferences are set correctly."""
        with patch('active_inference.environments.agents.factory.Agent', mock_pymdp_agent):
            agent, model, controls = build_gridworld_agent(
                n_rows=3,
                n_cols=4,
                reward_pos=(2, 3),
                punish_pos=(0, 3),
                c_reward=2.5,
                c_punish=-1.5
            )

        # Check reward and punish preferences
        reward_idx = 2 * 4 + 3  # (2, 3) -> index 11
        punish_idx = 0 * 4 + 3  # (0, 3) -> index 3

        assert model['C'][reward_idx] == 2.5
        assert model['C'][punish_idx] == -1.5

        # Other positions should be 0
        for i in range(len(model['C'])):
            if i not in [reward_idx, punish_idx]:
                assert model['C'][i] == 0.0

    def test_transition_matrix(self, mock_pymdp_agent):
        """Test that transition matrix B is constructed correctly."""
        with patch('active_inference.environments.agents.factory.Agent', mock_pymdp_agent):
            agent, model, controls = build_gridworld_agent(
                n_rows=3,
                n_cols=4,
                reward_pos=(2, 3),
                punish_pos=(0, 3)
            )

        B = model['B']
        assert B.shape == (12, 12, 4)

        # Test some specific transitions
        # From (1, 1) = index 5, action RIGHT = 1 -> (1, 2) = index 6
        assert B[6, 5, 1] == 1.0

        # From (1, 1) = index 5, action UP = 0 -> (0, 1) = index 1
        assert B[1, 5, 0] == 1.0

        # From (0, 1) = index 1, action UP = 0 -> (0, 1) = index 1 (boundary)
        assert B[1, 1, 0] == 1.0

        # From (0, 0) = index 0, action LEFT = 3 -> (0, 0) = index 0 (boundary)
        assert B[0, 0, 3] == 1.0

    def test_observation_matrix(self, mock_pymdp_agent):
        """Test that observation matrix A is identity (fully observable)."""
        with patch('active_inference.environments.agents.factory.Agent', mock_pymdp_agent):
            agent, model, controls = build_gridworld_agent(
                n_rows=3,
                n_cols=4,
                reward_pos=(2, 3),
                punish_pos=(0, 3)
            )

        A = model['A']
        assert A.shape == (12, 12)

        # Should be identity matrix (perfect observation)
        expected_A = np.eye(12)
        np.testing.assert_array_equal(A, expected_A)

    def test_sophisticated_inference_setup(self, mock_pymdp_agent):
        """Test sophisticated inference flag handling."""
        mock_instance = Mock()
        mock_instance.infer_policies = Mock()
        mock_instance.sample_action = Mock(return_value=0)

        # Mock agent with sophisticated attribute
        mock_instance.sophisticated = False
        mock_pymdp_agent.return_value = mock_instance

        with patch('active_inference.environments.agents.factory.Agent', mock_pymdp_agent):
            agent, model, controls = build_gridworld_agent(
                n_rows=3,
                n_cols=4,
                reward_pos=(2, 3),
                punish_pos=(0, 3),
                sophisticated=True
            )

        # Check that sophisticated was set
        assert mock_instance.sophisticated == True

    def test_policy_inference_wrapper(self, mock_pymdp_agent):
        """Test the policy inference wrapper functionality."""
        mock_instance = Mock()
        mock_instance.infer_policies = Mock()
        mock_instance.sample_action = Mock(return_value=0)

        # Mock agent with sophisticated attribute
        mock_instance.sophisticated = False
        mock_pymdp_agent.return_value = mock_instance

        with patch('active_inference.environments.agents.factory.Agent', mock_pymdp_agent):
            agent, model, controls = build_gridworld_agent(
                n_rows=3,
                n_cols=4,
                reward_pos=(2, 3),
                punish_pos=(0, 3)
            )

        # Test inference wrapper
        controls['infer_policies']()
        mock_instance.infer_policies.assert_called_once()

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Invalid grid size
        with pytest.raises(ValueError, match="Grid dimensions must be positive"):
            build_gridworld_agent(
                n_rows=0,
                n_cols=4,
                reward_pos=(0, 3),
                punish_pos=(0, 2)
            )

        # Position out of bounds
        with pytest.raises(ValueError, match="out of bounds"):
            build_gridworld_agent(
                n_rows=3,
                n_cols=4,
                reward_pos=(3, 3),  # Out of bounds
                punish_pos=(0, 3)
            )

        # Same position for reward and punish
        with pytest.raises(ValueError, match="must be different"):
            build_gridworld_agent(
                n_rows=3,
                n_cols=4,
                reward_pos=(2, 3),
                punish_pos=(2, 3)  # Same as reward
            )

    def test_agent_construction_fallback(self):
        """Test agent construction with different API fallbacks."""
        # Mock Agent constructor that fails on first attempt but succeeds on second
        call_count = 0

        def mock_agent_constructor(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TypeError("First constructor failed")
            return Mock()

        with patch('active_inference.environments.agents.factory.Agent', side_effect=mock_agent_constructor):
            agent, model, controls = build_gridworld_agent(
                n_rows=3,
                n_cols=3,
                reward_pos=(2, 2),
                punish_pos=(0, 2)
            )

        assert agent is not None
        assert call_count == 2  # Should have tried twice

    def test_agent_construction_failure(self):
        """Test complete failure of agent construction."""
        with patch('active_inference.environments.agents.factory.Agent', side_effect=TypeError("All failed")):
            with pytest.raises(RuntimeError, match="Could not construct pymdp.Agent"):
                build_gridworld_agent(
                    n_rows=3,
                    n_cols=3,
                    reward_pos=(2, 2),
                    punish_pos=(0, 2)
                )

    def test_helper_functions(self, mock_pymdp_agent):
        """Test that helper functions are included in model."""
        with patch('active_inference.environments.agents.factory.Agent', mock_pymdp_agent):
            agent, model, controls = build_gridworld_agent(
                n_rows=3,
                n_cols=4,
                reward_pos=(2, 3),
                punish_pos=(0, 3)
            )

        # Check helper functions
        assert 'pos_to_idx' in model
        assert callable(model['pos_to_idx'])

        # Test helper function
        assert model['pos_to_idx'](1, 2) == 1 * 4 + 2  # 6

        # Test reward/punish indices
        assert model['reward_idx'] == 2 * 4 + 3  # 11
        assert model['punish_idx'] == 0 * 4 + 3  # 3
