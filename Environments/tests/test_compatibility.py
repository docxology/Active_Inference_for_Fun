"""
Tests for compatibility helper functions.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from ..utils.compatibility import (
    infer_states_compat,
    sample_action_compat,
    reset_agent_compat,
    infer_policies_compat,
)


class TestInferStatesCompat:
    """Test infer_states_compat function."""

    def test_infer_states_single_observation(self, mock_agent):
        """Test inference with single scalar observation."""
        obs = 5
        infer_states_compat(mock_agent, obs)

        mock_agent.infer_states.assert_called_with(obs)

    def test_infer_states_fallback_to_list(self, mock_agent):
        """Test fallback when direct call fails."""
        mock_agent.infer_states.side_effect = [Exception("Direct failed"), None]

        obs = 5
        infer_states_compat(mock_agent, obs)

        # Should have tried [obs] on second call
        assert mock_agent.infer_states.call_count == 2
        mock_agent.infer_states.assert_called_with([obs])

    def test_infer_states_triple_observation(self, mock_agent):
        """Test inference with tri-modal observation tuple."""
        obs_triplet = (1, 2, 3)
        infer_states_compat(mock_agent, obs_triplet)

        mock_agent.infer_states.assert_called_with(obs_triplet)

    def test_infer_states_fallback_for_tuple(self, mock_agent):
        """Test fallback for tuple observations."""
        mock_agent.infer_states.side_effect = [Exception("Tuple failed"), None]

        obs_triplet = (1, 2, 3)
        infer_states_compat(mock_agent, obs_triplet)

        # Should have tried list(obs_triplet) on second call
        assert mock_agent.infer_states.call_count == 2
        mock_agent.infer_states.assert_called_with([1, 2, 3])

    def test_infer_states_all_failures(self, mock_agent):
        """Test complete failure of all inference attempts."""
        mock_agent.infer_states.side_effect = Exception("All failed")

        with pytest.raises(RuntimeError, match="Could not infer states"):
            infer_states_compat(mock_agent, 5)


class TestSampleActionCompat:
    """Test sample_action_compat function."""

    def test_sample_action_integer_return(self, mock_agent):
        """Test when agent returns integer directly."""
        mock_agent.sample_action.return_value = 2
        action = sample_action_compat(mock_agent)

        assert action == 2
        assert isinstance(action, int)

    def test_sample_action_single_element_list(self, mock_agent):
        """Test when agent returns single-element list."""
        mock_agent.sample_action.return_value = [3]
        action = sample_action_compat(mock_agent)

        assert action == 3
        assert isinstance(action, int)

    def test_sample_action_tuple(self, mock_agent):
        """Test when agent returns tuple."""
        mock_agent.sample_action.return_value = (1,)
        action = sample_action_compat(mock_agent)

        assert action == 1
        assert isinstance(action, int)

    def test_sample_action_numpy_array(self, mock_agent):
        """Test when agent returns numpy array."""
        mock_agent.sample_action.return_value = np.array([2])
        action = sample_action_compat(mock_agent)

        assert action == 2
        assert isinstance(action, int)

    def test_sample_action_numpy_scalar(self, mock_agent):
        """Test when agent returns 0-d numpy array."""
        mock_agent.sample_action.return_value = np.array(1)
        action = sample_action_compat(mock_agent)

        assert action == 1
        assert isinstance(action, int)

    def test_sample_action_multiple_values_error(self, mock_agent):
        """Test error when agent returns multiple actions."""
        mock_agent.sample_action.return_value = [1, 2]

        with pytest.raises(ValueError, match="Expected single action"):
            sample_action_compat(mock_agent)

    def test_sample_action_unexpected_type_error(self, mock_agent):
        """Test error when agent returns unexpected type."""
        mock_agent.sample_action.return_value = "invalid"

        with pytest.raises(ValueError, match="Unexpected action type"):
            sample_action_compat(mock_agent)

    def test_sample_action_failure(self, mock_agent):
        """Test exception handling."""
        mock_agent.sample_action.side_effect = Exception("Sampling failed")

        with pytest.raises(RuntimeError, match="Action sampling failed"):
            sample_action_compat(mock_agent)


class TestResetAgentCompat:
    """Test reset_agent_compat function."""

    def test_reset_with_method(self, mock_agent):
        """Test successful reset when method exists."""
        mock_agent.reset = Mock()
        reset_agent_compat(mock_agent)

        mock_agent.reset.assert_called_once()

    def test_reset_without_method(self, mock_agent):
        """Test when agent has no reset method."""
        if hasattr(mock_agent, 'reset'):
            delattr(mock_agent, 'reset')

        # Should not raise error
        reset_agent_compat(mock_agent)

    def test_reset_method_fails(self, mock_agent):
        """Test when reset method exists but fails."""
        mock_agent.reset = Mock(side_effect=Exception("Reset failed"))

        # Should not raise error (soft failure)
        reset_agent_compat(mock_agent)
        mock_agent.reset.assert_called_once()


class TestInferPoliciesCompat:
    """Test infer_policies_compat function."""

    def test_infer_policies_with_controls_wrapper(self, mock_agent):
        """Test using factory-provided wrapper."""
        controls = {"infer_policies": Mock()}
        infer_policies_compat(mock_agent, sophisticated=True, controls=controls)

        controls["infer_policies"].assert_called_once()
        mock_agent.infer_policies.assert_not_called()

    def test_infer_policies_sophisticated_mode(self, mock_agent):
        """Test sophisticated inference with kwargs."""
        infer_policies_compat(mock_agent, sophisticated=True)

        # Should try sophisticated kwargs
        mock_agent.infer_policies.assert_called_once()
        call_kwargs = mock_agent.infer_policies.call_args[1]
        assert "mode" in call_kwargs or "method" in call_kwargs

    def test_infer_policies_sophisticated_fallback(self, mock_agent):
        """Test fallback when sophisticated kwargs fail."""
        mock_agent.infer_policies.side_effect = [TypeError("No mode"), TypeError("No method"), None]

        infer_policies_compat(mock_agent, sophisticated=True)

        assert mock_agent.infer_policies.call_count == 3

    def test_infer_policies_default_mode(self, mock_agent):
        """Test default inference (non-sophisticated)."""
        infer_policies_compat(mock_agent, sophisticated=False)

        mock_agent.infer_policies.assert_called_once_with()

    def test_infer_policies_failure(self, mock_agent):
        """Test complete failure of policy inference."""
        mock_agent.infer_policies.side_effect = Exception("All failed")

        with pytest.raises(RuntimeError, match="Policy inference failed"):
            infer_policies_compat(mock_agent, sophisticated=False)
