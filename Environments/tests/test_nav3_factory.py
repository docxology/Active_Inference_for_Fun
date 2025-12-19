"""
Comprehensive tests for tri-modal navigation agent factory.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from ..agents.nav3_factory import build_trimodal_nav_agent


class TestBuildTrimodalNavAgent:
    """Test build_trimodal_nav_agent function."""

    def test_basic_agent_creation(self, mock_pymdp_agent):
        """Test basic tri-modal agent creation."""
        with patch('active_inference.environments.agents.nav3_factory.Agent', mock_pymdp_agent):
            agent, model, controls = build_trimodal_nav_agent(
                n_rows=4,
                n_cols=4,
                reward_pos=(3, 3),
                punish_pos=(0, 3)
            )

        assert agent is not None
        assert isinstance(model, dict)
        assert isinstance(controls, dict)

        # Check model structure
        assert 'A' in model
        assert 'B' in model
        assert 'C' in model
        assert 'D' in model
        assert 'spaces' in model
        assert 'semantics' in model

    def test_matrix_shapes(self, mock_pymdp_agent):
        """Test that matrices have correct shapes for 4x4 grid."""
        with patch('active_inference.environments.agents.nav3_factory.Agent', mock_pymdp_agent):
            agent, model, controls = build_trimodal_nav_agent(
                n_rows=4,
                n_cols=4,
                reward_pos=(3, 3),
                punish_pos=(0, 3)
            )

        # Check matrix shapes
        A = model['A']
        B = model['B']
        C = model['C']
        D = model['D']

        n_states = 4 * 4 * 4  # rows × cols × orientations = 64

        assert len(A) == 3  # Three modalities
        assert A[0].shape == (7, n_states)  # Distance (max range + 1)
        assert A[1].shape == (3, n_states)  # Terminal type
        assert A[2].shape == (4, n_states)  # Cell type

        assert len(B) == 1  # Single factor
        assert B[0].shape == (n_states, n_states, 3)  # States × States × Actions

        assert len(C) == 3  # Three modalities
        assert C[0].shape == (7,)  # Distance preferences
        assert C[1].shape == (3,)  # Terminal preferences
        assert C[2].shape == (4,)  # Cell preferences

        assert len(D) == 1  # Single factor
        assert D[0].shape == (n_states,)  # State prior

    def test_observation_matrix_modality_1(self, mock_pymdp_agent):
        """Test distance observation matrix (modality 1)."""
        with patch('active_inference.environments.agents.nav3_factory.Agent', mock_pymdp_agent):
            agent, model, controls = build_trimodal_nav_agent(
                n_rows=4,
                n_cols=4,
                reward_pos=(3, 3),
                punish_pos=(0, 3)
            )

        A1 = model['A'][0]  # Distance modality
        n_states = 64

        # Test a specific state: position (1, 1), facing North (ori=0)
        # State index = (1 * 4 + 1) * 4 + 0 = 20
        state_idx = 1 * 4 * 4 + 1 * 4 + 0  # pos (1,1), ori 0

        # Facing North from (1, 1), distance to northern edge should be 1
        # (position 0 in distance modality corresponds to distance 0)
        assert A1[1, state_idx] == 1.0  # Distance 1

    def test_observation_matrix_modality_2(self, mock_pymdp_agent):
        """Test terminal type observation matrix (modality 2)."""
        with patch('active_inference.environments.agents.nav3_factory.Agent', mock_pymdp_agent):
            agent, model, controls = build_trimodal_nav_agent(
                n_rows=4,
                n_cols=4,
                reward_pos=(3, 3),
                punish_pos=(0, 3)
            )

        A2 = model['A'][1]  # Terminal type modality

        # Test facing toward reward: pos (3, 2), ori East (1)
        # State index = (3 * 4 + 2) * 4 + 1 = 50
        state_idx = 3 * 4 * 4 + 2 * 4 + 1

        # Should see GREEN (index 2) terminal
        assert A2[2, state_idx] == 1.0  # GREEN

    def test_observation_matrix_modality_3(self, mock_pymdp_agent):
        """Test cell type observation matrix (modality 3)."""
        with patch('active_inference.environments.agents.nav3_factory.Agent', mock_pymdp_agent):
            agent, model, controls = build_trimodal_nav_agent(
                n_rows=4,
                n_cols=4,
                reward_pos=(3, 3),
                punish_pos=(0, 3)
            )

        A3 = model['A'][2]  # Cell type modality

        # Test at reward position: pos (3, 3), any orientation
        # State indices = (3 * 4 + 3) * 4 + ori = 60 + ori
        for ori in range(4):
            state_idx = 3 * 4 * 4 + 3 * 4 + ori
            assert A3[3, state_idx] == 1.0  # GREEN (reward cell)

    def test_transition_matrix(self, mock_pymdp_agent):
        """Test transition matrix B."""
        with patch('active_inference.environments.agents.nav3_factory.Agent', mock_pymdp_agent):
            agent, model, controls = build_trimodal_nav_agent(
                n_rows=4,
                n_cols=4,
                reward_pos=(3, 3),
                punish_pos=(0, 3)
            )

        B = model['B'][0]

        # Test forward movement: from (1, 1, North) -> (0, 1, North)
        # State 1*16 + 1*4 + 0 = 20, action FWD=0 -> state 0*16 + 1*4 + 0 = 4
        assert B[4, 20, 0] == 1.0

        # Test turn left: from (1, 1, North) -> (1, 1, West)
        # State 20, action TURN_L=1 -> state 1*16 + 1*4 + 3 = 23
        assert B[23, 20, 1] == 1.0

        # Test turn right: from (1, 1, North) -> (1, 1, East)
        # State 20, action TURN_R=2 -> state 1*16 + 1*4 + 1 = 21
        assert B[21, 20, 2] == 1.0

    def test_preferences(self, mock_pymdp_agent):
        """Test preference matrices C."""
        with patch('active_inference.environments.agents.nav3_factory.Agent', mock_pymdp_agent):
            agent, model, controls = build_trimodal_nav_agent(
                n_rows=4,
                n_cols=4,
                reward_pos=(3, 3),
                punish_pos=(0, 3),
                c_green=2.0,
                c_red=-1.5
            )

        C1, C2, C3 = model['C']

        # C1 (distances) should be neutral (zeros)
        assert np.all(C1 == 0.0)

        # C2 (terminal types) should be neutral (zeros)
        assert np.all(C2 == 0.0)

        # C3 (cell types) should have preferences
        assert C3[3] == 2.0   # GREEN preference
        assert C3[2] == -1.5  # RED preference
        assert C3[0] == 0.0   # EMPTY neutral
        assert C3[1] == 0.0   # EDGE neutral

    def test_prior_distribution(self, mock_pymdp_agent):
        """Test prior distribution D."""
        with patch('active_inference.environments.agents.nav3_factory.Agent', mock_pymdp_agent):
            agent, model, controls = build_trimodal_nav_agent(
                n_rows=4,
                n_cols=4,
                reward_pos=(3, 3),
                punish_pos=(0, 3),
                start_pos=(1, 1),
                start_ori="N"
            )

        D = model['D'][0]
        n_states = 64

        # Should have single 1.0 at start state
        expected_state = 1 * 4 * 4 + 1 * 4 + 0  # pos (1,1), ori North (0)
        assert D[expected_state] == 1.0
        assert np.sum(D) == 1.0

    def test_random_start_prior(self, mock_pymdp_agent):
        """Test uniform prior for random starts."""
        with patch('active_inference.environments.agents.nav3_factory.Agent', mock_pymdp_agent):
            agent, model, controls = build_trimodal_nav_agent(
                n_rows=4,
                n_cols=4,
                reward_pos=(3, 3),
                punish_pos=(0, 3),
                start_pos=None  # Random start
            )

        D = model['D'][0]
        n_states = 64

        # Should be uniform over all states
        expected_prob = 1.0 / n_states
        assert np.allclose(D, expected_prob)

    def test_observation_noise(self, mock_pymdp_agent):
        """Test observation noise functionality."""
        with patch('active_inference.environments.agents.nav3_factory.Agent', mock_pymdp_agent):
            agent, model, controls = build_trimodal_nav_agent(
                n_rows=4,
                n_cols=4,
                reward_pos=(3, 3),
                punish_pos=(0, 3),
                a_obs_noise=0.1
            )

        A1, A2, A3 = model['A']

        # With noise, columns should not sum to exactly 1
        # (they get renormalized after adding noise)
        assert not np.allclose(A1.sum(axis=0), 1.0)
        assert not np.allclose(A2.sum(axis=0), 1.0)
        assert not np.allclose(A3.sum(axis=0), 1.0)

        # But they should still sum to approximately 1 (after renormalization)
        assert np.allclose(A1.sum(axis=0), 1.0, atol=1e-10)
        assert np.allclose(A2.sum(axis=0), 1.0, atol=1e-10)
        assert np.allclose(A3.sum(axis=0), 1.0, atol=1e-10)

    def test_model_noise(self, mock_pymdp_agent):
        """Test model noise in transitions."""
        with patch('active_inference.environments.agents.nav3_factory.Agent', mock_pymdp_agent):
            agent, model, controls = build_trimodal_nav_agent(
                n_rows=4,
                n_cols=4,
                reward_pos=(3, 3),
                punish_pos=(0, 3),
                b_model_noise=0.1
            )

        B = model['B'][0]

        # With noise, transitions should not be exactly deterministic
        # Check that some transitions have non-zero probability where they shouldn't
        assert B[4, 20, 0] < 1.0  # Forward should be less than 1

        # But rows should still sum to 1 (properly normalized)
        assert np.allclose(B.sum(axis=0), 1.0, atol=1e-10)

    def test_spaces_metadata(self, mock_pymdp_agent):
        """Test spaces metadata in model."""
        with patch('active_inference.environments.agents.nav3_factory.Agent', mock_pymdp_agent):
            agent, model, controls = build_trimodal_nav_agent(
                n_rows=4,
                n_cols=4,
                reward_pos=(3, 3),
                punish_pos=(0, 3)
            )

        spaces = model['spaces']
        assert spaces['S'] == 64  # 4×4×4 states
        assert spaces['O1'] == 7  # Distance observations
        assert spaces['O2'] == 3  # Terminal type observations
        assert spaces['O3'] == 4  # Cell type observations
        assert spaces['U'] == 3   # Actions

    def test_semantics_metadata(self, mock_pymdp_agent):
        """Test semantics metadata in model."""
        with patch('active_inference.environments.agents.nav3_factory.Agent', mock_pymdp_agent):
            agent, model, controls = build_trimodal_nav_agent(
                n_rows=4,
                n_cols=4,
                reward_pos=(3, 3),
                punish_pos=(0, 3)
            )

        semantics = model['semantics']
        assert semantics['modalities'] == ['distance', 'terminal_class', 'current_class']
        assert semantics['actions'] == ['forward', 'turn_left', 'turn_right']
        assert semantics['orientations'] == ['N', 'E', 'S', 'W']

        # Check class mappings
        classes = semantics['classes']
        assert classes['current_class'] == ['EMPTY', 'EDGE', 'RED', 'GREEN']
        assert classes['terminal_class'] == ['EDGE', 'RED', 'GREEN']

    def test_policy_inference_wrapper(self, mock_pymdp_agent):
        """Test policy inference wrapper."""
        mock_instance = Mock()
        mock_instance.infer_policies = Mock()
        mock_pymdp_agent.return_value = mock_instance

        with patch('active_inference.environments.agents.nav3_factory.Agent', mock_pymdp_agent):
            agent, model, controls = build_trimodal_nav_agent(
                n_rows=4,
                n_cols=4,
                reward_pos=(3, 3),
                punish_pos=(0, 3)
            )

        # Test wrapper exists and is callable
        assert callable(controls['infer_policies'])
        controls['infer_policies']()
        mock_instance.infer_policies.assert_called_once()

    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Invalid grid size
        with pytest.raises(ValueError, match="Grid must be at least 2×2"):
            build_trimodal_nav_agent(
                n_rows=1,
                n_cols=4,
                reward_pos=(0, 3),
                punish_pos=(0, 2)
            )

        # Position out of bounds
        with pytest.raises(ValueError, match="out of bounds"):
            build_trimodal_nav_agent(
                n_rows=4,
                n_cols=4,
                reward_pos=(4, 3),  # Out of bounds
                punish_pos=(0, 3)
            )

        # Invalid orientation
        with pytest.raises(ValueError, match="Invalid start orientation"):
            build_trimodal_nav_agent(
                n_rows=4,
                n_cols=4,
                reward_pos=(3, 3),
                punish_pos=(0, 3),
                start_ori="INVALID"
            )

    def test_agent_construction_failure(self):
        """Test failure when Agent cannot be constructed."""
        with patch('active_inference.environments.agents.nav3_factory.Agent', side_effect=Exception("Construction failed")):
            with pytest.raises(Exception):  # Should propagate the construction error
                build_trimodal_nav_agent(
                    n_rows=4,
                    n_cols=4,
                    reward_pos=(3, 3),
                    punish_pos=(0, 3)
                )
