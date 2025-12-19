"""
Comprehensive tests for GridWorld environment.
"""

import pytest
import numpy as np
from unittest.mock import patch

from ..core import GridWorld


class TestGridWorldInitialization:
    """Test GridWorld initialization and configuration."""

    def test_basic_initialization(self):
        """Test basic GridWorld creation with valid parameters."""
        env = GridWorld(
            n_rows=3,
            n_cols=4,
            reward_pos=(2, 3),
            punish_pos=(0, 3),
            render_mode=None
        )

        assert env.n_rows == 3
        assert env.n_cols == 4
        assert env.reward_pos == (2, 3)
        assert env.punish_pos == (0, 3)
        assert env.action_space.n == 4  # UP, RIGHT, DOWN, LEFT
        assert env.observation_space.n == 12  # 3 * 4

    def test_invalid_grid_size(self):
        """Test that invalid grid sizes raise errors."""
        with pytest.raises(ValueError, match="Grid must be at least 2×2"):
            GridWorld(n_rows=1, n_cols=3, reward_pos=(0, 2), punish_pos=(0, 0))

        with pytest.raises(ValueError, match="Grid must be at least 2×2"):
            GridWorld(n_rows=3, n_cols=1, reward_pos=(2, 0), punish_pos=(0, 0))

    def test_invalid_positions(self):
        """Test that invalid positions raise errors."""
        # Position out of bounds
        with pytest.raises(ValueError, match="out of bounds"):
            GridWorld(n_rows=3, n_cols=3, reward_pos=(3, 2), punish_pos=(0, 2))

        # Same position for reward and punish
        with pytest.raises(ValueError, match="must be different"):
            GridWorld(n_rows=3, n_cols=3, reward_pos=(2, 2), punish_pos=(2, 2))

    def test_custom_parameters(self):
        """Test custom parameter initialization."""
        env = GridWorld(
            n_rows=4,
            n_cols=5,
            reward_pos=(3, 4),
            punish_pos=(0, 4),
            start_pos=(1, 1),
            step_cost=-0.1,
            reward=2.0,
            punish=-3.0,
            max_steps=50,
            slip_prob=0.2
        )

        assert env.step_cost == -0.1
        assert env.r_reward == 2.0
        assert env.r_punish == -3.0
        assert env.max_steps == 50
        assert env.slip_prob == 0.2


class TestGridWorldReset:
    """Test GridWorld reset functionality."""

    def test_reset_with_fixed_start(self, simple_gridworld):
        """Test reset with fixed start position."""
        obs, info = simple_gridworld.reset(seed=42)

        # Should start at configured position (1, 1) -> index 4
        assert obs == 4  # (1 * 3) + 1
        assert info["pos"] == (1, 1)

    def test_reset_with_random_start(self):
        """Test reset with random start position."""
        env = GridWorld(
            n_rows=3,
            n_cols=3,
            reward_pos=(2, 2),
            punish_pos=(0, 2),
            start_pos=None,  # Random start
            render_mode=None
        )

        obs, info = env.reset(seed=42)
        # Should be at a valid non-terminal position
        assert obs != 2  # Not punish position (0, 2) -> index 2
        assert obs != 8  # Not reward position (2, 2) -> index 8
        assert 0 <= obs < 9

    def test_reset_reproducibility(self, simple_gridworld):
        """Test that reset is reproducible with same seed."""
        obs1, info1 = simple_gridworld.reset(seed=123)
        obs2, info2 = simple_gridworld.reset(seed=123)

        assert obs1 == obs2
        assert info1["pos"] == info2["pos"]


class TestGridWorldStep:
    """Test GridWorld step functionality."""

    def test_basic_movement(self, simple_gridworld):
        """Test basic movement actions."""
        simple_gridworld.reset(seed=42)

        # Move right from (1, 1) -> (1, 2)
        obs, reward, terminated, truncated, info = simple_gridworld.step(1)  # RIGHT
        assert obs == 5  # (1 * 3) + 2
        assert reward == 0.0  # No step cost
        assert not terminated
        assert not truncated
        assert info["pos"] == (1, 2)

        # Move down from (1, 2) -> (2, 2)
        obs, reward, terminated, truncated, info = simple_gridworld.step(2)  # DOWN
        assert obs == 8  # (2 * 3) + 2
        assert reward == 1.0  # Reward!
        assert terminated
        assert not truncated
        assert info["pos"] == (2, 2)

    def test_boundary_movement(self, simple_gridworld):
        """Test movement at boundaries."""
        simple_gridworld.reset(seed=42)

        # Try to move up from top row - should stay in place
        simple_gridworld.pos = (0, 1)  # Manually set to top row
        obs, reward, terminated, truncated, info = simple_gridworld.step(0)  # UP
        assert info["pos"] == (0, 1)  # Should not move

        # Try to move left from left edge
        obs, reward, terminated, truncated, info = simple_gridworld.step(3)  # LEFT
        assert info["pos"] == (0, 1)  # Should not move

    def test_terminal_states(self, simple_gridworld):
        """Test reaching terminal states."""
        simple_gridworld.reset(seed=42)

        # Move to punish position (0, 2)
        simple_gridworld.pos = (0, 1)
        obs, reward, terminated, truncated, info = simple_gridworld.step(1)  # RIGHT
        assert obs == 2  # (0 * 3) + 2
        assert reward == -1.0  # Punishment
        assert terminated
        assert not truncated
        assert info["pos"] == (0, 2)

    def test_step_cost(self):
        """Test step cost functionality."""
        env = GridWorld(
            n_rows=3,
            n_cols=3,
            reward_pos=(2, 2),
            punish_pos=(0, 2),
            step_cost=-0.5,
            render_mode=None
        )
        env.reset(seed=42)

        obs, reward, terminated, truncated, info = env.step(1)  # Any move
        assert reward == -0.5  # Step cost
        assert not terminated

    def test_slip_probability(self):
        """Test action slipping functionality."""
        env = GridWorld(
            n_rows=3,
            n_cols=3,
            reward_pos=(2, 2),
            punish_pos=(0, 2),
            slip_prob=1.0,  # Always slip
            render_mode=None
        )

        env.reset(seed=42)
        original_pos = env.pos

        # Action should always slip to random action
        obs, reward, terminated, truncated, info = env.step(0)  # Try to move UP
        # Position should change (unless slipped to same action and hit boundary)
        # Hard to test deterministically, but ensure valid position
        assert info["pos"][0] >= 0 and info["pos"][0] < 3
        assert info["pos"][1] >= 0 and info["pos"][1] < 3

    def test_max_steps(self, simple_gridworld):
        """Test episode truncation at max steps."""
        simple_gridworld.reset(seed=42)

        # Take max_steps actions
        for _ in range(simple_gridworld.max_steps):
            obs, reward, terminated, truncated, info = simple_gridworld.step(0)
            if terminated or truncated:
                break

        # Should eventually truncate
        assert truncated or terminated

    def test_invalid_action(self, simple_gridworld):
        """Test that invalid actions raise errors."""
        simple_gridworld.reset(seed=42)

        with pytest.raises(ValueError, match="Invalid action"):
            simple_gridworld.step(4)  # Invalid action

        with pytest.raises(ValueError, match="Invalid action"):
            simple_gridworld.step(-1)  # Invalid action


class TestGridWorldRendering:
    """Test GridWorld rendering functionality."""

    def test_text_rendering(self, simple_gridworld):
        """Test text-based rendering."""
        simple_gridworld.reset(seed=42)

        render_output = simple_gridworld.render()
        assert isinstance(render_output, str)
        assert "A" in render_output  # Agent
        assert "G" in render_output  # Goal/Reward
        assert "X" in render_output  # Punish

    def test_rgb_rendering(self, simple_gridworld):
        """Test RGB array rendering."""
        simple_gridworld.reset(seed=42)

        rgb_array = simple_gridworld.render()
        assert isinstance(rgb_array, np.ndarray)
        assert rgb_array.shape[2] == 3  # RGB channels
        assert rgb_array.dtype == np.uint8

    @patch('matplotlib.pyplot.ion')
    def test_human_rendering(self, mock_ion):
        """Test human rendering mode (mocked)."""
        env = GridWorld(
            n_rows=3,
            n_cols=3,
            reward_pos=(2, 2),
            punish_pos=(0, 2),
            render_mode="human"
        )
        env.reset(seed=42)

        result = env.render()
        assert result is None  # Human mode returns None
        mock_ion.assert_called_once()


class TestGridWorldUtilities:
    """Test GridWorld utility methods."""

    def test_position_conversion(self, simple_gridworld):
        """Test position to index and index to position conversion."""
        # Test _pos_to_idx
        assert simple_gridworld._pos_to_idx((0, 0)) == 0
        assert simple_gridworld._pos_to_idx((0, 1)) == 1
        assert simple_gridworld._pos_to_idx((1, 0)) == 3
        assert simple_gridworld._pos_to_idx((2, 2)) == 8

        # Test _idx_to_pos
        assert simple_gridworld._idx_to_pos(0) == (0, 0)
        assert simple_gridworld._idx_to_pos(1) == (0, 1)
        assert simple_gridworld._idx_to_pos(3) == (1, 0)
        assert simple_gridworld._idx_to_pos(8) == (2, 2)

    def test_position_validation(self, simple_gridworld):
        """Test position validation."""
        # Valid positions
        assert simple_gridworld._validate_pos((0, 0)) == (0, 0)
        assert simple_gridworld._validate_pos((2, 2)) == (2, 2)

        # Invalid positions
        with pytest.raises(AssertionError):
            simple_gridworld._validate_pos((-1, 0))

        with pytest.raises(AssertionError):
            simple_gridworld._validate_pos((3, 0))

        with pytest.raises(AssertionError):
            simple_gridworld._validate_pos((0, 3))


class TestGridWorldIntegration:
    """Integration tests for complete GridWorld episodes."""

    def test_shortest_path_episode(self, simple_gridworld):
        """Test a complete episode following optimal path."""
        obs, info = simple_gridworld.reset(seed=42)

        # From (1, 1) to (2, 2): should take RIGHT then DOWN
        obs, reward, terminated, truncated, info = simple_gridworld.step(1)  # RIGHT
        assert not terminated

        obs, reward, terminated, truncated, info = simple_gridworld.step(2)  # DOWN
        assert terminated
        assert reward == 1.0
        assert info["pos"] == (2, 2)

    def test_punishment_episode(self, simple_gridworld):
        """Test episode ending in punishment."""
        obs, info = simple_gridworld.reset(seed=42)

        # From (1, 1) to (0, 2): should take UP then RIGHT
        obs, reward, terminated, truncated, info = simple_gridworld.step(0)  # UP
        assert not terminated

        obs, reward, terminated, truncated, info = simple_gridworld.step(1)  # RIGHT
        assert terminated
        assert reward == -1.0
        assert info["pos"] == (0, 2)
