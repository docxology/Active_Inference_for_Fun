"""
Comprehensive tests for OrientedTriModalGrid environment.
"""

import pytest
import numpy as np
from unittest.mock import patch

from ..core import OrientedTriModalGrid


class TestOrientedGridInitialization:
    """Test OrientedTriModalGrid initialization and configuration."""

    def test_basic_initialization(self):
        """Test basic OrientedTriModalGrid creation."""
        env = OrientedTriModalGrid(
            n_rows=4,
            n_cols=4,
            reward_pos=(3, 3),
            punish_pos=(0, 3),
            start_pos=(1, 1),
            start_ori="N",
            render_mode=None
        )

        assert env.n_rows == 4
        assert env.n_cols == 4
        assert env.reward_pos == (3, 3)
        assert env.punish_pos == (0, 3)
        assert env.start_pos == (1, 1)
        assert env.start_ori == "N"
        assert env.action_space.n == 3  # FWD, TURN_L, TURN_R

        # Check observation space (3 modalities)
        assert len(env.observation_space.spaces) == 3
        assert env.observation_space.spaces[0].n == 7  # max_range + 1 (4+3-1+1=7)
        assert env.observation_space.spaces[1].n == 3   # EDGE, RED, GREEN
        assert env.observation_space.spaces[2].n == 4   # EMPTY, EDGE, RED, GREEN

    def test_invalid_grid_size(self):
        """Test that invalid grid sizes raise errors."""
        with pytest.raises(ValueError, match="Grid must be at least 2×2"):
            OrientedTriModalGrid(n_rows=1, n_cols=3, reward_pos=(0, 2), punish_pos=(0, 0))

    def test_invalid_positions(self):
        """Test that invalid positions raise errors."""
        # Position out of bounds
        with pytest.raises(ValueError, match="out of bounds"):
            OrientedTriModalGrid(n_rows=3, n_cols=3, reward_pos=(3, 2), punish_pos=(0, 2))

        # Same position for reward and punish
        with pytest.raises(ValueError, match="must be different"):
            OrientedTriModalGrid(n_rows=3, n_cols=3, reward_pos=(2, 2), punish_pos=(2, 2))

    def test_invalid_orientation(self):
        """Test invalid start orientation."""
        with pytest.raises(ValueError, match="Invalid start orientation"):
            OrientedTriModalGrid(n_rows=3, n_cols=3, reward_pos=(2, 2), punish_pos=(0, 2),
                               start_ori="INVALID")


class TestOrientedGridReset:
    """Test OrientedTriModalGrid reset functionality."""

    def test_reset_with_fixed_start(self, oriented_grid):
        """Test reset with fixed start position and orientation."""
        obs, info = oriented_grid.reset(seed=42)

        assert len(obs) == 3  # Tri-modal observation
        assert isinstance(obs, tuple)
        # Should be at configured position (1, 1) facing North
        assert oriented_grid.pos == (1, 1)
        assert oriented_grid.ori == 0  # North = 0

    def test_reset_with_random_start(self):
        """Test reset with random start position."""
        env = OrientedTriModalGrid(
            n_rows=4,
            n_cols=4,
            reward_pos=(3, 3),
            punish_pos=(0, 3),
            start_pos=None,  # Random start
            start_ori="E",
            render_mode=None
        )

        obs, info = env.reset(seed=42)
        # Should be at a valid position
        assert 0 <= env.pos[0] < 4
        assert 0 <= env.pos[1] < 4
        assert env.ori == 1  # East = 1

    def test_reset_reproducibility(self, oriented_grid):
        """Test that reset is reproducible."""
        obs1, info1 = oriented_grid.reset(seed=123)
        obs2, info2 = oriented_grid.reset(seed=123)

        assert obs1 == obs2


class TestOrientedGridStep:
    """Test OrientedTriModalGrid step functionality."""

    def test_forward_movement(self, oriented_grid):
        """Test forward movement in different orientations."""
        oriented_grid.reset(seed=42)

        # Facing North - forward should move up (decrease row)
        original_pos = oriented_grid.pos
        obs, reward, terminated, truncated, info = oriented_grid.step(0)  # FWD
        assert oriented_grid.pos[0] == original_pos[0] - 1  # Row decreases
        assert oriented_grid.pos[1] == original_pos[1]      # Col stays same

        # Turn right to face East
        obs, reward, terminated, truncated, info = oriented_grid.step(2)  # TURN_R
        assert oriented_grid.ori == 1  # East

        # Forward should now move right (increase col)
        original_pos = oriented_grid.pos
        obs, reward, terminated, truncated, info = oriented_grid.step(0)  # FWD
        assert oriented_grid.pos[0] == original_pos[0]      # Row stays same
        assert oriented_grid.pos[1] == original_pos[1] + 1  # Col increases

    def test_rotation(self, oriented_grid):
        """Test rotation actions."""
        oriented_grid.reset(seed=42)
        original_ori = oriented_grid.ori

        # Turn left
        obs, reward, terminated, truncated, info = oriented_grid.step(1)  # TURN_L
        assert oriented_grid.ori == (original_ori - 1) % 4

        # Turn right
        obs, reward, terminated, truncated, info = oriented_grid.step(2)  # TURN_R
        assert oriented_grid.ori == original_ori  # Back to original

    def test_boundary_movement(self, oriented_grid):
        """Test movement at boundaries."""
        oriented_grid.reset(seed=42)

        # Move to north edge facing north
        oriented_grid.pos = (0, 1)
        oriented_grid.ori = 0  # North
        original_pos = oriented_grid.pos

        # Try to move forward from north edge - should stay put
        obs, reward, terminated, truncated, info = oriented_grid.step(0)  # FWD
        assert oriented_grid.pos == original_pos

    def test_terminal_states(self, oriented_grid):
        """Test reaching terminal states."""
        oriented_grid.reset(seed=42)

        # Move to reward position
        oriented_grid.pos = (3, 2)
        oriented_grid.ori = 1  # East

        # Move east to (3, 3) - reward position
        obs, reward, terminated, truncated, info = oriented_grid.step(0)  # FWD
        assert terminated
        assert reward == 1.0
        assert oriented_grid.pos == (3, 3)

    def test_max_steps(self, oriented_grid):
        """Test episode truncation at max steps."""
        oriented_grid.reset(seed=42)

        # Take max_steps actions
        for _ in range(oriented_grid.max_steps):
            obs, reward, terminated, truncated, info = oriented_grid.step(0)  # Always forward
            if terminated or truncated:
                break

        # Should eventually truncate
        assert truncated or terminated

    def test_invalid_action(self, oriented_grid):
        """Test invalid actions."""
        oriented_grid.reset(seed=42)

        with pytest.raises(ValueError, match="Invalid action"):
            oriented_grid.step(3)  # Invalid action

        with pytest.raises(ValueError, match="Invalid action"):
            oriented_grid.step(-1)  # Invalid action


class TestOrientedGridObservations:
    """Test OrientedTriModalGrid observation functionality."""

    def test_observation_structure(self, oriented_grid):
        """Test that observations have correct structure."""
        obs, info = oriented_grid.reset(seed=42)

        assert len(obs) == 3
        assert isinstance(obs, tuple)

        # Check ranges
        distance, terminal_type, cell_type = obs
        assert 0 <= distance <= 6  # Max range for 4x4 grid
        assert 0 <= terminal_type <= 2  # EDGE, RED, GREEN
        assert 0 <= cell_type <= 3  # EMPTY, EDGE, RED, GREEN

    def test_distance_calculation(self, oriented_grid):
        """Test distance observation calculation."""
        oriented_grid.reset(seed=42)

        # Face north from (1, 1) - should see distance to northern edge
        oriented_grid.ori = 0  # North
        obs = oriented_grid._get_obs()
        distance, terminal_type, cell_type = obs

        # Distance to northern edge from (1, 1) facing north
        # Should see the edge immediately (distance 0)
        assert distance == 0
        assert terminal_type == 1  # EDGE

    def test_terminal_type_detection(self, oriented_grid):
        """Test terminal type observation."""
        oriented_grid.reset(seed=42)

        # Position agent to face reward
        oriented_grid.pos = (3, 2)
        oriented_grid.ori = 1  # East (toward reward at 3, 3)

        obs = oriented_grid._get_obs()
        distance, terminal_type, cell_type = obs

        assert distance == 1  # One step to reward
        assert terminal_type == 2  # GREEN (reward)

    def test_cell_type_observation(self, oriented_grid):
        """Test current cell type observation."""
        oriented_grid.reset(seed=42)

        # At reward position
        oriented_grid.pos = (3, 3)
        obs = oriented_grid._get_obs()
        distance, terminal_type, cell_type = obs

        assert cell_type == 3  # GREEN

        # At punish position
        oriented_grid.pos = (0, 3)
        obs = oriented_grid._get_obs()
        distance, terminal_type, cell_type = obs

        assert cell_type == 2  # RED


class TestOrientedGridRendering:
    """Test OrientedTriModalGrid rendering functionality."""

    @patch('matplotlib.pyplot.ion')
    def test_human_rendering(self, mock_ion, oriented_grid):
        """Test human rendering mode."""
        oriented_grid.reset(seed=42)

        result = oriented_grid.render()
        assert result is None  # Human mode returns None
        mock_ion.assert_called_once()

    def test_rgb_rendering(self, oriented_grid):
        """Test RGB array rendering."""
        oriented_grid.reset(seed=42)

        rgb_array = oriented_grid.render()
        assert isinstance(rgb_array, np.ndarray)
        assert rgb_array.shape[2] == 3  # RGB channels
        assert rgb_array.dtype == np.uint8


class TestOrientedGridUtilities:
    """Test OrientedTriModalGrid utility methods."""

    def test_class_detection(self, oriented_grid):
        """Test cell class detection."""
        # Test various positions
        assert oriented_grid._class_of((3, 3)) == 3  # GREEN (reward)
        assert oriented_grid._class_of((0, 3)) == 2  # RED (punish)
        assert oriented_grid._class_of((0, 0)) == 1  # EDGE (boundary)
        assert oriented_grid._class_of((1, 1)) == 0  # EMPTY

    def test_raycast_terminal(self, oriented_grid):
        """Test raycast terminal detection."""
        # Face north from (1, 1) - should hit northern edge
        distance, terminal = oriented_grid._raycast_terminal((1, 1), 0)  # North
        assert distance == 1  # One step to edge
        assert terminal == 1   # EDGE

        # Face east from (3, 2) - should hit reward
        distance, terminal = oriented_grid._raycast_terminal((3, 2), 1)  # East
        assert distance == 1  # One step to reward
        assert terminal == 3   # GREEN


class TestOrientedGridIntegration:
    """Integration tests for complete episodes."""

    def test_navigation_to_reward(self, oriented_grid):
        """Test navigating to reward position."""
        obs, info = oriented_grid.reset(seed=42)

        # Simple sequence: turn right, forward to reward
        # Start at (1, 1) facing North
        obs, reward, terminated, truncated, info = oriented_grid.step(2)  # TURN_R -> East
        assert not terminated

        obs, reward, terminated, truncated, info = oriented_grid.step(0)  # FWD
        assert not terminated

        obs, reward, terminated, truncated, info = oriented_grid.step(0)  # FWD
        assert not terminated

        obs, reward, terminated, truncated, info = oriented_grid.step(2)  # TURN_R -> South
        assert not terminated

        obs, reward, terminated, truncated, info = oriented_grid.step(0)  # FWD -> (3, 3)
        assert terminated
        assert reward == 1.0
        assert oriented_grid.pos == (3, 3)

    def test_punishment_episode(self, oriented_grid):
        """Test episode ending in punishment."""
        obs, info = oriented_grid.reset(seed=42)

        # Navigate to punish position (0, 3)
        # From (1, 1) facing North: turn right, forward, turn left, forward
        obs, reward, terminated, truncated, info = oriented_grid.step(2)  # TURN_R -> East
        obs, reward, terminated, truncated, info = oriented_grid.step(0)  # FWD -> (1, 2)
        obs, reward, terminated, truncated, info = oriented_grid.step(0)  # FWD -> (1, 3)
        obs, reward, terminated, truncated, info = oriented_grid.step(1)  # TURN_L -> North
        obs, reward, terminated, truncated, info = oriented_grid.step(0)  # FWD -> (0, 3)

        assert terminated
        assert reward == -1.0
        assert oriented_grid.pos == (0, 3)
