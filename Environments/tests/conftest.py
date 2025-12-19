"""
Pytest configuration and fixtures for Active Inference environments tests.
"""

import pytest
import numpy as np

# Check what dependencies are available
GYMNASIUM_AVAILABLE = False
PYMDP_AVAILABLE = False

try:
    import gymnasium
    GYMNASIUM_AVAILABLE = True
except ImportError:
    pass

try:
    import pymdp
    PYMDP_AVAILABLE = True
except ImportError:
    pass

# Import the package under test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Only import what we can
if GYMNASIUM_AVAILABLE:
    from ..core import GridWorld, OrientedTriModalGrid
else:
    GridWorld = None
    OrientedTriModalGrid = None


@pytest.fixture
def simple_gridworld():
    """Simple 3x3 GridWorld for testing."""
    pytest.importorskip("gymnasium", reason="gymnasium required for environment tests")
    env = GridWorld(
        n_rows=3,
        n_cols=3,
        reward_pos=(2, 2),
        punish_pos=(0, 2),
        start_pos=(1, 1),
        max_steps=10,
        render_mode=None
    )
    return env


@pytest.fixture
def gridworld_5x7():
    """Standard 5x7 GridWorld for testing."""
    pytest.importorskip("gymnasium", reason="gymnasium required for environment tests")
    return GridWorld(
        n_rows=5,
        n_cols=7,
        reward_pos=(4, 6),
        punish_pos=(0, 6),
        start_pos=(0, 0),
        render_mode=None
    )


@pytest.fixture
def oriented_grid():
    """Simple oriented tri-modal grid for testing."""
    pytest.importorskip("gymnasium", reason="gymnasium required for environment tests")
    return OrientedTriModalGrid(
        n_rows=4,
        n_cols=4,
        reward_pos=(3, 3),
        punish_pos=(0, 3),
        start_pos=(1, 1),
        start_ori="N",
        max_steps=20,
        render_mode=None
    )


@pytest.fixture
def mock_agent():
    """Mock pymdp agent for testing compatibility functions."""
    agent = Mock()

    # Mock basic agent interface
    agent.infer_states = Mock(return_value=None)
    agent.infer_policies = Mock(return_value=None)
    agent.sample_action = Mock(return_value=0)

    # Mock attributes for sophisticated inference
    agent.sophisticated = False
    agent.use_sophisticated_inference = False

    return agent


@pytest.fixture
def rng():
    """Deterministic random number generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture(autouse=True)
def disable_logging():
    """Disable logging in tests unless explicitly needed."""
    import logging
    logging.getLogger().setLevel(logging.CRITICAL)
    logging.getLogger('active_inference').setLevel(logging.CRITICAL)


@pytest.fixture
def mock_pymdp_agent():
    """Mock pymdp Agent class for testing agent factories."""
    mock_agent_class = Mock()

    # Create mock instance
    mock_instance = Mock()
    mock_instance.infer_states = Mock()
    mock_instance.infer_policies = Mock()
    mock_instance.sample_action = Mock(return_value=0)
    mock_instance.reset = Mock()

    mock_agent_class.return_value = mock_instance
    return mock_agent_class
