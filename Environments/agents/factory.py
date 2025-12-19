"""
Active Inference agent factory for GridWorld environments.

This module provides functions to construct Active Inference agents
tailored for grid world environments with specific generative models.
"""

import logging
from typing import Dict, Optional, Tuple, Any

import numpy as np

try:
    from pymdp.agent import Agent
except ImportError:
    raise ImportError("pymdp package required for Active Inference agents")

logger = logging.getLogger(__name__)

def build_gridworld_agent(
    n_rows: int,
    n_cols: int,
    reward_pos: Tuple[int, int],
    punish_pos: Tuple[int, int],
    start_pos: Optional[Tuple[int, int]] = None,
    c_reward: float = 3.0,
    c_punish: float = -3.0,
    policy_len: int = 4,
    gamma: float = 16.0,
    action_selection: str = "stochastic",
    sophisticated: bool = False,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    """
    Build a fully-observable Active Inference agent for GridWorld.

    Constructs an Active Inference agent with appropriate generative model
    (A, B, C, D matrices) for a grid world environment. The agent assumes
    perfect observability of its position.

    Args:
        n_rows: Number of grid rows
        n_cols: Number of grid columns
        reward_pos: (row, col) position of rewarding terminal state
        punish_pos: (row, col) position of punishing terminal state
        start_pos: (row, col) starting position, or None for uniform prior
        c_reward: Preference strength for reward observations
        c_punish: Preference strength for punishment observations
        policy_len: Length of planning horizon (policy depth)
        gamma: Policy precision parameter
        action_selection: Action selection method ("stochastic" or "deterministic")
        sophisticated: Whether to use sophisticated inference if available

    Returns:
        Tuple of (agent, model, controls) where:
        - agent: pymdp Agent instance
        - model: Dict with A, B, C, D matrices and helper functions
        - controls: Dict with policy inference wrapper and configuration

    Raises:
        ValueError: If grid dimensions are invalid or positions are out of bounds
        RuntimeError: If agent construction fails with all attempted methods
    """
    # Validate inputs
    if n_rows < 1 or n_cols < 1:
        raise ValueError("Grid dimensions must be positive")

    if not (0 <= reward_pos[0] < n_rows and 0 <= reward_pos[1] < n_cols):
        raise ValueError(f"Reward position {reward_pos} out of bounds for {n_rows}×{n_cols} grid")

    if not (0 <= punish_pos[0] < n_rows and 0 <= punish_pos[1] < n_cols):
        raise ValueError(f"Punish position {punish_pos} out of bounds for {n_rows}×{n_cols} grid")

    if start_pos is not None and not (0 <= start_pos[0] < n_rows and 0 <= start_pos[1] < n_cols):
        raise ValueError(f"Start position {start_pos} out of bounds for {n_rows}×{n_cols} grid")

    logger.info(
        f"Building GridWorld agent: {n_rows}×{n_cols} grid, "
        f"reward={reward_pos}, punish={punish_pos}, start={start_pos}, "
        f"policy_len={policy_len}, sophisticated={sophisticated}"
    )

    S = n_rows * n_cols  # Number of states (positions)
    O = S              # Number of observations (fully observable)
    U = 4              # Number of actions (UP, RIGHT, DOWN, LEFT)

    def pos_to_idx(r: int, c: int) -> int:
        return r * n_cols + c

    def clip_move(r: int, c: int, u: int) -> tuple[int, int]:
        if u == 0: r = max(0, r - 1)           # up
        elif u == 1: c = min(n_cols - 1, c + 1)# right
        elif u == 2: r = min(n_rows - 1, r + 1)# down
        elif u == 3: c = max(0, c - 1)         # left
        return r, c

    # A: identity (obs == state)
    A = np.eye(O, S, dtype=np.float64)

    # B: deterministic transitions (S x S x U)
    B = np.zeros((S, S, U), dtype=np.float64)
    for s_prev in range(S):
        r, c = divmod(s_prev, n_cols)
        for u in range(U):
            r2, c2 = clip_move(r, c, u)
            s_next = pos_to_idx(r2, c2)
            B[s_next, s_prev, u] = 1.0

    # C: outcome preferences over observations
    C = np.zeros(O, dtype=np.float64)
    C[pos_to_idx(*reward_pos)] = c_reward
    C[pos_to_idx(*punish_pos)] = c_punish

    # D: prior over initial state
    if start_pos is not None:
        D = np.zeros(S, dtype=np.float64)
        D[pos_to_idx(*start_pos)] = 1.0
    else:
        D = np.ones(S, dtype=np.float64)
        D[[pos_to_idx(*reward_pos), pos_to_idx(*punish_pos)]] = 0.0
        D /= D.sum()

    # Instantiate Agent with broad compatibility across pymdp versions
    agent = None
    last_error = None

    agent_construction_attempts = [
        {
            "name": "full kwargs with sophisticated",
            "ctor": lambda: Agent(
                A=A, B=B, C=C, D=D,
                policy_len=policy_len, gamma=gamma,
                action_selection=action_selection,
                sophisticated=sophisticated
            )
        },
        {
            "name": "kwargs without sophisticated",
            "ctor": lambda: Agent(
                A=A, B=B, C=C, D=D,
                policy_len=policy_len, gamma=gamma,
                action_selection=action_selection
            )
        },
        {
            "name": "positional fallback",
            "ctor": lambda: Agent(A, B, C, D, policy_len=policy_len)
        }
    ]

    for attempt in agent_construction_attempts:
        try:
            logger.debug(f"Trying agent construction: {attempt['name']}")
            agent = attempt["ctor"]()
            logger.debug(f"Successfully created agent with: {attempt['name']}")
            break
        except Exception as e:
            last_error = e
            logger.debug(f"Agent construction failed ({attempt['name']}): {e}")

    if agent is None:
        raise RuntimeError(
            f"Could not construct pymdp.Agent with any method. "
            f"Last error: {last_error}"
        )

    # Set sophisticated mode if requested and supported
    if sophisticated:
        sophisticated_attrs = [
            "sophisticated",
            "use_sophisticated_inference",
            "sophisticated_inference"
        ]

        for attr in sophisticated_attrs:
            if hasattr(agent, attr):
                try:
                    setattr(agent, attr, True)
                    logger.debug(f"Set {attr} = True on agent")
                    break
                except Exception as e:
                    logger.debug(f"Failed to set {attr}: {e}")

    # Policy inference wrapper with sophisticated mode support
    def infer_policies_wrapper() -> Any:
        """
        Infer policies using appropriate method for agent configuration.

        Uses sophisticated inference if requested and available,
        otherwise falls back to default inference.
        """
        if sophisticated:
            # Try sophisticated inference methods
            sophisticated_kwargs = [
                {"mode": "sophisticated"},
                {"method": "sophisticated"}
            ]

            for kwargs in sophisticated_kwargs:
                try:
                    agent.infer_policies(**kwargs)
                    logger.debug("Used sophisticated policy inference")
                    return
                except (TypeError, AttributeError):
                    continue

            logger.warning("Sophisticated inference requested but not available, using default")

        # Default inference
        agent.infer_policies()
        logger.debug("Used default policy inference")

    model = {
        "A": A, "B": B, "C": C, "D": D,
        "pos_to_idx": pos_to_idx,
        "reward_idx": pos_to_idx(*reward_pos),
        "punish_idx": pos_to_idx(*punish_pos),
    }
    controls = {
        "infer_policies": infer_policies_wrapper,
        "sophisticated": sophisticated,
        "policy_len": policy_len,
        "gamma": gamma,
        "action_selection": action_selection,
    }
    return agent, model, controls
