"""
Human-playable GridWorld demo with graphical rendering.

This script creates a GridWorld environment and lets the user watch
random actions being taken with real-time graphical rendering.
"""

import argparse
import logging
import time
from typing import NoReturn

import numpy as np

from ..core import GridWorld
from ..utils.parsers import create_gridworld_parser

logger = logging.getLogger(__name__)


def run_human_demo(
    n_rows: int = 5,
    n_cols: int = 7,
    reward_pos: tuple[int, int] = (4, 6),
    punish_pos: tuple[int, int] = (0, 6),
    start_pos: tuple[int, int] = (0, 0),
    step_cost: float = 0.0,
    reward: float = 1.0,
    punish: float = -1.0,
    max_steps: int = 200,
    slip_prob: float = 0.0,
    seed: int = 0,
    max_episodes: int = 10,
    frame_delay: float = 0.05,
    reset_delay: float = 1.5,
) -> None:
    """
    Run human-viewable GridWorld demo with random policy.

    Args:
        n_rows: Number of grid rows
        n_cols: Number of grid columns
        reward_pos: Position of reward terminal
        punish_pos: Position of punish terminal
        start_pos: Starting position
        step_cost: Cost per step
        reward: Reward value
        punish: Punishment value
        max_steps: Maximum steps per episode
        slip_prob: Action slip probability
        seed: Random seed
        max_episodes: Maximum number of episodes to show
        frame_delay: Delay between frames (seconds)
        reset_delay: Delay after episode end (seconds)
    """
    logger.info(f"Starting human GridWorld demo: {n_rows}×{n_cols} grid")

    env = GridWorld(
        n_rows=n_rows,
        n_cols=n_cols,
        reward_pos=reward_pos,
        punish_pos=punish_pos,
        start_pos=start_pos,
        step_cost=step_cost,
        reward=reward,
        punish=punish,
        max_steps=max_steps,
        slip_prob=slip_prob,
        render_mode="human",
    )

    rng = np.random.default_rng(seed)
    episode_count = 0

    try:
        while episode_count < max_episodes:
            obs, info = env.reset(seed=seed + episode_count)
            total_reward = 0.0
            steps = 0

            logger.info(f"Starting episode {episode_count + 1}")

            while True:
                # Random action
                action = int(rng.integers(0, env.action_space.n))
                obs, r, terminated, truncated, info = env.step(action)
                total_reward += r
                steps += 1

                # Render the environment
                env.render()

                # Small delay for animation
                time.sleep(frame_delay)

                if terminated or truncated:
                    outcome = "reward" if terminated and env.pos == reward_pos else \
                             "punish" if terminated and env.pos == punish_pos else "timeout"
                    logger.info(
                        f"Episode {episode_count + 1} ended: {outcome}, "
                        f"steps={steps}, return={total_reward:.2f}"
                    )

                    # Pause to show final state
                    time.sleep(reset_delay)
                    episode_count += 1
                    break

    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    finally:
        env.close()
        logger.info("Demo finished")


def main() -> NoReturn:
    """Main entry point with argument parsing."""
    parser = create_gridworld_parser(
        description="Human-viewable GridWorld demo with random policy",
        add_experiment=False,
        add_agent=False,
        add_plotting=False,
        add_live_demo=False,
    )

    parser.add_argument(
        "--max-episodes",
        type=int,
        default=10,
        help="Maximum number of episodes to show"
    )
    parser.add_argument(
        "--frame-delay",
        type=float,
        default=0.05,
        help="Delay between frames (seconds)"
    )
    parser.add_argument(
        "--reset-delay",
        type=float,
        default=1.5,
        help="Delay after episode end (seconds)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Parse positions
    from ..utils.parsers import parse_gridworld_args
    gridworld_args = parse_gridworld_args(args)

    # Run demo
    run_human_demo(
        **gridworld_args,
        seed=args.seed,
        max_episodes=args.max_episodes,
        frame_delay=args.frame_delay,
        reset_delay=args.reset_delay,
    )


if __name__ == "__main__":
    main()

