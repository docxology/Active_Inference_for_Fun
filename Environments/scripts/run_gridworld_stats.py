"""
Run GridWorld statistics experiments with random policy.

This script runs multiple episodes with a random policy and generates
comprehensive statistics and plots showing performance distributions.
"""

import argparse
import logging
from typing import List, NoReturn

import numpy as np

from ..core import GridWorld
from ..utils.episode_runners import RandomEpisodeRunner, EpisodeResult
from ..utils.plotting import plot_experiment_stats
from ..utils.parsers import create_gridworld_parser

logger = logging.getLogger(__name__)


def run_statistics_experiment(
    n_rows: int = 5,
    n_cols: int = 7,
    reward_pos: tuple[int, int] = (4, 6),
    punish_pos: tuple[int, int] = (0, 6),
    start_pos: tuple[int, int] | None = (0, 0),
    step_cost: float = 0.0,
    reward: float = 1.0,
    punish: float = -1.0,
    max_steps: int = 200,
    slip_prob: float = 0.0,
    episodes: int = 2000,
    seed: int = 0,
    ma_window: int = 50,
    savefig: str = "",
) -> None:
    """
    Run statistics experiment with random policy.

    Args:
        n_rows: Number of grid rows
        n_cols: Number of grid columns
        reward_pos: Position of reward terminal
        punish_pos: Position of punish terminal
        start_pos: Starting position (or None for random)
        step_cost: Cost per step
        reward: Reward value
        punish: Punishment value
        max_steps: Maximum steps per episode
        slip_prob: Action slip probability
        episodes: Number of episodes to run
        seed: Random seed
        ma_window: Moving average window for plots
        savefig: Optional path to save figure
    """
    logger.info(f"Running statistics experiment: {episodes} episodes on {n_rows}×{n_cols} grid")

    # Create environment
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
        render_mode=None,  # No rendering for stats
    )

    rng = np.random.default_rng(seed)

    # Run episodes
    returns = np.zeros(episodes)
    steps = np.zeros(episodes, dtype=int)
    outcomes: List[str] = []

    for ep in range(episodes):
        runner = RandomEpisodeRunner(env, rng, episode_idx=ep)
        result = runner.run_episode()

        returns[ep] = result.total_return
        steps[ep] = result.steps
        outcomes.append(result.outcome)

    # Calculate summary statistics
    from collections import Counter
    outcome_counts = Counter(outcomes)
    success_rate = outcome_counts.get("reward", 0) / episodes
    punish_rate = outcome_counts.get("punish", 0) / episodes
    timeout_rate = outcome_counts.get("timeout", 0) / episodes
    avg_return = float(np.mean(returns))
    avg_steps = float(np.mean(steps))

    # Print results
    print(f"Episodes:        {episodes}")
    print(f"Success rate:    {success_rate:.3f}")
    print(f"Punish rate:     {punish_rate:.3f}")
    print(f"Timeout rate:    {timeout_rate:.3f}")
    print(f"Avg return:      {avg_return:.3f}")
    print(f"Avg steps:       {avg_steps:.2f}")

    # Create plots
    plot_experiment_stats(
        returns=returns,
        steps=steps,
        outcomes=outcomes,
        ma_window=ma_window,
        savefig=savefig if savefig else None,
        title_prefix=f"GridWorld Statistics ({n_rows}×{n_cols})"
    )

    logger.info(f"Statistics experiment completed: {episodes} episodes, success rate {success_rate:.3f}")

def main() -> NoReturn:
    """Main entry point with argument parsing."""
    parser = create_gridworld_parser(
        description="Run GridWorld statistics experiments with random policy",
        add_experiment=True,
        add_plotting=True,
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Parse arguments
    from ..utils.parsers import parse_gridworld_args
    gridworld_args = parse_gridworld_args(args)

    # Run experiment
    run_statistics_experiment(
        **gridworld_args,
        episodes=args.episodes,
        seed=args.seed,
        ma_window=args.ma_window,
        savefig=args.savefig,
    )


if __name__ == "__main__":
    main()

