"""
Argument parsing utilities for Active Inference experiments.

This module provides standardized argument parsers and parsing functions
used across different experiment scripts.
"""

import argparse
from typing import Optional, Tuple


def parse_pos(s: str) -> Optional[Tuple[int, int]]:
    """
    Parse position string into (row, col) tuple.

    Args:
        s: Position string in format "row,col" or "random"

    Returns:
        Tuple of (row, col) integers, or None for random positioning

    Raises:
        ValueError: If position string format is invalid
    """
    if s.strip().lower() == "random":
        return None

    try:
        r, c = s.split(",")
        return (int(r.strip()), int(c.strip()))
    except ValueError as e:
        raise ValueError(f"Invalid position format '{s}'. Expected 'row,col' or 'random'") from e


def add_gridworld_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add common GridWorld environment arguments to parser.

    Args:
        parser: ArgumentParser instance to add arguments to
    """
    grid_group = parser.add_argument_group('GridWorld Environment')

    grid_group.add_argument(
        "--rows",
        type=int,
        default=5,
        help="Number of grid rows"
    )
    grid_group.add_argument(
        "--cols",
        type=int,
        default=7,
        help="Number of grid columns"
    )
    grid_group.add_argument(
        "--reward-pos",
        type=str,
        default="4,6",
        help="Position of reward cell as 'row,col'"
    )
    grid_group.add_argument(
        "--punish-pos",
        type=str,
        default="0,6",
        help="Position of punishment cell as 'row,col'"
    )
    grid_group.add_argument(
        "--start-pos",
        type=str,
        default="0,0",
        help="Starting position as 'row,col' or 'random' for random starts"
    )
    grid_group.add_argument(
        "--step-cost",
        type=float,
        default=0.0,
        help="Reward penalty per step (negative to encourage shorter paths)"
    )
    grid_group.add_argument(
        "--reward",
        type=float,
        default=1.0,
        help="Terminal reward value"
    )
    grid_group.add_argument(
        "--punish",
        type=float,
        default=-1.0,
        help="Terminal punishment value"
    )
    grid_group.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum steps per episode"
    )
    grid_group.add_argument(
        "--slip-prob",
        type=float,
        default=0.0,
        help="Probability of action slipping (stochasticity)"
    )


def add_agent_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add common Active Inference agent arguments to parser.

    Args:
        parser: ArgumentParser instance to add arguments to
    """
    agent_group = parser.add_argument_group('Active Inference Agent')

    agent_group.add_argument(
        "--policy-len",
        type=int,
        default=4,
        help="Policy length (planning horizon)"
    )
    agent_group.add_argument(
        "--gamma",
        type=float,
        default=16.0,
        help="Policy precision parameter"
    )
    agent_group.add_argument(
        "--act-sel",
        type=str,
        default="stochastic",
        choices=["stochastic", "deterministic"],
        help="Action selection mode"
    )
    agent_group.add_argument(
        "--c-reward",
        type=float,
        default=3.0,
        help="Preference strength for reward observations"
    )
    agent_group.add_argument(
        "--c-punish",
        type=float,
        default=-3.0,
        help="Preference strength for punishment observations"
    )
    agent_group.add_argument(
        "--sophisticated",
        action="store_true",
        help="Enable sophisticated inference if supported by pymdp"
    )


def add_experiment_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add common experiment arguments to parser.

    Args:
        parser: ArgumentParser instance to add arguments to
    """
    exp_group = parser.add_argument_group('Experiment Settings')

    exp_group.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Number of episodes to run"
    )
    exp_group.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (1 = sequential)"
    )
    exp_group.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )


def add_plotting_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add common plotting arguments to parser.

    Args:
        parser: ArgumentParser instance to add arguments to
    """
    plot_group = parser.add_argument_group('Plotting')

    plot_group.add_argument(
        "--ma-window",
        type=int,
        default=50,
        help="Moving average window for plots"
    )
    plot_group.add_argument(
        "--savefig",
        type=str,
        default="",
        help="Optional path to save figure (PNG)"
    )


def add_live_demo_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add common live demo arguments to parser.

    Args:
        parser: ArgumentParser instance to add arguments to
    """
    demo_group = parser.add_argument_group('Live Demo')

    demo_group.add_argument(
        "--fps",
        type=float,
        default=12.0,
        help="Frames per second for rendering"
    )
    demo_group.add_argument(
        "--episodes-random",
        type=int,
        default=3,
        help="Number of random episodes to show"
    )
    demo_group.add_argument(
        "--episodes-aif",
        type=int,
        default=3,
        help="Number of AIF episodes to show"
    )


def parse_gridworld_args(args: argparse.Namespace) -> dict:
    """
    Parse and validate GridWorld-related arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Dictionary of validated GridWorld parameters

    Raises:
        ValueError: If position arguments are invalid
    """
    try:
        reward_pos = parse_pos(args.reward_pos)
        punish_pos = parse_pos(args.punish_pos)
        start_pos = parse_pos(args.start_pos)
    except ValueError as e:
        raise ValueError(f"Position parsing error: {e}") from e

    return {
        'n_rows': args.rows,
        'n_cols': args.cols,
        'reward_pos': reward_pos,
        'punish_pos': punish_pos,
        'start_pos': start_pos,
        'step_cost': args.step_cost,
        'reward': args.reward,
        'punish': args.punish,
        'max_steps': args.max_steps,
        'slip_prob': args.slip_prob,
    }


def parse_agent_args(args: argparse.Namespace) -> dict:
    """
    Parse Active Inference agent arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Dictionary of agent parameters
    """
    return {
        'policy_len': args.policy_len,
        'gamma': args.gamma,
        'action_selection': args.act_sel,
        'c_reward': args.c_reward,
        'c_punish': args.c_punish,
        'sophisticated': getattr(args, 'sophisticated', False),
    }


def create_gridworld_parser(
    description: str = "GridWorld experiment",
    add_agent: bool = False,
    add_experiment: bool = False,
    add_plotting: bool = False,
    add_live_demo: bool = False,
) -> argparse.ArgumentParser:
    """
    Create a configured argument parser for GridWorld experiments.

    Args:
        description: Parser description
        add_agent: Whether to add agent arguments
        add_experiment: Whether to add experiment arguments
        add_plotting: Whether to add plotting arguments
        add_live_demo: Whether to add live demo arguments

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(description=description)

    add_gridworld_arguments(parser)

    if add_agent:
        add_agent_arguments(parser)

    if add_experiment:
        add_experiment_arguments(parser)

    if add_plotting:
        add_plotting_arguments(parser)

    if add_live_demo:
        add_live_demo_arguments(parser)

    return parser
