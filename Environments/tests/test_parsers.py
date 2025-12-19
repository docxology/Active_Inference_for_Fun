"""
Tests for argument parsing utilities.
"""

import pytest
import argparse

from ..utils.parsers import (
    parse_pos,
    add_gridworld_arguments,
    add_agent_arguments,
    add_experiment_arguments,
    add_plotting_arguments,
    add_live_demo_arguments,
    parse_gridworld_args,
    parse_agent_args,
    create_gridworld_parser,
)


class TestParsePos:
    """Test parse_pos function."""

    def test_parse_pos_coordinates(self):
        """Test parsing coordinate strings."""
        assert parse_pos("2,3") == (2, 3)
        assert parse_pos("0,0") == (0, 0)
        assert parse_pos("10,5") == (10, 5)

    def test_parse_pos_random(self):
        """Test parsing 'random' string."""
        assert parse_pos("random") is None
        assert parse_pos("RANDOM") is None
        assert parse_pos(" Random ") is None

    def test_parse_pos_invalid_format(self):
        """Test invalid position format."""
        with pytest.raises(ValueError, match="Invalid position format"):
            parse_pos("2")

        with pytest.raises(ValueError, match="Invalid position format"):
            parse_pos("2,3,4")

        with pytest.raises(ValueError, match="Invalid position format"):
            parse_pos("invalid")

        with pytest.raises(ValueError, match="Invalid position format"):
            parse_pos("2.5,3")


class TestArgumentGroupAdders:
    """Test argument group adder functions."""

    def test_add_gridworld_arguments(self):
        """Test adding GridWorld arguments."""
        parser = argparse.ArgumentParser()
        add_gridworld_arguments(parser)

        # Should have all expected arguments
        args = parser.parse_args([
            "--rows", "5", "--cols", "7",
            "--reward-pos", "4,6", "--punish-pos", "0,6",
            "--start-pos", "1,1", "--step-cost", "-0.1",
            "--reward", "2.0", "--punish", "-1.0",
            "--max-steps", "100", "--slip-prob", "0.2"
        ])

        assert args.rows == 5
        assert args.cols == 7
        assert args.reward_pos == "4,6"
        assert args.punish_pos == "0,6"
        assert args.start_pos == "1,1"
        assert args.step_cost == -0.1
        assert args.reward == 2.0
        assert args.punish == -1.0
        assert args.max_steps == 100
        assert args.slip_prob == 0.2

    def test_add_agent_arguments(self):
        """Test adding agent arguments."""
        parser = argparse.ArgumentParser()
        add_agent_arguments(parser)

        args = parser.parse_args([
            "--policy-len", "6", "--gamma", "12.0",
            "--act-sel", "deterministic",
            "--c-reward", "4.0", "--c-punish", "-2.0",
            "--sophisticated"
        ])

        assert args.policy_len == 6
        assert args.gamma == 12.0
        assert args.act_sel == "deterministic"
        assert args.c_reward == 4.0
        assert args.c_punish == -2.0
        assert args.sophisticated is True

    def test_add_experiment_arguments(self):
        """Test adding experiment arguments."""
        parser = argparse.ArgumentParser()
        add_experiment_arguments(parser)

        args = parser.parse_args([
            "--episodes", "500", "--workers", "8", "--seed", "123"
        ])

        assert args.episodes == 500
        assert args.workers == 8
        assert args.seed == 123

    def test_add_plotting_arguments(self):
        """Test adding plotting arguments."""
        parser = argparse.ArgumentParser()
        add_plotting_arguments(parser)

        args = parser.parse_args([
            "--ma-window", "10", "--savefig", "plot.png"
        ])

        assert args.ma_window == 10
        assert args.savefig == "plot.png"

    def test_add_live_demo_arguments(self):
        """Test adding live demo arguments."""
        parser = argparse.ArgumentParser()
        add_live_demo_arguments(parser)

        args = parser.parse_args([
            "--fps", "30.0", "--episodes-random", "5", "--episodes-aif", "3"
        ])

        assert args.fps == 30.0
        assert args.episodes_random == 5
        assert args.episodes_aif == 3


class TestParseArgsFunctions:
    """Test argument parsing functions."""

    def test_parse_gridworld_args(self):
        """Test parse_gridworld_args function."""
        class MockArgs:
            def __init__(self):
                self.rows = 4
                self.cols = 5
                self.reward_pos = "3,4"
                self.punish_pos = "0,4"
                self.start_pos = "1,1"
                self.step_cost = -0.05
                self.reward = 1.5
                self.punish = -1.5
                self.max_steps = 150
                self.slip_prob = 0.1

        args = MockArgs()
        result = parse_gridworld_args(args)

        expected = {
            'n_rows': 4,
            'n_cols': 5,
            'reward_pos': (3, 4),
            'punish_pos': (0, 4),
            'start_pos': (1, 1),
            'step_cost': -0.05,
            'reward': 1.5,
            'punish': -1.5,
            'max_steps': 150,
            'slip_prob': 0.1,
        }

        assert result == expected

    def test_parse_gridworld_args_with_random_start(self):
        """Test parse_gridworld_args with random start."""
        class MockArgs:
            def __init__(self):
                self.rows = 3
                self.cols = 3
                self.reward_pos = "2,2"
                self.punish_pos = "0,2"
                self.start_pos = "random"
                self.step_cost = 0.0
                self.reward = 1.0
                self.punish = -1.0
                self.max_steps = 100
                self.slip_prob = 0.0

        args = MockArgs()
        result = parse_gridworld_args(args)

        assert result['start_pos'] is None

    def test_parse_gridworld_args_invalid_position(self):
        """Test parse_gridworld_args with invalid position."""
        class MockArgs:
            def __init__(self):
                self.reward_pos = "invalid"
                self.punish_pos = "0,2"
                self.rows = 3
                self.cols = 3

        args = MockArgs()

        with pytest.raises(ValueError, match="Invalid position format"):
            parse_gridworld_args(args)

    def test_parse_agent_args(self):
        """Test parse_agent_args function."""
        class MockArgs:
            def __init__(self):
                self.policy_len = 5
                self.gamma = 10.0
                self.act_sel = "stochastic"
                self.c_reward = 3.5
                self.c_punish = -2.5
                self.sophisticated = True

        args = MockArgs()
        result = parse_agent_args(args)

        expected = {
            'policy_len': 5,
            'gamma': 10.0,
            'action_selection': "stochastic",
            'c_reward': 3.5,
            'c_punish': -2.5,
            'sophisticated': True,
        }

        assert result == expected


class TestCreateGridworldParser:
    """Test create_gridworld_parser function."""

    def test_create_basic_parser(self):
        """Test creating basic GridWorld parser."""
        parser = create_gridworld_parser("Test parser")

        # Should have GridWorld arguments
        args = parser.parse_args([
            "--rows", "4", "--cols", "4",
            "--reward-pos", "3,3", "--punish-pos", "0,3"
        ])

        assert args.rows == 4
        assert args.cols == 4
        assert args.reward_pos == "3,3"

    def test_create_full_parser(self):
        """Test creating parser with all argument groups."""
        parser = create_gridworld_parser(
            "Full parser",
            add_agent=True,
            add_experiment=True,
            add_plotting=True,
            add_live_demo=True
        )

        # Should accept arguments from all groups
        args = parser.parse_args([
            # GridWorld
            "--rows", "5", "--cols", "5",
            "--reward-pos", "4,4", "--punish-pos", "0,4",
            # Agent
            "--policy-len", "4", "--gamma", "16.0",
            # Experiment
            "--episodes", "100", "--workers", "4",
            # Plotting
            "--ma-window", "20",
            # Live demo
            "--fps", "15.0"
        ])

        assert args.rows == 5
        assert args.cols == 5
        assert args.policy_len == 4
        assert args.episodes == 100
        assert args.ma_window == 20
        assert args.fps == 15.0
