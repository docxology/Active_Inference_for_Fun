"""
Tests for plotting utilities.
"""

import pytest
import numpy as np
from unittest.mock import patch, Mock

from ..utils.plotting import (
    moving_average,
    plot_experiment_comparison,
    plot_experiment_stats,
    plot_free_energy_terms,
    init_live_demo_figure,
)


class TestMovingAverage:
    """Test moving_average function."""

    def test_moving_average_basic(self):
        """Test basic moving average calculation."""
        x = np.array([1, 2, 3, 4, 5])
        result = moving_average(x, w=3)

        # Should pad with NaN for first w-1 elements
        expected = np.array([np.nan, np.nan, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(result, expected)

    def test_moving_average_window_one(self):
        """Test moving average with window size 1."""
        x = np.array([1, 2, 3])
        result = moving_average(x, w=1)

        np.testing.assert_array_equal(result, x)

    def test_moving_average_full_window(self):
        """Test moving average where window equals array size."""
        x = np.array([1, 2, 3])
        result = moving_average(x, w=3)

        expected = np.array([np.nan, np.nan, 2.0])
        np.testing.assert_array_equal(result, expected)

    def test_moving_average_empty_array(self):
        """Test moving average with empty array."""
        x = np.array([])
        result = moving_average(x, w=2)

        assert len(result) == 0


class TestPlotExperimentComparison:
    """Test plot_experiment_comparison function."""

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.subplot')
    @patch('matplotlib.pyplot.show')
    def test_plot_experiment_comparison_basic(self, mock_show, mock_subplot, mock_figure):
        """Test basic experiment comparison plotting."""
        # Mock matplotlib components
        mock_ax = Mock()
        mock_subplot.return_value = mock_ax

        returns_rand = np.array([1.0, -1.0, 0.5])
        returns_aif = np.array([1.0, 1.0, 1.0])
        steps_rand = np.array([5, 10, 8])
        steps_aif = np.array([3, 3, 4])
        outcomes_rand = ["reward", "punish", "timeout"]
        outcomes_aif = ["reward", "reward", "reward"]

        plot_experiment_comparison(
            returns_rand=returns_rand,
            returns_aif=returns_aif,
            steps_rand=steps_rand,
            steps_aif=steps_aif,
            outcomes_rand=outcomes_rand,
            outcomes_aif=outcomes_aif,
            ma_window=2,
            savefig=None
        )

        # Check that plotting functions were called
        mock_figure.assert_called_once()
        assert mock_subplot.call_count >= 4  # At least 4 subplots
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.figure')
    def test_plot_experiment_comparison_savefig(self, mock_figure, mock_savefig):
        """Test experiment comparison with savefig."""
        returns_rand = np.array([1.0])
        returns_aif = np.array([1.0])
        steps_rand = np.array([5])
        steps_aif = np.array([3])
        outcomes_rand = ["reward"]
        outcomes_aif = ["reward"]

        plot_experiment_comparison(
            returns_rand=returns_rand,
            returns_aif=returns_aif,
            steps_rand=steps_rand,
            steps_aif=steps_aif,
            outcomes_rand=outcomes_rand,
            outcomes_aif=outcomes_aif,
            savefig="test.png"
        )

        mock_savefig.assert_called_once_with("test.png", dpi=150, bbox_inches='tight')

    def test_plot_experiment_comparison_no_matplotlib(self):
        """Test graceful handling when matplotlib is unavailable."""
        with patch.dict('sys.modules', {'matplotlib': None}):
            # Should not raise error
            plot_experiment_comparison(
                returns_rand=np.array([1.0]),
                returns_aif=np.array([1.0]),
                steps_rand=np.array([5]),
                steps_aif=np.array([3]),
                outcomes_rand=["reward"],
                outcomes_aif=["reward"]
            )


class TestPlotExperimentStats:
    """Test plot_experiment_stats function."""

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.subplot')
    @patch('matplotlib.pyplot.show')
    def test_plot_experiment_stats_basic(self, mock_show, mock_subplot, mock_figure):
        """Test basic experiment stats plotting."""
        mock_ax = Mock()
        mock_subplot.return_value = mock_ax

        returns = np.array([1.0, -1.0, 0.5, 1.0])
        steps = np.array([3, 8, 5, 4])
        outcomes = ["reward", "punish", "timeout", "reward"]

        plot_experiment_stats(
            returns=returns,
            steps=steps,
            outcomes=outcomes,
            ma_window=2,
            savefig=None
        )

        mock_figure.assert_called_once()
        assert mock_subplot.call_count >= 4
        mock_show.assert_called_once()

    def test_plot_experiment_stats_savefig(self):
        """Test experiment stats with savefig."""
        with patch('matplotlib.pyplot.savefig'):
            returns = np.array([1.0])
            steps = np.array([3])
            outcomes = ["reward"]

            plot_experiment_stats(
                returns=returns,
                steps=steps,
                outcomes=outcomes,
                savefig="stats.png"
            )


class TestPlotFreeEnergyTerms:
    """Test plot_free_energy_terms function."""

    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.show')
    def test_plot_free_energy_terms_basic(self, mock_show, mock_figure):
        """Test free energy terms plotting."""
        history = {
            "complexity": [0.1, 0.2, 0.15],
            "accuracy": [0.05, 0.08, 0.06],
            "extrinsic": [1.0, 0.9, 0.95],
            "epistemic": [0.01, 0.02, 0.015]
        }

        plot_free_energy_terms(
            history=history,
            episode_idx=1,
            title_suffix="Test"
        )

        mock_figure.assert_called_once()
        mock_show.assert_called_once()


class TestInitLiveDemoFigure:
    """Test init_live_demo_figure function."""

    @patch('matplotlib.pyplot.ion')
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.figure')
    def test_init_live_demo_figure(self, mock_figure, mock_show, mock_ion):
        """Test live demo figure initialization."""
        mock_fig = Mock()
        mock_ax_grid = Mock()
        mock_ax_bars = Mock()
        mock_bar = Mock()

        mock_figure.return_value = mock_fig
        mock_fig.add_subplot.side_effect = [mock_ax_grid, mock_ax_bars]
        mock_ax_bars.bar.return_value = [mock_bar]

        result = init_live_demo_figure(n_rows=4, n_cols=4)

        assert len(result) == 6  # Should return 6 items
        fig, ax_grid, ax_bars, bars, texts, plt = result

        assert fig is mock_fig
        assert ax_grid is mock_ax_grid
        assert ax_bars is mock_ax_bars
        assert bars == [mock_bar]

        mock_ion.assert_called_once()
        mock_show.assert_called_once()

    def test_init_live_demo_figure_no_matplotlib(self):
        """Test error when matplotlib unavailable."""
        with patch.dict('sys.modules', {'matplotlib': None}):
            with pytest.raises(ImportError):
                init_live_demo_figure(n_rows=4, n_cols=4)
