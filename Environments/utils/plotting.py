"""
Plotting utilities for Active Inference experiments.

This module provides standardized plotting functions for visualizing
experiment results, environment states, and agent metrics.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    """
    Compute moving average of array with window size w.

    Args:
        x: Input array
        w: Window size for moving average

    Returns:
        Array with moving average applied, padded with NaN for alignment
    """
    if w <= 1:
        return x

    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    ma = (cumsum[w:] - cumsum[:-w]) / float(w)

    # Pad to original length for nicer plotting
    pad = np.full(w - 1, np.nan)
    return np.concatenate([pad, ma])


def plot_experiment_comparison(
    returns_rand: np.ndarray,
    returns_aif: np.ndarray,
    steps_rand: np.ndarray,
    steps_aif: np.ndarray,
    outcomes_rand: List[str],
    outcomes_aif: List[str],
    ma_window: int = 50,
    figsize: Tuple[int, int] = (12, 8),
    savefig: Optional[str] = None,
    title_suffix: str = "",
) -> None:
    """
    Create comprehensive comparison plot for random vs AIF experiments.

    Args:
        returns_rand: Returns from random episodes
        returns_aif: Returns from AIF episodes
        steps_rand: Steps from random episodes
        steps_aif: Steps from AIF episodes
        outcomes_rand: Outcome strings from random episodes
        outcomes_aif: Outcome strings from AIF episodes
        ma_window: Window size for moving average
        figsize: Figure size (width, height)
        savefig: Optional path to save figure
        title_suffix: Additional text for plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plotting")
        return

    from collections import Counter

    # Compute summary statistics
    def summarize(returns, steps, outcomes):
        cnt = Counter(outcomes)
        return {
            "success_rate": cnt.get("reward", 0) / len(outcomes),
            "punish_rate": cnt.get("punish", 0) / len(outcomes),
            "timeout_rate": cnt.get("timeout", 0) / len(outcomes),
            "avg_return": float(np.mean(returns)),
            "avg_steps": float(np.mean(steps)),
            "counts": cnt,
        }

    sum_r = summarize(returns_rand, steps_rand, outcomes_rand)
    sum_a = summarize(returns_aif, steps_aif, outcomes_aif)

    # Create figure
    fig = plt.figure(figsize=figsize)

    # 1. Episode returns with moving averages
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(returns_rand, alpha=0.35, lw=0.8, label="Random")
    ax1.plot(returns_aif, alpha=0.35, lw=0.8, label="AIF")
    ax1.plot(moving_average(returns_rand, ma_window), lw=2, label=f"Random (MA={ma_window})")
    ax1.plot(moving_average(returns_aif, ma_window), lw=2, label=f"AIF (MA={ma_window})")
    ax1.set_title("Episode Returns")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Steps per episode (moving average only)
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(moving_average(steps_rand.astype(float), ma_window), lw=2, label="Random (MA)")
    ax2.plot(moving_average(steps_aif.astype(float), ma_window), lw=2, label="AIF (MA)")
    ax2.set_title("Steps per Episode (MA)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 3. Outcome counts (bar chart)
    ax3 = plt.subplot(2, 2, 3)
    labels = ["reward", "punish", "timeout"]
    r_counts = np.array([sum_r["counts"].get(k, 0) for k in labels])
    a_counts = np.array([sum_a["counts"].get(k, 0) for k in labels])
    x = np.arange(len(labels))
    w = 0.4
    ax3.bar(x - w/2, r_counts, width=w, label="Random")
    ax3.bar(x + w/2, a_counts, width=w, label="AIF")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_title("Outcome Counts")
    ax3.set_ylabel("Count")

    # Add count labels on bars
    for i, (rv, av) in enumerate(zip(r_counts, a_counts)):
        ax3.text(i - w/2, rv, str(int(rv)), ha="center", va="bottom")
        ax3.text(i + w/2, av, str(int(av)), ha="center", va="bottom")

    ax3.legend()

    # 4. Return distribution histograms
    ax4 = plt.subplot(2, 2, 4)
    ax4.hist(returns_rand, bins=30, alpha=0.6, label="Random")
    ax4.hist(returns_aif, bins=30, alpha=0.6, label="AIF")
    ax4.set_title("Return Distribution")
    ax4.set_xlabel("Return")
    ax4.set_ylabel("Frequency")
    ax4.legend()

    # Overall title
    fig.suptitle(
        f"Active Inference vs Random Comparison{title_suffix}\n"
        f"Random: success={sum_r['success_rate']:.3f}, avgR={sum_r['avg_return']:.3f}, "
        f"avgSteps={sum_r['avg_steps']:.1f} | "
        f"AIF: success={sum_a['success_rate']:.3f}, avgR={sum_a['avg_return']:.3f}, "
        f"avgSteps={sum_a['avg_steps']:.1f}",
        fontsize=11
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.92])

    if savefig:
        plt.savefig(savefig, dpi=150, bbox_inches='tight')
        logger.info(f"Saved figure to: {savefig}")

    plt.show()


def plot_experiment_stats(
    returns: np.ndarray,
    steps: np.ndarray,
    outcomes: List[str],
    ma_window: int = 50,
    figsize: Tuple[int, int] = (12, 7),
    savefig: Optional[str] = None,
    title_prefix: str = "GridWorld Stats",
) -> None:
    """
    Create statistical plots for experiment results.

    Args:
        returns: Episode returns
        steps: Episode steps
        outcomes: Episode outcome strings
        ma_window: Moving average window
        figsize: Figure size
        savefig: Optional save path
        title_prefix: Prefix for plot title
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plotting")
        return

    from collections import Counter

    # Compute statistics
    outcome_counts = Counter(outcomes)
    success_rate = outcome_counts.get("reward", 0) / len(outcomes)
    punish_rate = outcome_counts.get("punish", 0) / len(outcomes)
    timeout_rate = outcome_counts.get("timeout", 0) / len(outcomes)
    avg_return = np.mean(returns)
    avg_steps = np.mean(steps)

    fig = plt.figure(figsize=figsize)

    # 1. Returns with moving average
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(returns, lw=0.8, alpha=0.5)
    ma = moving_average(returns, ma_window)
    ax1.plot(ma, lw=2)
    ax1.set_title(f"Episode Returns (MA window={ma_window})")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Return")
    ax1.grid(True, alpha=0.3)

    # 2. Steps per episode
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(steps, lw=0.8, alpha=0.7)
    ax2.set_title("Steps per Episode")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.grid(True, alpha=0.3)

    # 3. Outcome distribution
    ax3 = plt.subplot(2, 2, 3)
    labels = ["reward", "punish", "timeout"]
    counts = [outcome_counts.get(k, 0) for k in labels]
    ax3.bar(labels, counts)
    ax3.set_title("Episode Outcomes")
    ax3.set_ylabel("Count")
    for i, c in enumerate(counts):
        ax3.text(i, c, str(c), ha="center", va="bottom")

    # 4. Return histogram
    ax4 = plt.subplot(2, 2, 4)
    ax4.hist(returns, bins=30, alpha=0.9)
    ax4.set_title("Return Distribution")
    ax4.set_xlabel("Return")
    ax4.set_ylabel("Frequency")

    fig.suptitle(
        f"{title_prefix}\n"
        f"Success={success_rate:.3f}, Punish={punish_rate:.3f}, Timeout={timeout_rate:.3f}, "
        f"AvgReturn={avg_return:.3f}, AvgSteps={avg_steps:.2f}",
        fontsize=11
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.93])

    if savefig:
        plt.savefig(savefig, dpi=150, bbox_inches='tight')
        logger.info(f"Saved figure to: {savefig}")

    plt.show()


def plot_free_energy_terms(
    history: Dict[str, List[float]],
    episode_idx: int = 1,
    title_suffix: str = "",
    figsize: Tuple[int, int] = (10, 6),
) -> None:
    """
    Plot per-step free energy terms from an episode.

    Args:
        history: Dictionary with keys 'complexity', 'accuracy', 'extrinsic', 'epistemic'
        episode_idx: Episode number for title
        title_suffix: Additional title text
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping plotting")
        return

    t = np.arange(len(history["complexity"]))

    plt.figure(figsize=figsize)
    plt.plot(t, history["complexity"], label="Complexity (KL(q||prior))", linewidth=2)
    plt.plot(t, history["accuracy"], label="Accuracy (-E_q ln p(o|s))", linewidth=2)
    plt.plot(t, history["extrinsic"], label="Extrinsic (utility)", linewidth=2)
    plt.plot(t, history["epistemic"], label="Epistemic (state info gain)", linewidth=2)

    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title(f"Per-step Free Energy Terms — Episode {episode_idx}{(' — ' + title_suffix) if title_suffix else ''}")
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.show()


def init_live_demo_figure(n_rows: int, n_cols: int, cell_size: int = 32) -> Tuple:
    """
    Initialize matplotlib figure for live grid world demo.

    Returns:
        Tuple of (fig, ax_grid, ax_bars, bars, texts, plt)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required for live demo")

    fig = plt.figure(figsize=(10, 5))
    ax_grid = fig.add_subplot(1, 2, 1)
    ax_bars = fig.add_subplot(1, 2, 2)

    # Initialize bars for free energy terms
    terms = ["Complexity", "Accuracy", "Extrinsic", "Epistemic"]
    bars = ax_bars.bar(terms, [0, 0, 0, 0])
    ax_bars.set_ylim(-1, 5)
    ax_bars.grid(True, axis="y", alpha=0.3)
    ax_bars.set_title("Free Energy Terms")

    # Initialize text labels
    texts = []
    for bar in bars:
        height = bar.get_height()
        text = ax_bars.text(bar.get_x() + bar.get_width()/2., height,
                          '.2f', ha='center', va='bottom')
        texts.append(text)

    plt.ion()
    plt.show(block=False)

    return fig, ax_grid, ax_bars, bars, texts, plt
