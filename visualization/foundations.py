"""Visualization helpers for stochastic-process foundations."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


def plot_paths(
    times: FloatArray,
    paths: FloatArray,
    title: str,
    y_label: str,
    max_paths: int = 20,
) -> tuple[Figure, Axes]:
    """Plot a limited number of simulated paths on a shared time grid."""
    fig, ax = plt.subplots(figsize=(10, 6))
    paths_to_plot = paths[:max_paths]

    for path in paths_to_plot:
        ax.plot(times, path, linewidth=1.2, alpha=0.75)

    ax.set_title(title)
    ax.set_xlabel("Time in years")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, ax


def plot_log_return_histogram(
    log_returns: FloatArray,
    title: str = "Distribution of Log Returns",
    bins: int = 50,
) -> tuple[Figure, Axes]:
    """Plot the empirical distribution of simulated or historical log returns."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(np.ravel(log_returns), bins=bins, density=True, alpha=0.75)
    ax.set_title(title)
    ax.set_xlabel("Log return")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, ax

