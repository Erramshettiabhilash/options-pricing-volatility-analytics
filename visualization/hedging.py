"""Visualization helpers for dynamic hedging simulations."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from simulations.hedging import DeltaHedgingResult


def plot_hedge_pnl_distribution(
    result: DeltaHedgingResult,
    title: str = "Delta-Hedging P&L Distribution",
    bins: int = 60,
) -> tuple[Figure, Axes]:
    """Plot the terminal distribution of hedge P&L."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(result.hedge_pnl, bins=bins, alpha=0.78)
    ax.axvline(result.summary["mean_pnl"], color="black", linestyle="--", linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("Terminal hedge P&L")
    ax.set_ylabel("Path count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, ax


def plot_mean_hedge_book(
    result: DeltaHedgingResult,
    title: str = "Average Delta Hedge Through Time",
) -> tuple[Figure, Axes]:
    """Plot the average stock hedge and average spot path through time."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        result.hedge_book["time"],
        result.hedge_book["mean_stock_position"],
        linewidth=2.0,
        label="Mean stock hedge",
    )
    ax.set_title(title)
    ax.set_xlabel("Time in years")
    ax.set_ylabel("Mean stock position")
    ax.grid(True, alpha=0.3)

    twin = ax.twinx()
    twin.plot(
        result.hedge_book["time"],
        result.hedge_book["mean_spot"],
        color="tab:orange",
        linewidth=1.7,
        alpha=0.8,
        label="Mean spot",
    )
    twin.set_ylabel("Mean spot")

    lines, labels = ax.get_legend_handles_labels()
    twin_lines, twin_labels = twin.get_legend_handles_labels()
    ax.legend(lines + twin_lines, labels + twin_labels, loc="best")
    fig.tight_layout()

    return fig, ax

