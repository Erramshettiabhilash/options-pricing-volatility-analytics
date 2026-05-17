"""Visualization helpers for Monte Carlo option pricing."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from simulations.monte_carlo import MonteCarloResult


def plot_discounted_payoff_distribution(
    result: MonteCarloResult,
    title: str = "Discounted Monte Carlo Payoff Distribution",
    bins: int = 60,
) -> tuple[Figure, Axes]:
    """Plot the distribution of discounted pathwise option payoffs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(result.discounted_payoffs, bins=bins, alpha=0.78)
    ax.axvline(result.price, color="black", linestyle="--", linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("Discounted payoff")
    ax.set_ylabel("Path count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, ax


def plot_monte_carlo_convergence(
    convergence: pd.DataFrame,
    theoretical_price: float | None = None,
    title: str = "Monte Carlo Convergence",
) -> tuple[Figure, Axes]:
    """Plot Monte Carlo price estimates and 95% confidence intervals."""
    required_columns = {"n_paths", "price", "ci_95_low", "ci_95_high"}
    missing = required_columns.difference(convergence.columns)
    if missing:
        raise ValueError(f"convergence is missing required columns: {sorted(missing)}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        convergence["n_paths"],
        convergence["price"],
        marker="o",
        linewidth=2.0,
        label="Monte Carlo price",
    )
    ax.fill_between(
        convergence["n_paths"],
        convergence["ci_95_low"],
        convergence["ci_95_high"],
        alpha=0.2,
        label="95% confidence interval",
    )

    if theoretical_price is not None:
        ax.axhline(
            theoretical_price,
            color="black",
            linestyle="--",
            linewidth=2.0,
            label="Black-Scholes price",
        )

    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel("Number of simulated paths")
    ax.set_ylabel("Option price")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()

    return fig, ax

