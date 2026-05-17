"""Visualization helpers for stress-testing analytics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_scenario_pnl(
    stress_table: pd.DataFrame,
    title: str = "Scenario Stress P&L",
) -> tuple[Figure, Axes]:
    """Plot portfolio P&L across deterministic stress scenarios."""
    required = {"scenario", "pnl"}
    missing = required.difference(stress_table.columns)
    if missing:
        raise ValueError(f"stress_table is missing required columns: {sorted(missing)}")

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["tab:red" if value < 0 else "tab:green" for value in stress_table["pnl"]]
    ax.bar(stress_table["scenario"], stress_table["pnl"], color=colors, alpha=0.82)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Portfolio P&L")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    return fig, ax


def plot_stress_loss_distribution(
    stress_paths: pd.DataFrame,
    title: str = "Monte Carlo Stress P&L Distribution",
    bins: int = 70,
) -> tuple[Figure, Axes]:
    """Plot simulated stressed portfolio P&L."""
    if "pnl" not in stress_paths.columns:
        raise ValueError("stress_paths is missing required column: pnl")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(stress_paths["pnl"], bins=bins, alpha=0.78)
    ax.axvline(stress_paths["pnl"].mean(), color="black", linestyle="--", linewidth=2)
    ax.axvline(
        stress_paths["pnl"].quantile(0.05),
        color="tab:red",
        linestyle="--",
        linewidth=2,
        label="5% tail",
    )
    ax.set_title(title)
    ax.set_xlabel("Portfolio P&L")
    ax.set_ylabel("Path count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    return fig, ax


def plot_greek_exposures_under_stress(
    stress_table: pd.DataFrame,
    title: str = "Greek Exposures Under Stress",
) -> tuple[Figure, Axes]:
    """Plot stressed Delta, Gamma, Vega, and Theta exposures by scenario."""
    required = {
        "scenario",
        "delta_exposure",
        "gamma_exposure",
        "vega_exposure",
        "theta_exposure",
    }
    missing = required.difference(stress_table.columns)
    if missing:
        raise ValueError(f"stress_table is missing required columns: {sorted(missing)}")

    exposures = stress_table.set_index("scenario")[
        ["delta_exposure", "gamma_exposure", "vega_exposure", "theta_exposure"]
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    exposures.plot(kind="bar", ax=ax)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Scenario")
    ax.set_ylabel("Exposure")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    return fig, ax

