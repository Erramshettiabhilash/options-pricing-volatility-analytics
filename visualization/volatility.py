"""Visualization helpers for implied volatility analytics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from analytics.volatility_surface import (
    VolatilitySurfaceGrid,
    interpolate_volatility_surface,
    pivot_volatility_surface,
)
from models.sabr import SABRCalibrationResult


def plot_volatility_smile(
    implied_vols: pd.DataFrame,
    title: str = "Implied Volatility Smile",
) -> tuple[Figure, Axes]:
    """Plot implied volatility against strike, grouped by maturity if present."""
    required = {"strike", "implied_volatility"}
    missing = required.difference(implied_vols.columns)
    if missing:
        raise ValueError(f"implied_vols is missing required columns: {sorted(missing)}")

    fig, ax = plt.subplots(figsize=(10, 6))

    if "time_to_maturity" in implied_vols.columns:
        for maturity, group in implied_vols.groupby("time_to_maturity"):
            sorted_group = group.sort_values("strike")
            ax.plot(
                sorted_group["strike"],
                sorted_group["implied_volatility"],
                marker="o",
                linewidth=2.0,
                label=f"T={maturity:g}y",
            )
        ax.legend(title="Maturity")
    else:
        sorted_vols = implied_vols.sort_values("strike")
        ax.plot(
            sorted_vols["strike"],
            sorted_vols["implied_volatility"],
            marker="o",
            linewidth=2.0,
        )

    ax.set_title(title)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied volatility")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, ax


def plot_term_structure(
    term_structure: pd.DataFrame,
    title: str = "Implied Volatility Term Structure",
) -> tuple[Figure, Axes]:
    """Plot implied volatility across maturity for a chosen moneyness bucket."""
    required = {"time_to_maturity", "implied_volatility"}
    missing = required.difference(term_structure.columns)
    if missing:
        raise ValueError(
            f"term_structure is missing required columns: {sorted(missing)}"
        )

    sorted_term = term_structure.sort_values("time_to_maturity")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        sorted_term["time_to_maturity"],
        sorted_term["implied_volatility"],
        marker="o",
        linewidth=2.0,
    )
    ax.set_title(title)
    ax.set_xlabel("Time to maturity in years")
    ax.set_ylabel("Implied volatility")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, ax


def plot_skew_by_maturity(
    skew_table: pd.DataFrame,
    title: str = "Volatility Skew by Maturity",
) -> tuple[Figure, Axes]:
    """Plot estimated IV skew slope by maturity."""
    required = {"time_to_maturity", "skew_slope"}
    missing = required.difference(skew_table.columns)
    if missing:
        raise ValueError(f"skew_table is missing required columns: {sorted(missing)}")

    sorted_skew = skew_table.sort_values("time_to_maturity")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhline(0.0, color="black", linestyle="--", alpha=0.4)
    ax.plot(
        sorted_skew["time_to_maturity"],
        sorted_skew["skew_slope"],
        marker="o",
        linewidth=2.0,
    )
    ax.set_title(title)
    ax.set_xlabel("Time to maturity in years")
    ax.set_ylabel("Slope of IV vs log-moneyness")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, ax


def plot_volatility_surface_3d(
    implied_vols: pd.DataFrame,
    title: str = "Implied Volatility Surface",
) -> tuple[Figure, Axes]:
    """Plot a 3D implied volatility surface with Matplotlib."""
    surface = pivot_volatility_surface(implied_vols)
    strikes = surface.columns.to_numpy(dtype=float)
    maturities = surface.index.to_numpy(dtype=float)
    strike_grid, maturity_grid = np.meshgrid(strikes, maturities)

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        strike_grid,
        maturity_grid,
        surface.to_numpy(dtype=float),
        cmap="viridis",
        edgecolor="none",
        alpha=0.92,
    )
    ax.set_title(title)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Time to maturity")
    ax.set_zlabel("Implied volatility")
    fig.tight_layout()

    return fig, ax


def create_plotly_volatility_surface(
    implied_vols: pd.DataFrame,
    strikes: np.ndarray | None = None,
    maturities: np.ndarray | None = None,
    title: str = "Interactive Implied Volatility Surface",
) -> go.Figure:
    """Create an interactive Plotly 3D implied volatility surface."""
    if strikes is None:
        strikes = np.sort(implied_vols["strike"].unique()).astype(float)
    if maturities is None:
        maturities = np.sort(implied_vols["time_to_maturity"].unique()).astype(float)

    grid: VolatilitySurfaceGrid = interpolate_volatility_surface(
        implied_vols,
        strikes=strikes,
        maturities=maturities,
    )

    figure = go.Figure(
        data=[
            go.Surface(
                x=grid.strikes,
                y=grid.maturities,
                z=grid.implied_volatilities,
                colorscale="Viridis",
                colorbar={"title": "IV"},
            )
        ]
    )
    figure.update_layout(
        title=title,
        scene={
            "xaxis_title": "Strike",
            "yaxis_title": "Maturity",
            "zaxis_title": "Implied volatility",
        },
        margin={"l": 0, "r": 0, "b": 0, "t": 50},
    )

    return figure


def plot_sabr_calibration(
    calibration: SABRCalibrationResult,
    title: str = "SABR Calibration vs Market Smile",
) -> tuple[Figure, Axes]:
    """Plot calibrated SABR implied vols against market implied vols."""
    table = calibration.calibration_table.sort_values("strike")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        table["strike"],
        table["market_volatility"],
        marker="o",
        linestyle="",
        label="Market IV",
    )
    ax.plot(
        table["strike"],
        table["sabr_volatility"],
        linewidth=2.0,
        label="SABR fit",
    )
    ax.set_title(title)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Implied volatility")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    return fig, ax
