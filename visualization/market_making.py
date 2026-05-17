"""Visualization helpers for options market-making analytics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def plot_inventory_by_instrument(
    inventory: pd.DataFrame,
    title: str = "Market-Maker Inventory by Strike",
) -> tuple[Figure, Axes]:
    """Plot final option inventory by strike."""
    required = {"strike", "quantity"}
    missing = required.difference(inventory.columns)
    if missing:
        raise ValueError(f"inventory is missing required columns: {sorted(missing)}")

    sorted_inventory = inventory.sort_values("strike")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(sorted_inventory["strike"].astype(str), sorted_inventory["quantity"])
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title(title)
    ax.set_xlabel("Strike")
    ax.set_ylabel("Option inventory")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    return fig, ax


def plot_cumulative_spread_capture(
    trades: pd.DataFrame,
    title: str = "Cumulative Spread Capture",
) -> tuple[Figure, Axes]:
    """Plot cumulative spread captured across simulated trades."""
    required = {"trade_id", "spread_capture"}
    missing = required.difference(trades.columns)
    if missing:
        raise ValueError(f"trades is missing required columns: {sorted(missing)}")

    sorted_trades = trades.sort_values("trade_id").copy()
    sorted_trades["cumulative_spread_capture"] = sorted_trades[
        "spread_capture"
    ].cumsum()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        sorted_trades["trade_id"],
        sorted_trades["cumulative_spread_capture"],
        linewidth=2.0,
    )
    ax.set_title(title)
    ax.set_xlabel("Trade number")
    ax.set_ylabel("Cumulative spread capture")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, ax


def plot_theta_decay(
    theta_projection: pd.DataFrame,
    title: str = "Projected Theta Decay",
) -> tuple[Figure, Axes]:
    """Plot projected theta P&L over a short horizon."""
    required = {"day", "projected_theta_pnl"}
    missing = required.difference(theta_projection.columns)
    if missing:
        raise ValueError(
            f"theta_projection is missing required columns: {sorted(missing)}"
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        theta_projection["day"],
        theta_projection["projected_theta_pnl"],
        marker="o",
        linewidth=2.0,
    )
    ax.axhline(0.0, color="black", linestyle="--", alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel("Day")
    ax.set_ylabel("Projected theta P&L")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, ax

