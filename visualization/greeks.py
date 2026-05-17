"""Visualization helpers for Black-Scholes Greeks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from analytics.greeks import calculate_greeks
from models.black_scholes import BlackScholesInputs
from utils.types import OptionType


FloatArray = NDArray[np.float64]


def plot_greeks_vs_spot(
    base_inputs: BlackScholesInputs,
    option_type: OptionType,
    spot_min: float | None = None,
    spot_max: float | None = None,
    n_points: int = 100,
) -> tuple[Figure, NDArray[np.object_]]:
    """Plot Delta, Gamma, Vega, Theta, and Rho across spot prices."""
    base_inputs.validate()
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2. Received {n_points}.")

    lower = spot_min if spot_min is not None else 0.5 * base_inputs.strike
    upper = spot_max if spot_max is not None else 1.5 * base_inputs.strike
    if lower <= 0 or upper <= lower:
        raise ValueError("spot range must be positive and increasing.")

    spots = np.linspace(lower, upper, n_points)
    values = {
        "Delta": [],
        "Gamma": [],
        "Vega": [],
        "Theta": [],
        "Rho": [],
    }

    for spot in spots:
        inputs = BlackScholesInputs(
            spot=float(spot),
            strike=base_inputs.strike,
            time_to_maturity=base_inputs.time_to_maturity,
            risk_free_rate=base_inputs.risk_free_rate,
            volatility=base_inputs.volatility,
        )
        greeks = calculate_greeks(inputs, option_type)
        values["Delta"].append(greeks.delta)
        values["Gamma"].append(greeks.gamma)
        values["Vega"].append(greeks.vega)
        values["Theta"].append(greeks.theta)
        values["Rho"].append(greeks.rho)

    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    flat_axes = axes.ravel()

    for ax, (name, series) in zip(flat_axes, values.items(), strict=False):
        ax.plot(spots, series, linewidth=2.0)
        ax.axvline(base_inputs.strike, color="black", linestyle="--", alpha=0.35)
        ax.set_title(name)
        ax.set_xlabel("Spot price")
        ax.grid(True, alpha=0.3)

    flat_axes[-1].axis("off")
    fig.suptitle(f"Black-Scholes Greeks vs Spot ({option_type.value})", y=0.995)
    fig.tight_layout()

    return fig, axes

