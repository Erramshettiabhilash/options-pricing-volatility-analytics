"""Payoff and profit/loss visualizations for vanilla options."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from models.black_scholes import BlackScholesInputs, black_scholes_price, intrinsic_value
from utils.types import OptionType


FloatArray = NDArray[np.float64]


def option_payoff_profile(
    strike: float,
    option_type: OptionType,
    terminal_prices: FloatArray,
    position: float = 1.0,
) -> FloatArray:
    """Calculate terminal payoff for an option position across terminal prices."""
    if strike <= 0:
        raise ValueError(f"strike must be positive. Received {strike}.")

    terminal_prices = np.asarray(terminal_prices, dtype=np.float64)
    if np.any(terminal_prices < 0):
        raise ValueError("terminal_prices must be non-negative.")

    return position * np.array(
        [
            intrinsic_value(float(price), strike, option_type)
            for price in terminal_prices
        ],
        dtype=np.float64,
    )


def option_profit_profile(
    inputs: BlackScholesInputs,
    option_type: OptionType,
    terminal_prices: FloatArray,
    position: float = 1.0,
) -> FloatArray:
    """Calculate terminal option profit after paying or receiving premium."""
    premium = black_scholes_price(inputs, option_type)
    payoff = option_payoff_profile(
        strike=inputs.strike,
        option_type=option_type,
        terminal_prices=terminal_prices,
        position=position,
    )

    return payoff - position * premium


def plot_option_payoff(
    inputs: BlackScholesInputs,
    option_type: OptionType,
    spot_min: float | None = None,
    spot_max: float | None = None,
    n_points: int = 200,
    position: float = 1.0,
    include_profit: bool = True,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot terminal payoff and optional profit profile for a vanilla option."""
    inputs.validate()
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2. Received {n_points}.")

    lower = spot_min if spot_min is not None else 0.5 * inputs.strike
    upper = spot_max if spot_max is not None else 1.5 * inputs.strike
    if lower < 0 or upper <= lower:
        raise ValueError("spot range must be non-negative and increasing.")

    terminal_prices = np.linspace(lower, upper, n_points)
    payoff = option_payoff_profile(
        strike=inputs.strike,
        option_type=option_type,
        terminal_prices=terminal_prices,
        position=position,
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(terminal_prices, payoff, linewidth=2.2, label="Payoff")

    if include_profit:
        profit = option_profit_profile(
            inputs=inputs,
            option_type=option_type,
            terminal_prices=terminal_prices,
            position=position,
        )
        ax.plot(terminal_prices, profit, linewidth=2.2, label="Profit")

    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.axvline(inputs.strike, color="black", linestyle="--", alpha=0.4, label="Strike")
    ax.axvline(inputs.spot, color="tab:orange", linestyle=":", alpha=0.8, label="Spot")
    ax.set_title(title or f"{position:g}x {option_type.value.title()} Payoff")
    ax.set_xlabel("Underlying price at expiry")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    return fig, ax


def plot_strategy_payoff(
    legs: list[tuple[BlackScholesInputs, OptionType, float]],
    spot_min: float | None = None,
    spot_max: float | None = None,
    n_points: int = 250,
    title: str = "Option Strategy Payoff",
) -> tuple[Figure, Axes]:
    """Plot aggregate payoff and profit for a multi-leg vanilla option strategy."""
    if not legs:
        raise ValueError("legs must contain at least one option leg.")
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2. Received {n_points}.")

    strikes = [leg[0].strike for leg in legs]
    lower = spot_min if spot_min is not None else 0.5 * min(strikes)
    upper = spot_max if spot_max is not None else 1.5 * max(strikes)
    if lower < 0 or upper <= lower:
        raise ValueError("spot range must be non-negative and increasing.")

    terminal_prices = np.linspace(lower, upper, n_points)
    total_payoff = np.zeros_like(terminal_prices)
    total_profit = np.zeros_like(terminal_prices)

    for inputs, option_type, position in legs:
        total_payoff += option_payoff_profile(
            inputs.strike,
            option_type,
            terminal_prices,
            position=position,
        )
        total_profit += option_profit_profile(
            inputs,
            option_type,
            terminal_prices,
            position=position,
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(terminal_prices, total_payoff, linewidth=2.2, label="Strategy payoff")
    ax.plot(terminal_prices, total_profit, linewidth=2.2, label="Strategy profit")
    ax.axhline(0.0, color="black", linewidth=1.0)
    for strike in sorted(set(strikes)):
        ax.axvline(strike, color="black", linestyle="--", alpha=0.25)
    ax.set_title(title)
    ax.set_xlabel("Underlying price at expiry")
    ax.set_ylabel("Value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    return fig, ax

