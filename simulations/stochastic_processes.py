"""Stochastic process utilities for derivatives simulations.

The functions in this module are intentionally small and explicit. They are the
foundation for later Monte Carlo pricing, hedging simulations, and stress tests.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from utils.validation import validate_positive


FloatArray = NDArray[np.float64]


def time_grid(time_to_maturity: float, n_steps: int) -> FloatArray:
    """Create an evenly spaced time grid from 0 to maturity.

    Args:
        time_to_maturity: Time horizon in years.
        n_steps: Number of simulation steps.

    Returns:
        Array of length ``n_steps + 1`` including both 0 and maturity.
    """
    validate_positive(time_to_maturity, "time_to_maturity")
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1. Received {n_steps}.")

    return np.linspace(0.0, time_to_maturity, n_steps + 1, dtype=np.float64)


def generate_brownian_motion(
    time_to_maturity: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> FloatArray:
    """Generate standard Brownian motion paths.

    Brownian motion starts at zero and evolves through independent normal
    increments with variance equal to the time step.

    Args:
        time_to_maturity: Time horizon in years.
        n_steps: Number of simulation steps.
        n_paths: Number of independent paths.
        seed: Optional random seed for reproducible research.

    Returns:
        Array with shape ``(n_paths, n_steps + 1)``.
    """
    validate_positive(time_to_maturity, "time_to_maturity")
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1. Received {n_steps}.")
    if n_paths < 1:
        raise ValueError(f"n_paths must be at least 1. Received {n_paths}.")

    rng = np.random.default_rng(seed)
    dt = time_to_maturity / n_steps
    increments = rng.normal(
        loc=0.0,
        scale=np.sqrt(dt),
        size=(n_paths, n_steps),
    )
    brownian_paths = np.concatenate(
        [np.zeros((n_paths, 1), dtype=np.float64), np.cumsum(increments, axis=1)],
        axis=1,
    )

    return brownian_paths


def generate_gbm_paths(
    spot: float,
    drift: float,
    volatility: float,
    time_to_maturity: float,
    n_steps: int,
    n_paths: int,
    seed: int | None = None,
) -> FloatArray:
    """Generate Geometric Brownian Motion price paths.

    The exact GBM solution is:

    ``S_t = S_0 * exp((mu - 0.5 * sigma^2) * t + sigma * W_t)``

    Args:
        spot: Initial asset price.
        drift: Annualized drift parameter.
        volatility: Annualized volatility parameter.
        time_to_maturity: Time horizon in years.
        n_steps: Number of simulation steps.
        n_paths: Number of independent paths.
        seed: Optional random seed for reproducible research.

    Returns:
        Simulated price paths with shape ``(n_paths, n_steps + 1)``.
    """
    validate_positive(spot, "spot")
    validate_positive(time_to_maturity, "time_to_maturity")
    if volatility < 0:
        raise ValueError(f"volatility must be non-negative. Received {volatility}.")

    times = time_grid(time_to_maturity, n_steps)
    brownian_paths = generate_brownian_motion(
        time_to_maturity=time_to_maturity,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )
    exponent = (drift - 0.5 * volatility**2) * times + volatility * brownian_paths

    return spot * np.exp(exponent)


def calculate_log_returns(prices: FloatArray, axis: int = -1) -> FloatArray:
    """Calculate continuously compounded returns from a price array.

    Args:
        prices: Positive price observations.
        axis: Axis along which consecutive returns are calculated.

    Returns:
        Log returns with one fewer observation along ``axis``.
    """
    prices = np.asarray(prices, dtype=np.float64)
    if np.any(prices <= 0):
        raise ValueError("prices must be strictly positive to calculate log returns.")

    return np.diff(np.log(prices), axis=axis)

