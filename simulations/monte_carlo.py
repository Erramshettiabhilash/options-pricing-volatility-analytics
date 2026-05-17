"""Monte Carlo pricing engine for European options."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from models.black_scholes import BlackScholesInputs
from simulations.stochastic_processes import generate_gbm_paths
from utils.types import OptionType


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MonteCarloResult:
    """Result of a Monte Carlo option pricing run.

    Attributes:
        price: Discounted average payoff.
        standard_error: Standard error of the discounted payoff estimator.
        confidence_interval_95: Approximate 95% confidence interval.
        discounted_payoffs: Pathwise discounted payoffs.
        terminal_prices: Simulated terminal underlying prices.
        paths: Optional simulated underlying price paths.
    """

    price: float
    standard_error: float
    confidence_interval_95: tuple[float, float]
    discounted_payoffs: FloatArray
    terminal_prices: FloatArray
    paths: FloatArray | None = None


def european_payoff(
    terminal_prices: FloatArray,
    strike: float,
    option_type: OptionType,
) -> FloatArray:
    """Calculate European vanilla option payoffs from terminal prices."""
    terminal_prices = np.asarray(terminal_prices, dtype=np.float64)

    if strike <= 0:
        raise ValueError(f"strike must be positive. Received {strike}.")

    if option_type == OptionType.CALL:
        return np.maximum(terminal_prices - strike, 0.0)
    if option_type == OptionType.PUT:
        return np.maximum(strike - terminal_prices, 0.0)

    raise ValueError(f"Unsupported option type: {option_type}.")


def monte_carlo_price(
    inputs: BlackScholesInputs,
    option_type: OptionType,
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int | None = None,
    return_paths: bool = False,
) -> MonteCarloResult:
    """Price a European option with risk-neutral GBM Monte Carlo.

    Under risk-neutral pricing, GBM uses ``risk_free_rate`` as the drift:

    ``dS_t = r S_t dt + sigma S_t dW_t``
    """
    inputs.validate()
    if inputs.time_to_maturity <= 0:
        raise ValueError("time_to_maturity must be positive for Monte Carlo pricing.")
    if n_paths < 2:
        raise ValueError(f"n_paths must be at least 2. Received {n_paths}.")
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1. Received {n_steps}.")

    paths = generate_gbm_paths(
        spot=inputs.spot,
        drift=inputs.risk_free_rate,
        volatility=inputs.volatility,
        time_to_maturity=inputs.time_to_maturity,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )
    terminal_prices = paths[:, -1]
    payoffs = european_payoff(terminal_prices, inputs.strike, option_type)
    discount_factor = exp(-inputs.risk_free_rate * inputs.time_to_maturity)
    discounted_payoffs = discount_factor * payoffs

    price = float(np.mean(discounted_payoffs))
    standard_error = float(np.std(discounted_payoffs, ddof=1) / sqrt(n_paths))
    confidence_interval_95 = (
        price - 1.96 * standard_error,
        price + 1.96 * standard_error,
    )

    return MonteCarloResult(
        price=price,
        standard_error=standard_error,
        confidence_interval_95=confidence_interval_95,
        discounted_payoffs=discounted_payoffs,
        terminal_prices=terminal_prices,
        paths=paths if return_paths else None,
    )


def monte_carlo_convergence(
    inputs: BlackScholesInputs,
    option_type: OptionType,
    path_counts: list[int],
    n_steps: int = 252,
    seed: int | None = None,
) -> pd.DataFrame:
    """Run Monte Carlo pricing across increasing path counts."""
    if not path_counts:
        raise ValueError("path_counts must contain at least one value.")

    rows: list[dict[str, float | int]] = []
    for index, n_paths in enumerate(path_counts):
        result = monte_carlo_price(
            inputs=inputs,
            option_type=option_type,
            n_paths=n_paths,
            n_steps=n_steps,
            seed=None if seed is None else seed + index,
            return_paths=False,
        )
        rows.append(
            {
                "n_paths": n_paths,
                "price": result.price,
                "standard_error": result.standard_error,
                "ci_95_low": result.confidence_interval_95[0],
                "ci_95_high": result.confidence_interval_95[1],
            }
        )

    return pd.DataFrame(rows)

