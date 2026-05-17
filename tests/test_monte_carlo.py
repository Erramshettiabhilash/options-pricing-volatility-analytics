"""Tests for Monte Carlo option pricing."""

import numpy as np
import pytest

from models.black_scholes import BlackScholesInputs, black_scholes_price
from simulations.monte_carlo import (
    european_payoff,
    monte_carlo_convergence,
    monte_carlo_price,
)
from utils.types import OptionType


def test_european_call_and_put_payoffs() -> None:
    """Payoff calculations should vectorize across terminal prices."""
    terminal_prices = np.array([80.0, 100.0, 120.0])

    np.testing.assert_allclose(
        european_payoff(terminal_prices, 100.0, OptionType.CALL),
        np.array([0.0, 0.0, 20.0]),
    )
    np.testing.assert_allclose(
        european_payoff(terminal_prices, 100.0, OptionType.PUT),
        np.array([20.0, 0.0, 0.0]),
    )


def test_monte_carlo_price_is_reproducible_with_seed() -> None:
    """A fixed seed should produce repeatable research results."""
    inputs = BlackScholesInputs(100.0, 100.0, 1.0, 0.05, 0.20)

    first = monte_carlo_price(inputs, OptionType.CALL, n_paths=5_000, n_steps=10, seed=1)
    second = monte_carlo_price(
        inputs, OptionType.CALL, n_paths=5_000, n_steps=10, seed=1
    )

    assert first.price == pytest.approx(second.price)
    assert first.standard_error == pytest.approx(second.standard_error)


def test_monte_carlo_call_price_matches_black_scholes_within_sampling_error() -> None:
    """Monte Carlo should converge to the Black-Scholes European call price."""
    inputs = BlackScholesInputs(100.0, 100.0, 1.0, 0.05, 0.20)
    result = monte_carlo_price(
        inputs,
        OptionType.CALL,
        n_paths=120_000,
        n_steps=1,
        seed=42,
    )
    theoretical_price = black_scholes_price(inputs, OptionType.CALL)

    assert abs(result.price - theoretical_price) < 4.0 * result.standard_error
    assert result.confidence_interval_95[0] < result.price < result.confidence_interval_95[1]


def test_monte_carlo_can_return_full_paths() -> None:
    """The engine should optionally retain full simulated paths for visualization."""
    inputs = BlackScholesInputs(100.0, 100.0, 1.0, 0.05, 0.20)
    result = monte_carlo_price(
        inputs,
        OptionType.PUT,
        n_paths=100,
        n_steps=12,
        seed=7,
        return_paths=True,
    )

    assert result.paths is not None
    assert result.paths.shape == (100, 13)
    assert result.terminal_prices.shape == (100,)
    assert result.discounted_payoffs.shape == (100,)


def test_monte_carlo_convergence_returns_expected_columns() -> None:
    """Convergence runs should produce a tidy table for plotting and reporting."""
    inputs = BlackScholesInputs(100.0, 100.0, 1.0, 0.05, 0.20)
    table = monte_carlo_convergence(
        inputs,
        OptionType.CALL,
        path_counts=[100, 200],
        n_steps=1,
        seed=10,
    )

    assert list(table.columns) == [
        "n_paths",
        "price",
        "standard_error",
        "ci_95_low",
        "ci_95_high",
    ]
    assert table["n_paths"].tolist() == [100, 200]


def test_monte_carlo_rejects_invalid_configuration() -> None:
    """Invalid simulation settings should fail with clear errors."""
    inputs = BlackScholesInputs(100.0, 100.0, 1.0, 0.05, 0.20)

    with pytest.raises(ValueError, match="n_paths must be at least 2"):
        monte_carlo_price(inputs, OptionType.CALL, n_paths=1)
    with pytest.raises(ValueError, match="n_steps must be at least 1"):
        monte_carlo_price(inputs, OptionType.CALL, n_steps=0)

