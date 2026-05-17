"""Tests for stochastic-process simulation utilities."""

import numpy as np
import pytest

from simulations.stochastic_processes import (
    calculate_log_returns,
    generate_brownian_motion,
    generate_gbm_paths,
    time_grid,
)


def test_time_grid_includes_start_and_maturity() -> None:
    """The time grid should include both endpoints."""
    grid = time_grid(time_to_maturity=1.0, n_steps=4)

    np.testing.assert_allclose(grid, np.array([0.0, 0.25, 0.5, 0.75, 1.0]))


def test_brownian_motion_shape_and_initial_value() -> None:
    """Brownian paths should start at zero and have one more column than steps."""
    paths = generate_brownian_motion(
        time_to_maturity=1.0,
        n_steps=252,
        n_paths=10,
        seed=7,
    )

    assert paths.shape == (10, 253)
    np.testing.assert_allclose(paths[:, 0], 0.0)


def test_brownian_motion_is_reproducible_with_seed() -> None:
    """A fixed seed should produce reproducible paths for research notebooks."""
    first = generate_brownian_motion(1.0, 10, 3, seed=123)
    second = generate_brownian_motion(1.0, 10, 3, seed=123)

    np.testing.assert_allclose(first, second)


def test_gbm_paths_are_positive_and_start_at_spot() -> None:
    """GBM prices should remain positive and start at the requested spot."""
    spot = 100.0
    paths = generate_gbm_paths(
        spot=spot,
        drift=0.05,
        volatility=0.2,
        time_to_maturity=1.0,
        n_steps=252,
        n_paths=20,
        seed=42,
    )

    assert paths.shape == (20, 253)
    assert np.all(paths > 0)
    np.testing.assert_allclose(paths[:, 0], spot)


def test_gbm_without_volatility_matches_deterministic_growth() -> None:
    """With zero volatility, GBM should collapse to deterministic exponential growth."""
    spot = 100.0
    drift = 0.05
    maturity = 1.0
    steps = 4
    paths = generate_gbm_paths(
        spot=spot,
        drift=drift,
        volatility=0.0,
        time_to_maturity=maturity,
        n_steps=steps,
        n_paths=2,
        seed=42,
    )
    expected = spot * np.exp(drift * time_grid(maturity, steps))

    np.testing.assert_allclose(paths[0], expected)
    np.testing.assert_allclose(paths[1], expected)


def test_calculate_log_returns() -> None:
    """Log returns should equal log of consecutive price ratios."""
    prices = np.array([100.0, 105.0, 103.0])
    expected = np.array([np.log(105.0 / 100.0), np.log(103.0 / 105.0)])

    np.testing.assert_allclose(calculate_log_returns(prices), expected)


def test_log_returns_reject_non_positive_prices() -> None:
    """Log returns are undefined for zero or negative prices."""
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_log_returns(np.array([100.0, 0.0, 101.0]))

