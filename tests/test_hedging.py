"""Tests for dynamic hedging simulations."""

import numpy as np
import pandas as pd
import pytest

from simulations.hedging import (
    gamma_vega_exposure_table,
    hedge_neutralizing_trades,
    simulate_delta_hedge,
)
from models.black_scholes import BlackScholesInputs
from utils.types import OptionType


def test_delta_hedging_result_shapes() -> None:
    """Delta hedging should return pathwise arrays and a hedge book."""
    inputs = BlackScholesInputs(100.0, 100.0, 1.0, 0.05, 0.20)
    result = simulate_delta_hedge(
        inputs,
        OptionType.CALL,
        n_paths=100,
        n_steps=12,
        seed=1,
    )

    assert result.hedge_pnl.shape == (100,)
    assert result.stock_paths.shape == (100, 13)
    assert result.hedge_book["step"].tolist()[0] == 0
    assert result.hedge_book["step"].tolist()[-1] == 11
    assert {"mean_pnl", "std_pnl", "initial_delta"}.issubset(result.summary)


def test_delta_hedging_is_reproducible_with_seed() -> None:
    """A fixed seed should produce repeatable hedge P&L."""
    inputs = BlackScholesInputs(100.0, 100.0, 1.0, 0.05, 0.20)
    first = simulate_delta_hedge(inputs, OptionType.PUT, n_paths=200, n_steps=20, seed=7)
    second = simulate_delta_hedge(inputs, OptionType.PUT, n_paths=200, n_steps=20, seed=7)

    np.testing.assert_allclose(first.hedge_pnl, second.hedge_pnl)


def test_delta_hedging_average_error_is_small_under_matching_assumptions() -> None:
    """With matching model and realized vol, average hedge error should be modest."""
    inputs = BlackScholesInputs(100.0, 100.0, 1.0, 0.05, 0.20)
    result = simulate_delta_hedge(
        inputs,
        OptionType.CALL,
        n_paths=2_000,
        n_steps=126,
        seed=42,
    )

    assert abs(result.summary["mean_pnl"]) < 0.75
    assert result.summary["std_pnl"] > 0.0


def test_transaction_costs_reduce_short_option_hedge_pnl() -> None:
    """Trading costs should reduce average P&L for the same simulated paths."""
    inputs = BlackScholesInputs(100.0, 100.0, 1.0, 0.05, 0.20)
    without_costs = simulate_delta_hedge(
        inputs,
        OptionType.CALL,
        n_paths=500,
        n_steps=52,
        transaction_cost_rate=0.0,
        seed=9,
    )
    with_costs = simulate_delta_hedge(
        inputs,
        OptionType.CALL,
        n_paths=500,
        n_steps=52,
        transaction_cost_rate=0.001,
        seed=9,
    )

    assert with_costs.summary["mean_pnl"] < without_costs.summary["mean_pnl"]


def test_gamma_vega_exposure_table_and_neutralizing_trades() -> None:
    """Portfolio Gamma and Vega can be neutralized with two hedge instruments."""
    portfolio = pd.DataFrame(
        [
            {
                "strike": 100.0,
                "time_to_maturity": 1.0,
                "volatility": 0.20,
                "option_type": "call",
                "quantity": -10.0,
            },
            {
                "strike": 95.0,
                "time_to_maturity": 0.5,
                "volatility": 0.25,
                "option_type": "put",
                "quantity": 5.0,
            },
        ]
    )
    exposures = gamma_vega_exposure_table(portfolio, spot=100.0, risk_free_rate=0.03)
    trades = hedge_neutralizing_trades(
        exposures,
        gamma_hedge_gamma=0.05,
        gamma_hedge_vega=15.0,
        vega_hedge_gamma=0.01,
        vega_hedge_vega=40.0,
    )

    final_gamma = (
        trades["initial_gamma"]
        + trades["gamma_hedge_quantity"] * 0.05
        + trades["vega_hedge_quantity"] * 0.01
    )
    final_vega = (
        trades["initial_vega"]
        + trades["gamma_hedge_quantity"] * 15.0
        + trades["vega_hedge_quantity"] * 40.0
    )

    assert {"delta_exposure", "gamma_exposure", "vega_exposure"}.issubset(
        exposures.columns
    )
    assert final_gamma == pytest.approx(0.0)
    assert final_vega == pytest.approx(0.0)


def test_delta_hedging_rejects_invalid_inputs() -> None:
    """Invalid hedging configurations should fail clearly."""
    inputs = BlackScholesInputs(100.0, 100.0, 1.0, 0.05, 0.20)

    with pytest.raises(ValueError, match="n_steps must be at least 1"):
        simulate_delta_hedge(inputs, OptionType.CALL, n_steps=0)
    with pytest.raises(ValueError, match="transaction_cost_rate must be non-negative"):
        simulate_delta_hedge(inputs, OptionType.CALL, transaction_cost_rate=-0.01)

