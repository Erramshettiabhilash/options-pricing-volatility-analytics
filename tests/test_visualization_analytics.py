"""Tests for payoff visualizations and analytics reporting helpers."""

import numpy as np
import pandas as pd
import pytest

from analytics.reporting import build_step12_analytics_summary, summarize_numeric_columns
from models.black_scholes import BlackScholesInputs, black_scholes_price
from utils.types import OptionType
from visualization.payoffs import (
    option_payoff_profile,
    option_profit_profile,
    plot_option_payoff,
    plot_strategy_payoff,
)


def test_option_payoff_profile_for_call_and_put() -> None:
    """Payoff profiles should match vanilla terminal payoff definitions."""
    terminal_prices = np.array([80.0, 100.0, 120.0])

    call = option_payoff_profile(100.0, OptionType.CALL, terminal_prices)
    put = option_payoff_profile(100.0, OptionType.PUT, terminal_prices)

    np.testing.assert_allclose(call, np.array([0.0, 0.0, 20.0]))
    np.testing.assert_allclose(put, np.array([20.0, 0.0, 0.0]))


def test_option_profit_profile_subtracts_premium() -> None:
    """Long-option profit should equal payoff less initial premium."""
    inputs = BlackScholesInputs(100.0, 100.0, 1.0, 0.05, 0.20)
    terminal_prices = np.array([100.0])
    premium = black_scholes_price(inputs, OptionType.CALL)

    profit = option_profit_profile(inputs, OptionType.CALL, terminal_prices)

    assert profit[0] == pytest.approx(-premium)


def test_payoff_plots_return_matplotlib_figures() -> None:
    """Payoff plotting functions should return Matplotlib figure and axes objects."""
    inputs = BlackScholesInputs(100.0, 100.0, 1.0, 0.05, 0.20)

    fig, ax = plot_option_payoff(inputs, OptionType.CALL)
    strategy_fig, strategy_ax = plot_strategy_payoff(
        [
            (inputs, OptionType.CALL, 1.0),
            (BlackScholesInputs(100.0, 110.0, 1.0, 0.05, 0.20), OptionType.CALL, -1.0),
        ]
    )

    assert fig is not None
    assert ax is not None
    assert strategy_fig is not None
    assert strategy_ax is not None


def test_summarize_numeric_columns() -> None:
    """Numeric summary should include common descriptive statistics."""
    table = pd.DataFrame({"pnl": [1.0, -2.0, 3.0], "vega": [10.0, 20.0, 30.0]})
    summary = summarize_numeric_columns(table, ["pnl", "vega"])

    assert summary["metric"].tolist() == ["pnl", "vega"]
    assert {"mean", "std", "min", "median", "max"}.issubset(summary.columns)


def test_build_step12_analytics_summary() -> None:
    """Headline analytics should be combined into a tidy summary table."""
    summary = build_step12_analytics_summary(
        monte_carlo_summary={"price": 10.0, "standard_error": 0.1},
        hedging_summary={"mean_pnl": 0.0},
        stress_summary={"p05_pnl": -5.0},
    )

    assert set(summary.columns) == {"section", "metric", "value"}
    assert len(summary) == 4

