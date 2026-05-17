"""Tests for volatility surface analytics."""

import numpy as np
import pandas as pd
import pytest

from analytics.volatility_surface import (
    add_moneyness_columns,
    build_implied_volatility_surface,
    interpolate_volatility_surface,
    pivot_volatility_surface,
    skew_slope_by_maturity,
    smile_for_maturity,
    synthetic_equity_index_volatility,
    term_structure_at_moneyness,
)
from models.black_scholes import BlackScholesInputs, black_scholes_price
from utils.types import OptionType


def _sample_iv_table() -> pd.DataFrame:
    rows = []
    spot = 100.0
    for maturity in [0.5, 1.0]:
        for strike in [90.0, 100.0, 110.0]:
            implied_volatility = 0.20 - 0.08 * np.log(strike / spot) + 0.01 * maturity
            rows.append(
                {
                    "spot": spot,
                    "strike": strike,
                    "time_to_maturity": maturity,
                    "implied_volatility": implied_volatility,
                }
            )
    return add_moneyness_columns(pd.DataFrame(rows))


def test_add_moneyness_columns() -> None:
    """Moneyness columns should be added without mutating the input table."""
    table = pd.DataFrame({"spot": [100.0], "strike": [110.0]})
    output = add_moneyness_columns(table)

    assert "moneyness" not in table.columns
    assert output.loc[0, "moneyness"] == pytest.approx(1.10)
    assert output.loc[0, "log_moneyness"] == pytest.approx(np.log(1.10))


def test_smile_for_maturity_returns_sorted_strikes() -> None:
    """A smile should contain one maturity and sorted strikes."""
    table = _sample_iv_table()
    smile = smile_for_maturity(table, 1.0)

    assert smile["time_to_maturity"].nunique() == 1
    assert smile["strike"].tolist() == [90.0, 100.0, 110.0]


def test_term_structure_at_moneyness_uses_nearest_bucket() -> None:
    """Term structure should select one nearest-moneyness quote per maturity."""
    table = _sample_iv_table()
    term = term_structure_at_moneyness(table, target_moneyness=1.0)

    assert term["time_to_maturity"].tolist() == [0.5, 1.0]
    assert term["strike"].tolist() == [100.0, 100.0]


def test_skew_slope_by_maturity_is_negative_for_equity_skew() -> None:
    """The sample surface has negative IV slope versus log-moneyness."""
    table = _sample_iv_table()
    skew = skew_slope_by_maturity(table)

    assert (skew["skew_slope"] < 0).all()


def test_pivot_and_interpolate_surface() -> None:
    """Surface pivoting and interpolation should produce regular grids."""
    table = _sample_iv_table()
    pivot = pivot_volatility_surface(table)
    grid = interpolate_volatility_surface(
        table,
        strikes=np.array([90.0, 100.0, 110.0]),
        maturities=np.array([0.5, 0.75, 1.0]),
    )

    assert pivot.shape == (2, 3)
    assert grid.implied_volatilities.shape == (3, 3)
    assert np.isfinite(grid.implied_volatilities).all()


def test_build_implied_volatility_surface_from_option_chain() -> None:
    """Option prices should be converted into an enriched IV surface table."""
    spot = 100.0
    rate = 0.03
    rows = []
    for maturity in [0.5, 1.0]:
        for strike in [90.0, 100.0, 110.0]:
            true_vol = synthetic_equity_index_volatility(strike, maturity, spot=spot)
            inputs = BlackScholesInputs(spot, strike, maturity, rate, true_vol)
            rows.append(
                {
                    "spot": spot,
                    "strike": strike,
                    "time_to_maturity": maturity,
                    "risk_free_rate": rate,
                    "option_type": "call",
                    "market_price": black_scholes_price(inputs, OptionType.CALL),
                }
            )

    surface = build_implied_volatility_surface(pd.DataFrame(rows), method="bisection")

    assert {"moneyness", "log_moneyness", "implied_volatility"}.issubset(
        surface.columns
    )
    assert surface["iv_converged"].all()

