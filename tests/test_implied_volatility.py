"""Tests for implied volatility solvers."""

import pandas as pd
import pytest

from analytics.implied_volatility import (
    extract_implied_volatility_table,
    implied_volatility,
    implied_volatility_bisection,
    implied_volatility_newton,
    no_arbitrage_price_bounds,
    validate_market_price,
)
from models.black_scholes import BlackScholesInputs, black_scholes_price
from utils.types import OptionType


def test_no_arbitrage_bounds_for_call_and_put() -> None:
    """European option prices should have clear no-arbitrage bounds."""
    inputs = BlackScholesInputs(100.0, 100.0, 1.0, 0.05, 0.20)

    call_lower, call_upper = no_arbitrage_price_bounds(inputs, OptionType.CALL)
    put_lower, put_upper = no_arbitrage_price_bounds(inputs, OptionType.PUT)

    assert call_lower == pytest.approx(4.87705755)
    assert call_upper == pytest.approx(100.0)
    assert put_lower == pytest.approx(0.0)
    assert put_upper == pytest.approx(95.12294245)


def test_bisection_recovers_known_call_volatility() -> None:
    """Bisection should recover the volatility used to generate the price."""
    true_vol = 0.20
    inputs = BlackScholesInputs(100.0, 100.0, 1.0, 0.05, true_vol)
    market_price = black_scholes_price(inputs, OptionType.CALL)

    result = implied_volatility_bisection(inputs, OptionType.CALL, market_price)

    assert result.converged
    assert result.implied_volatility == pytest.approx(true_vol, abs=1e-7)
    assert abs(result.pricing_error) < 1e-6


def test_newton_recovers_known_put_volatility() -> None:
    """Newton-Raphson should recover the volatility used to generate the price."""
    true_vol = 0.35
    inputs = BlackScholesInputs(100.0, 95.0, 0.75, 0.03, true_vol)
    market_price = black_scholes_price(inputs, OptionType.PUT)

    result = implied_volatility_newton(
        inputs,
        OptionType.PUT,
        market_price,
        initial_vol=0.25,
    )

    assert result.converged
    assert result.implied_volatility == pytest.approx(true_vol, abs=1e-7)


def test_combined_solver_falls_back_and_recovers_volatility() -> None:
    """The combined solver should return a converged IV result."""
    true_vol = 0.45
    inputs = BlackScholesInputs(100.0, 130.0, 1.5, 0.02, true_vol)
    market_price = black_scholes_price(inputs, OptionType.CALL)

    result = implied_volatility(
        inputs,
        OptionType.CALL,
        market_price,
        initial_vol=0.01,
    )

    assert result.converged
    assert result.implied_volatility == pytest.approx(true_vol, abs=1e-7)


def test_invalid_market_price_raises_value_error() -> None:
    """Prices outside no-arbitrage bounds should not produce implied vols."""
    inputs = BlackScholesInputs(100.0, 100.0, 1.0, 0.05, 0.20)

    with pytest.raises(ValueError, match="violates no-arbitrage bounds"):
        validate_market_price(inputs, OptionType.CALL, market_price=101.0)


def test_extract_implied_volatility_table() -> None:
    """A tidy option chain should produce a tidy implied-volatility table."""
    spot = 100.0
    rate = 0.05
    rows = []
    for strike, true_vol in [(90.0, 0.24), (100.0, 0.20), (110.0, 0.22)]:
        inputs = BlackScholesInputs(spot, strike, 1.0, rate, true_vol)
        rows.append(
            {
                "strike": strike,
                "time_to_maturity": 1.0,
                "option_type": "call",
                "market_price": black_scholes_price(inputs, OptionType.CALL),
            }
        )

    table = extract_implied_volatility_table(
        pd.DataFrame(rows),
        default_spot=spot,
        default_risk_free_rate=rate,
    )

    assert table["iv_converged"].all()
    assert table["implied_volatility"].tolist() == pytest.approx([0.24, 0.20, 0.22])
