"""Tests for Black-Scholes pricing."""

from math import exp

import pytest

from models.black_scholes import (
    BlackScholesInputs,
    black_scholes_price,
    d1,
    d2,
    intrinsic_value,
    standard_normal_cdf,
)
from utils.types import OptionType


def test_standard_normal_cdf_known_values() -> None:
    """The hand-built normal CDF should match familiar reference values."""
    assert standard_normal_cdf(0.0) == pytest.approx(0.5)
    assert standard_normal_cdf(1.0) == pytest.approx(0.8413447461)
    assert standard_normal_cdf(-1.0) == pytest.approx(0.1586552539)


def test_d1_and_d2_known_values() -> None:
    """d1 and d2 should match a standard at-the-money benchmark."""
    inputs = BlackScholesInputs(
        spot=100.0,
        strike=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.20,
    )

    assert d1(inputs) == pytest.approx(0.35)
    assert d2(inputs) == pytest.approx(0.15)


def test_black_scholes_call_and_put_known_values() -> None:
    """European option prices should match standard Black-Scholes benchmarks."""
    inputs = BlackScholesInputs(
        spot=100.0,
        strike=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.20,
    )

    call_price = black_scholes_price(inputs, OptionType.CALL)
    put_price = black_scholes_price(inputs, OptionType.PUT)

    assert call_price == pytest.approx(10.4505835722)
    assert put_price == pytest.approx(5.5735260223)


def test_put_call_parity() -> None:
    """Call and put prices should satisfy no-arbitrage put-call parity."""
    inputs = BlackScholesInputs(
        spot=100.0,
        strike=95.0,
        time_to_maturity=0.75,
        risk_free_rate=0.03,
        volatility=0.25,
    )

    call_price = black_scholes_price(inputs, OptionType.CALL)
    put_price = black_scholes_price(inputs, OptionType.PUT)
    parity_rhs = inputs.spot - inputs.strike * exp(
        -inputs.risk_free_rate * inputs.time_to_maturity
    )

    assert call_price - put_price == pytest.approx(parity_rhs)


def test_price_at_expiry_equals_intrinsic_value() -> None:
    """At expiry there is no optionality left, only payoff."""
    inputs = BlackScholesInputs(
        spot=105.0,
        strike=100.0,
        time_to_maturity=0.0,
        risk_free_rate=0.05,
        volatility=0.20,
    )

    assert black_scholes_price(inputs, OptionType.CALL) == pytest.approx(5.0)
    assert black_scholes_price(inputs, OptionType.PUT) == pytest.approx(0.0)
    assert intrinsic_value(95.0, 100.0, OptionType.PUT) == pytest.approx(5.0)


def test_zero_volatility_uses_deterministic_discounted_payoff() -> None:
    """With zero volatility, the risk-neutral terminal spot is deterministic."""
    inputs = BlackScholesInputs(
        spot=100.0,
        strike=95.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.0,
    )

    assert black_scholes_price(inputs, OptionType.CALL) == pytest.approx(
        100.0 - 95.0 * 2.718281828459045**-0.05
    )
    assert black_scholes_price(inputs, OptionType.PUT) == pytest.approx(0.0)


def test_invalid_inputs_raise_value_error() -> None:
    """Invalid prices, maturities, and volatilities should fail loudly."""
    bad_spot = BlackScholesInputs(0.0, 100.0, 1.0, 0.05, 0.20)
    bad_time = BlackScholesInputs(100.0, 100.0, -1.0, 0.05, 0.20)
    bad_vol = BlackScholesInputs(100.0, 100.0, 1.0, 0.05, -0.20)

    with pytest.raises(ValueError, match="spot must be positive"):
        black_scholes_price(bad_spot, OptionType.CALL)
    with pytest.raises(ValueError, match="time_to_maturity must be non-negative"):
        black_scholes_price(bad_time, OptionType.CALL)
    with pytest.raises(ValueError, match="volatility must be non-negative"):
        black_scholes_price(bad_vol, OptionType.CALL)
