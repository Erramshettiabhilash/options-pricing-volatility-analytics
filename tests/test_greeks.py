"""Tests for analytical Black-Scholes Greeks."""

import pytest

from analytics.greeks import calculate_greeks, delta, gamma, rho, theta, vega
from models.black_scholes import BlackScholesInputs, standard_normal_pdf
from utils.types import OptionType


def test_standard_normal_pdf_known_values() -> None:
    """The normal density should match common reference values."""
    assert standard_normal_pdf(0.0) == pytest.approx(0.3989422804)
    assert standard_normal_pdf(1.0) == pytest.approx(0.2419707245)


def test_call_greeks_known_values() -> None:
    """Call Greeks should match a standard Black-Scholes benchmark."""
    inputs = BlackScholesInputs(
        spot=100.0,
        strike=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.20,
    )

    greeks = calculate_greeks(inputs, OptionType.CALL)

    assert greeks.delta == pytest.approx(0.6368306512)
    assert greeks.gamma == pytest.approx(0.0187620173)
    assert greeks.vega == pytest.approx(37.5240346917)
    assert greeks.theta == pytest.approx(-6.4140275464)
    assert greeks.rho == pytest.approx(53.2324815454)


def test_put_greeks_known_values() -> None:
    """Put Greeks should match a standard Black-Scholes benchmark."""
    inputs = BlackScholesInputs(
        spot=100.0,
        strike=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.20,
    )

    greeks = calculate_greeks(inputs, OptionType.PUT)

    assert greeks.delta == pytest.approx(-0.3631693488)
    assert greeks.gamma == pytest.approx(0.0187620173)
    assert greeks.vega == pytest.approx(37.5240346917)
    assert greeks.theta == pytest.approx(-1.6578804239)
    assert greeks.rho == pytest.approx(-41.8904609047)


def test_individual_greek_functions_match_container() -> None:
    """Individual functions and the aggregate container should stay consistent."""
    inputs = BlackScholesInputs(120.0, 100.0, 0.5, 0.04, 0.30)
    greeks = calculate_greeks(inputs, OptionType.CALL)

    assert greeks.delta == pytest.approx(delta(inputs, OptionType.CALL))
    assert greeks.gamma == pytest.approx(gamma(inputs))
    assert greeks.vega == pytest.approx(vega(inputs))
    assert greeks.theta == pytest.approx(theta(inputs, OptionType.CALL))
    assert greeks.rho == pytest.approx(rho(inputs, OptionType.CALL))


def test_greeks_reject_expired_or_zero_volatility_options() -> None:
    """Analytical Greeks are singular at expiry or zero volatility."""
    expired = BlackScholesInputs(100.0, 100.0, 0.0, 0.05, 0.20)
    zero_vol = BlackScholesInputs(100.0, 100.0, 1.0, 0.05, 0.0)

    with pytest.raises(ValueError, match="time_to_maturity must be positive"):
        calculate_greeks(expired, OptionType.CALL)
    with pytest.raises(ValueError, match="volatility must be positive"):
        calculate_greeks(zero_vol, OptionType.CALL)

