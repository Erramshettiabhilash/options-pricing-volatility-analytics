"""Numerical stability and architecture-quality tests."""

import math

import numpy as np
import pandas as pd
import pytest

from analytics.greeks import calculate_greeks
from analytics.implied_volatility import implied_volatility_bisection
from analytics.stress_testing import StressScenario, portfolio_stress_test
from models.black_scholes import BlackScholesInputs, black_scholes_price
from models.sabr import SABRParameters, sabr_smile
from simulations.market_making import MarketMakingConfig, quote_option
from utils.types import OptionType


def test_black_scholes_extreme_moneyness_remains_finite() -> None:
    """Very high or low strikes should not produce NaN or infinite prices."""
    deep_otm_call = BlackScholesInputs(spot=100.0, strike=1_000.0, time_to_maturity=2.0, risk_free_rate=0.03, volatility=0.80)
    deep_itm_call = BlackScholesInputs(spot=1_000.0, strike=100.0, time_to_maturity=2.0, risk_free_rate=0.03, volatility=0.80)

    for inputs in [deep_otm_call, deep_itm_call]:
        price = black_scholes_price(inputs, OptionType.CALL)
        greeks = calculate_greeks(inputs, OptionType.CALL)

        assert math.isfinite(price)
        assert price >= 0.0
        assert all(math.isfinite(value) for value in vars(greeks).values())


def test_implied_volatility_handles_high_volatility_option_price() -> None:
    """The IV solver should recover a high but finite volatility."""
    true_vol = 1.20
    inputs = BlackScholesInputs(100.0, 120.0, 1.5, 0.02, true_vol)
    market_price = black_scholes_price(inputs, OptionType.PUT)

    result = implied_volatility_bisection(inputs, OptionType.PUT, market_price)

    assert result.converged
    assert result.implied_volatility == pytest.approx(true_vol, abs=1e-6)


def test_sabr_wing_volatilities_remain_positive_and_finite() -> None:
    """SABR smile generation should stay finite in broad strike wings."""
    params = SABRParameters(alpha=2.0, beta=0.5, rho=-0.45, nu=0.90)
    strikes = np.array([40.0, 60.0, 80.0, 100.0, 130.0, 170.0, 220.0])

    vols = sabr_smile(100.0, strikes, 2.0, params)

    assert np.isfinite(vols).all()
    assert (vols > 0.0).all()


def test_market_making_quotes_do_not_cross_under_large_inventory() -> None:
    """Inventory and toxicity adjustments should not create crossed quotes."""
    inputs = BlackScholesInputs(100.0, 100.0, 1.0, 0.03, 0.25)
    config = MarketMakingConfig(max_inventory=10.0, inventory_skew=0.50)

    quote = quote_option(
        inputs,
        OptionType.CALL,
        inventory=1_000.0,
        config=config,
        order_flow_toxicity=1.0,
    )

    assert quote.bid >= 0.0
    assert quote.ask >= quote.bid


def test_stress_testing_outputs_finite_portfolio_metrics() -> None:
    """Stress scenarios should produce finite aggregate portfolio risk values."""
    positions = pd.DataFrame(
        [
            {
                "strike": 100.0,
                "time_to_maturity": 1.0,
                "volatility": 0.20,
                "option_type": "call",
                "quantity": -10.0,
            },
            {
                "strike": 80.0,
                "time_to_maturity": 0.25,
                "volatility": 0.35,
                "option_type": "put",
                "quantity": 6.0,
            },
        ]
    )
    scenarios = [
        StressScenario("crash", spot_multiplier=0.55, volatility_shift=0.50),
        StressScenario("rate_shock", rate_shift=0.05, volatility_multiplier=1.2),
    ]

    stress = portfolio_stress_test(positions, 100.0, 0.03, scenarios)

    numeric_values = stress.select_dtypes(include=[float, int]).to_numpy()
    assert np.isfinite(numeric_values).all()

