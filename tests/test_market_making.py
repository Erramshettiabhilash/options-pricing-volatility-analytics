"""Tests for the simplified options market-making engine."""

import pandas as pd
import pytest

from models.black_scholes import BlackScholesInputs
from simulations.market_making import (
    MarketMakingConfig,
    aggregate_risk_report,
    execute_quote_trade,
    market_making_risk_report,
    quote_option,
    simulate_market_making_session,
    theta_decay_analysis,
)
from utils.types import OptionType


def test_quote_option_places_fair_value_between_bid_and_ask() -> None:
    """A normal quote should surround fair value with positive spread."""
    inputs = BlackScholesInputs(100.0, 100.0, 1.0, 0.05, 0.20)
    quote = quote_option(inputs, OptionType.CALL, inventory=0.0)

    assert quote.bid < quote.fair_value < quote.ask
    assert quote.half_spread > 0.0


def test_toxic_order_flow_widens_spread() -> None:
    """Higher toxicity should increase the quoted half spread."""
    inputs = BlackScholesInputs(100.0, 100.0, 1.0, 0.05, 0.20)
    clean = quote_option(inputs, OptionType.CALL, inventory=0.0, order_flow_toxicity=0.0)
    toxic = quote_option(inputs, OptionType.CALL, inventory=0.0, order_flow_toxicity=1.0)

    assert toxic.half_spread > clean.half_spread


def test_long_inventory_shades_quote_lower() -> None:
    """Long inventory should shift the quote midpoint lower."""
    inputs = BlackScholesInputs(100.0, 100.0, 1.0, 0.05, 0.20)
    config = MarketMakingConfig(max_inventory=100.0)
    flat = quote_option(inputs, OptionType.CALL, inventory=0.0, config=config)
    long_inventory = quote_option(inputs, OptionType.CALL, inventory=50.0, config=config)

    assert long_inventory.mid < flat.mid


def test_execute_customer_buy_decreases_market_maker_inventory() -> None:
    """When a customer buys, the market maker sells at the ask."""
    inputs = BlackScholesInputs(100.0, 100.0, 1.0, 0.05, 0.20)
    quote = quote_option(inputs, OptionType.CALL, inventory=0.0)
    execution = execute_quote_trade(
        quote,
        trade_side="customer_buy",
        quantity=3.0,
        current_inventory=0.0,
        current_cash=0.0,
    )

    assert execution.inventory_after == pytest.approx(-3.0)
    assert execution.cash_after == pytest.approx(quote.ask * 3.0)
    assert execution.spread_capture > 0.0


def test_market_making_risk_report_and_theta_decay() -> None:
    """Risk report should aggregate Greeks and project theta decay."""
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
                "strike": 95.0,
                "time_to_maturity": 0.5,
                "volatility": 0.25,
                "option_type": "put",
                "quantity": 4.0,
            },
        ]
    )
    report = market_making_risk_report(positions, spot=100.0, risk_free_rate=0.03)
    aggregate = aggregate_risk_report(report)
    theta_projection = theta_decay_analysis(report, days=3)

    assert {"gamma_exposure", "theta_exposure", "vega_exposure"}.issubset(
        report.columns
    )
    assert set(aggregate) == {
        "market_value",
        "delta_exposure",
        "gamma_exposure",
        "vega_exposure",
        "theta_exposure",
        "rho_exposure",
    }
    assert theta_projection["day"].tolist() == [1, 2, 3]


def test_simulate_market_making_session_returns_trades_and_inventory() -> None:
    """The session simulator should produce trades and final inventory rows."""
    instruments = pd.DataFrame(
        [
            {
                "strike": 95.0,
                "time_to_maturity": 0.5,
                "volatility": 0.22,
                "option_type": "call",
            },
            {
                "strike": 105.0,
                "time_to_maturity": 1.0,
                "volatility": 0.24,
                "option_type": "put",
            },
        ]
    )

    trades, inventory = simulate_market_making_session(
        instruments,
        spot=100.0,
        risk_free_rate=0.03,
        n_trades=20,
        seed=11,
    )

    assert len(trades) == 20
    assert len(inventory) == 2
    assert trades["spread_capture"].sum() > 0.0


def test_market_making_rejects_invalid_configuration() -> None:
    """Invalid market-making configuration should fail clearly."""
    with pytest.raises(ValueError, match="max_inventory must be positive"):
        MarketMakingConfig(max_inventory=0.0).validate()

