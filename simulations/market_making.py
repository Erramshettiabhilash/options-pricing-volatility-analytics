"""Simplified options market-making engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from analytics.greeks import Greeks, calculate_greeks
from models.black_scholes import BlackScholesInputs, black_scholes_price
from utils.types import OptionType
from utils.validation import validate_probability


TradeSide = Literal["customer_buy", "customer_sell"]


@dataclass(frozen=True)
class MarketMakingConfig:
    """Controls how aggressively the market maker quotes options."""

    base_spread_rate: float = 0.02
    min_half_spread: float = 0.01
    inventory_skew: float = 0.05
    gamma_spread_multiplier: float = 2.0
    vega_spread_multiplier: float = 0.002
    toxicity_spread_multiplier: float = 1.5
    max_inventory: float = 100.0

    def validate(self) -> None:
        """Validate market-making configuration."""
        if self.base_spread_rate < 0:
            raise ValueError("base_spread_rate must be non-negative.")
        if self.min_half_spread < 0:
            raise ValueError("min_half_spread must be non-negative.")
        if self.inventory_skew < 0:
            raise ValueError("inventory_skew must be non-negative.")
        if self.gamma_spread_multiplier < 0:
            raise ValueError("gamma_spread_multiplier must be non-negative.")
        if self.vega_spread_multiplier < 0:
            raise ValueError("vega_spread_multiplier must be non-negative.")
        if self.toxicity_spread_multiplier < 0:
            raise ValueError("toxicity_spread_multiplier must be non-negative.")
        if self.max_inventory <= 0:
            raise ValueError("max_inventory must be positive.")


@dataclass(frozen=True)
class OptionQuote:
    """Bid/ask quote and risk diagnostics for one option."""

    fair_value: float
    mid: float
    bid: float
    ask: float
    half_spread: float
    inventory: float
    inventory_skew_adjustment: float
    gamma_spread_adjustment: float
    vega_spread_adjustment: float
    toxicity_spread_adjustment: float
    greeks: Greeks


@dataclass(frozen=True)
class TradeExecution:
    """Result of one market-making trade."""

    trade_side: TradeSide
    quantity: float
    trade_price: float
    inventory_after: float
    cash_after: float
    spread_capture: float


def quote_option(
    inputs: BlackScholesInputs,
    option_type: OptionType,
    inventory: float,
    config: MarketMakingConfig | None = None,
    order_flow_toxicity: float = 0.0,
) -> OptionQuote:
    """Create a bid/ask quote around Black-Scholes fair value."""
    inputs.validate()
    if inputs.time_to_maturity <= 0:
        raise ValueError("time_to_maturity must be positive for quoting.")
    if inputs.volatility <= 0:
        raise ValueError("volatility must be positive for quoting.")

    active_config = config or MarketMakingConfig()
    active_config.validate()
    validate_probability(order_flow_toxicity, "order_flow_toxicity")

    fair_value = black_scholes_price(inputs, option_type)
    greeks = calculate_greeks(inputs, option_type)
    inventory_ratio = np.clip(inventory / active_config.max_inventory, -1.0, 1.0)
    inventory_skew_adjustment = -(
        active_config.inventory_skew
        * inventory_ratio
        * max(fair_value, 1.0)
        * max(abs(greeks.delta), 0.10)
    )
    base_half_spread = max(
        active_config.min_half_spread,
        0.5 * active_config.base_spread_rate * max(fair_value, 1.0),
    )
    gamma_spread_adjustment = (
        active_config.gamma_spread_multiplier * abs(inventory * greeks.gamma)
    )
    vega_spread_adjustment = (
        active_config.vega_spread_multiplier * abs(inventory * greeks.vega / 100.0)
    )
    toxicity_spread_adjustment = (
        base_half_spread
        * active_config.toxicity_spread_multiplier
        * order_flow_toxicity
    )
    half_spread = (
        base_half_spread
        + gamma_spread_adjustment
        + vega_spread_adjustment
        + toxicity_spread_adjustment
    )
    mid = fair_value + inventory_skew_adjustment
    bid = max(mid - half_spread, 0.0)
    ask = max(mid + half_spread, bid)

    return OptionQuote(
        fair_value=fair_value,
        mid=mid,
        bid=bid,
        ask=ask,
        half_spread=half_spread,
        inventory=inventory,
        inventory_skew_adjustment=inventory_skew_adjustment,
        gamma_spread_adjustment=gamma_spread_adjustment,
        vega_spread_adjustment=vega_spread_adjustment,
        toxicity_spread_adjustment=toxicity_spread_adjustment,
        greeks=greeks,
    )


def execute_quote_trade(
    quote: OptionQuote,
    trade_side: TradeSide,
    quantity: float,
    current_inventory: float = 0.0,
    current_cash: float = 0.0,
) -> TradeExecution:
    """Execute a customer trade against the market maker's quote."""
    if quantity <= 0:
        raise ValueError("quantity must be positive.")

    if trade_side == "customer_buy":
        trade_price = quote.ask
        inventory_after = current_inventory - quantity
        cash_after = current_cash + trade_price * quantity
    elif trade_side == "customer_sell":
        trade_price = quote.bid
        inventory_after = current_inventory + quantity
        cash_after = current_cash - trade_price * quantity
    else:
        raise ValueError(f"Unsupported trade_side: {trade_side}.")

    return TradeExecution(
        trade_side=trade_side,
        quantity=quantity,
        trade_price=trade_price,
        inventory_after=inventory_after,
        cash_after=cash_after,
        spread_capture=abs(trade_price - quote.fair_value) * quantity,
    )


def market_making_risk_report(
    positions: pd.DataFrame,
    spot: float,
    risk_free_rate: float,
) -> pd.DataFrame:
    """Calculate option-level market-making risk exposures."""
    required = {"strike", "time_to_maturity", "volatility", "option_type", "quantity"}
    missing = required.difference(positions.columns)
    if missing:
        raise ValueError(f"positions is missing required columns: {sorted(missing)}")

    rows: list[dict[str, float | str]] = []
    for _, row in positions.iterrows():
        option_type = OptionType(str(row["option_type"]).lower())
        quantity = float(row["quantity"])
        inputs = BlackScholesInputs(
            spot=spot,
            strike=float(row["strike"]),
            time_to_maturity=float(row["time_to_maturity"]),
            risk_free_rate=risk_free_rate,
            volatility=float(row["volatility"]),
        )
        fair_value = black_scholes_price(inputs, option_type)
        greeks = calculate_greeks(inputs, option_type)
        rows.append(
            {
                "option_type": option_type.value,
                "strike": inputs.strike,
                "time_to_maturity": inputs.time_to_maturity,
                "volatility": inputs.volatility,
                "quantity": quantity,
                "fair_value": fair_value,
                "market_value": quantity * fair_value,
                "delta_exposure": quantity * greeks.delta,
                "gamma_exposure": quantity * greeks.gamma,
                "vega_exposure": quantity * greeks.vega,
                "theta_exposure": quantity * greeks.theta,
                "rho_exposure": quantity * greeks.rho,
            }
        )

    return pd.DataFrame(rows)


def aggregate_risk_report(risk_report: pd.DataFrame) -> dict[str, float]:
    """Aggregate option-level market-making exposures."""
    required = {
        "market_value",
        "delta_exposure",
        "gamma_exposure",
        "vega_exposure",
        "theta_exposure",
        "rho_exposure",
    }
    missing = required.difference(risk_report.columns)
    if missing:
        raise ValueError(f"risk_report is missing required columns: {sorted(missing)}")

    return {
        "market_value": float(risk_report["market_value"].sum()),
        "delta_exposure": float(risk_report["delta_exposure"].sum()),
        "gamma_exposure": float(risk_report["gamma_exposure"].sum()),
        "vega_exposure": float(risk_report["vega_exposure"].sum()),
        "theta_exposure": float(risk_report["theta_exposure"].sum()),
        "rho_exposure": float(risk_report["rho_exposure"].sum()),
    }


def theta_decay_analysis(
    risk_report: pd.DataFrame,
    days: int = 5,
    trading_days_per_year: int = 252,
) -> pd.DataFrame:
    """Project portfolio theta decay over a short horizon."""
    if days < 1:
        raise ValueError("days must be at least 1.")
    if trading_days_per_year <= 0:
        raise ValueError("trading_days_per_year must be positive.")
    if "theta_exposure" not in risk_report.columns:
        raise ValueError("risk_report is missing required column: theta_exposure")

    annual_theta = float(risk_report["theta_exposure"].sum())
    daily_theta = annual_theta / trading_days_per_year

    return pd.DataFrame(
        {
            "day": np.arange(1, days + 1),
            "projected_theta_pnl": daily_theta * np.arange(1, days + 1),
            "daily_theta": daily_theta,
            "annual_theta": annual_theta,
        }
    )


def simulate_market_making_session(
    instruments: pd.DataFrame,
    spot: float,
    risk_free_rate: float,
    n_trades: int = 100,
    config: MarketMakingConfig | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a simple sequence of customer option trades against quotes."""
    required = {"strike", "time_to_maturity", "volatility", "option_type"}
    missing = required.difference(instruments.columns)
    if missing:
        raise ValueError(f"instruments is missing required columns: {sorted(missing)}")
    if n_trades < 1:
        raise ValueError("n_trades must be at least 1.")

    rng = np.random.default_rng(seed)
    active_config = config or MarketMakingConfig()
    active_config.validate()
    inventory: dict[int, float] = {int(index): 0.0 for index in instruments.index}
    cash = 0.0
    trade_rows: list[dict[str, float | str | int]] = []

    for trade_id in range(n_trades):
        instrument_index = int(rng.choice(instruments.index.to_numpy()))
        instrument = instruments.loc[instrument_index]
        option_type = OptionType(str(instrument["option_type"]).lower())
        inputs = BlackScholesInputs(
            spot=spot,
            strike=float(instrument["strike"]),
            time_to_maturity=float(instrument["time_to_maturity"]),
            risk_free_rate=risk_free_rate,
            volatility=float(instrument["volatility"]),
        )
        toxicity = float(rng.uniform(0.0, 0.6))
        quote = quote_option(
            inputs,
            option_type,
            inventory=inventory[instrument_index],
            config=active_config,
            order_flow_toxicity=toxicity,
        )
        trade_side: TradeSide = (
            "customer_buy" if rng.random() < 0.5 else "customer_sell"
        )
        quantity = float(rng.integers(1, 6))
        execution = execute_quote_trade(
            quote,
            trade_side,
            quantity,
            current_inventory=inventory[instrument_index],
            current_cash=cash,
        )
        inventory[instrument_index] = execution.inventory_after
        cash = execution.cash_after

        trade_rows.append(
            {
                "trade_id": trade_id,
                "instrument_id": instrument_index,
                "trade_side": trade_side,
                "quantity": quantity,
                "trade_price": execution.trade_price,
                "fair_value": quote.fair_value,
                "bid": quote.bid,
                "ask": quote.ask,
                "half_spread": quote.half_spread,
                "spread_capture": execution.spread_capture,
                "inventory_after": execution.inventory_after,
                "cash_after": execution.cash_after,
                "order_flow_toxicity": toxicity,
            }
        )

    inventory_rows = []
    for instrument_index, quantity in inventory.items():
        instrument = instruments.loc[instrument_index]
        inventory_rows.append(
            {
                "instrument_id": instrument_index,
                "strike": float(instrument["strike"]),
                "time_to_maturity": float(instrument["time_to_maturity"]),
                "volatility": float(instrument["volatility"]),
                "option_type": str(instrument["option_type"]).lower(),
                "quantity": quantity,
            }
        )

    return pd.DataFrame(trade_rows), pd.DataFrame(inventory_rows)

