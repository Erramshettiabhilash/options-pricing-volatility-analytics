"""Dynamic Greeks hedging simulations."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from analytics.greeks import calculate_greeks, delta, gamma, vega
from models.black_scholes import BlackScholesInputs, black_scholes_price, intrinsic_value
from simulations.stochastic_processes import generate_gbm_paths, time_grid
from utils.types import OptionType


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DeltaHedgingResult:
    """Pathwise result of a dynamic delta-hedging simulation."""

    hedge_pnl: FloatArray
    terminal_prices: FloatArray
    option_payoffs: FloatArray
    final_cash: FloatArray
    final_stock_position: FloatArray
    stock_paths: FloatArray
    time_grid: FloatArray
    hedge_book: pd.DataFrame
    summary: dict[str, float]


def _pathwise_delta(
    spots: FloatArray,
    strike: float,
    time_to_maturity: float,
    risk_free_rate: float,
    volatility: float,
    option_type: OptionType,
) -> FloatArray:
    """Calculate Black-Scholes delta for each spot in a vector."""
    return np.array(
        [
            delta(
                BlackScholesInputs(
                    spot=float(spot),
                    strike=strike,
                    time_to_maturity=time_to_maturity,
                    risk_free_rate=risk_free_rate,
                    volatility=volatility,
                ),
                option_type,
            )
            for spot in spots
        ],
        dtype=np.float64,
    )


def simulate_delta_hedge(
    inputs: BlackScholesInputs,
    option_type: OptionType,
    n_paths: int = 5_000,
    n_steps: int = 252,
    realized_volatility: float | None = None,
    drift: float | None = None,
    hedge_volatility: float | None = None,
    option_position: float = -1.0,
    transaction_cost_rate: float = 0.0,
    seed: int | None = None,
) -> DeltaHedgingResult:
    """Simulate dynamic delta hedging for a European option.

    Args:
        inputs: Contract and pricing assumptions.
        option_type: Option type being hedged.
        n_paths: Number of simulated market paths.
        n_steps: Number of hedge intervals.
        realized_volatility: Volatility used to simulate the underlying.
        drift: Physical drift used to simulate the underlying.
        hedge_volatility: Volatility used by the hedger to compute delta.
        option_position: Option quantity. ``-1`` means short one option.
        transaction_cost_rate: Proportional cost paid on stock notional traded.
        seed: Optional random seed.

    Returns:
        Pathwise hedging P&L and summary diagnostics.
    """
    inputs.validate()
    if inputs.time_to_maturity <= 0:
        raise ValueError("time_to_maturity must be positive for hedging simulation.")
    if inputs.volatility <= 0:
        raise ValueError("volatility must be positive for hedging simulation.")
    if n_paths < 1:
        raise ValueError(f"n_paths must be at least 1. Received {n_paths}.")
    if n_steps < 1:
        raise ValueError(f"n_steps must be at least 1. Received {n_steps}.")
    if transaction_cost_rate < 0:
        raise ValueError("transaction_cost_rate must be non-negative.")

    realized_sigma = inputs.volatility if realized_volatility is None else realized_volatility
    hedge_sigma = inputs.volatility if hedge_volatility is None else hedge_volatility
    path_drift = inputs.risk_free_rate if drift is None else drift
    if realized_sigma < 0:
        raise ValueError("realized_volatility must be non-negative.")
    if hedge_sigma <= 0:
        raise ValueError("hedge_volatility must be positive.")

    dt = inputs.time_to_maturity / n_steps
    times = time_grid(inputs.time_to_maturity, n_steps)
    stock_paths = generate_gbm_paths(
        spot=inputs.spot,
        drift=path_drift,
        volatility=realized_sigma,
        time_to_maturity=inputs.time_to_maturity,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
    )

    initial_price = black_scholes_price(inputs, option_type)
    initial_delta = delta(inputs, option_type)
    stock_position = np.full(
        n_paths,
        -option_position * initial_delta,
        dtype=np.float64,
    )
    cash = np.full(
        n_paths,
        -option_position * initial_price - stock_position[0] * inputs.spot,
        dtype=np.float64,
    )
    cash -= np.abs(stock_position) * inputs.spot * transaction_cost_rate

    hedge_rows = [
        {
            "step": 0,
            "time": 0.0,
            "mean_spot": inputs.spot,
            "mean_delta": initial_delta,
            "mean_stock_position": float(np.mean(stock_position)),
            "mean_cash": float(np.mean(cash)),
            "mean_trade": float(np.mean(stock_position)),
        }
    ]

    for step in range(1, n_steps):
        cash *= exp(inputs.risk_free_rate * dt)
        current_spots = stock_paths[:, step]
        remaining_time = inputs.time_to_maturity - step * dt
        current_delta = _pathwise_delta(
            current_spots,
            strike=inputs.strike,
            time_to_maturity=remaining_time,
            risk_free_rate=inputs.risk_free_rate,
            volatility=hedge_sigma,
            option_type=option_type,
        )
        desired_stock_position = -option_position * current_delta
        trade = desired_stock_position - stock_position
        cash -= trade * current_spots
        cash -= np.abs(trade) * current_spots * transaction_cost_rate
        stock_position = desired_stock_position

        hedge_rows.append(
            {
                "step": step,
                "time": times[step],
                "mean_spot": float(np.mean(current_spots)),
                "mean_delta": float(np.mean(current_delta)),
                "mean_stock_position": float(np.mean(stock_position)),
                "mean_cash": float(np.mean(cash)),
                "mean_trade": float(np.mean(trade)),
            }
        )

    cash *= exp(inputs.risk_free_rate * dt)
    terminal_prices = stock_paths[:, -1]
    option_payoffs = np.array(
        [
            intrinsic_value(float(spot), inputs.strike, option_type)
            for spot in terminal_prices
        ],
        dtype=np.float64,
    )
    hedge_pnl = cash + stock_position * terminal_prices + option_position * option_payoffs
    hedge_pnl -= np.abs(stock_position) * terminal_prices * transaction_cost_rate

    summary = {
        "mean_pnl": float(np.mean(hedge_pnl)),
        "std_pnl": float(np.std(hedge_pnl, ddof=1)) if n_paths > 1 else 0.0,
        "median_pnl": float(np.median(hedge_pnl)),
        "p05_pnl": float(np.percentile(hedge_pnl, 5.0)),
        "p95_pnl": float(np.percentile(hedge_pnl, 95.0)),
        "mean_terminal_price": float(np.mean(terminal_prices)),
        "initial_option_price": float(initial_price),
        "initial_delta": float(initial_delta),
    }

    return DeltaHedgingResult(
        hedge_pnl=hedge_pnl,
        terminal_prices=terminal_prices,
        option_payoffs=option_payoffs,
        final_cash=cash,
        final_stock_position=stock_position,
        stock_paths=stock_paths,
        time_grid=times,
        hedge_book=pd.DataFrame(hedge_rows),
        summary=summary,
    )


def gamma_vega_exposure_table(
    options: pd.DataFrame,
    spot: float,
    risk_free_rate: float,
) -> pd.DataFrame:
    """Calculate option-level Delta, Gamma, and Vega exposures for a portfolio.

    Required columns are ``strike``, ``time_to_maturity``, ``volatility``,
    ``option_type``, and ``quantity``.
    """
    required = {"strike", "time_to_maturity", "volatility", "option_type", "quantity"}
    missing = required.difference(options.columns)
    if missing:
        raise ValueError(f"options is missing required columns: {sorted(missing)}")

    rows: list[dict[str, float | str]] = []
    for _, row in options.iterrows():
        option_type = OptionType(str(row["option_type"]).lower())
        quantity = float(row["quantity"])
        inputs = BlackScholesInputs(
            spot=spot,
            strike=float(row["strike"]),
            time_to_maturity=float(row["time_to_maturity"]),
            risk_free_rate=risk_free_rate,
            volatility=float(row["volatility"]),
        )
        greeks = calculate_greeks(inputs, option_type)
        rows.append(
            {
                "option_type": option_type.value,
                "strike": inputs.strike,
                "time_to_maturity": inputs.time_to_maturity,
                "quantity": quantity,
                "delta_exposure": quantity * greeks.delta,
                "gamma_exposure": quantity * greeks.gamma,
                "vega_exposure": quantity * greeks.vega,
            }
        )

    return pd.DataFrame(rows)


def hedge_neutralizing_trades(
    exposure_table: pd.DataFrame,
    gamma_hedge_gamma: float,
    gamma_hedge_vega: float,
    vega_hedge_gamma: float,
    vega_hedge_vega: float,
) -> dict[str, float]:
    """Solve two hedge quantities that neutralize Gamma and Vega exposures."""
    required = {"gamma_exposure", "vega_exposure"}
    missing = required.difference(exposure_table.columns)
    if missing:
        raise ValueError(f"exposure_table is missing required columns: {sorted(missing)}")

    current_gamma = float(exposure_table["gamma_exposure"].sum())
    current_vega = float(exposure_table["vega_exposure"].sum())
    hedge_matrix = np.array(
        [
            [gamma_hedge_gamma, vega_hedge_gamma],
            [gamma_hedge_vega, vega_hedge_vega],
        ],
        dtype=np.float64,
    )
    target = np.array([-current_gamma, -current_vega], dtype=np.float64)
    quantities = np.linalg.solve(hedge_matrix, target)

    return {
        "gamma_hedge_quantity": float(quantities[0]),
        "vega_hedge_quantity": float(quantities[1]),
        "initial_gamma": current_gamma,
        "initial_vega": current_vega,
    }

