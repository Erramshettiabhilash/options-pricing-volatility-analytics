"""Implied volatility solvers and smile extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp, isfinite

import pandas as pd

from analytics.greeks import vega
from models.black_scholes import BlackScholesInputs, black_scholes_price
from utils.types import OptionType
from utils.validation import validate_positive


@dataclass(frozen=True)
class ImpliedVolatilityResult:
    """Result returned by an implied volatility solver."""

    implied_volatility: float
    converged: bool
    iterations: int
    method: str
    pricing_error: float


def no_arbitrage_price_bounds(
    inputs: BlackScholesInputs,
    option_type: OptionType,
) -> tuple[float, float]:
    """Return no-arbitrage lower and upper bounds for a European option."""
    inputs.validate()
    discount_factor = exp(-inputs.risk_free_rate * inputs.time_to_maturity)
    discounted_strike = inputs.strike * discount_factor

    if option_type == OptionType.CALL:
        return max(inputs.spot - discounted_strike, 0.0), inputs.spot
    if option_type == OptionType.PUT:
        return max(discounted_strike - inputs.spot, 0.0), discounted_strike

    raise ValueError(f"Unsupported option type: {option_type}.")


def validate_market_price(
    inputs: BlackScholesInputs,
    option_type: OptionType,
    market_price: float,
    tolerance: float = 1e-12,
) -> None:
    """Validate that a market option price is finite and arbitrage-consistent."""
    inputs.validate()
    validate_positive(inputs.time_to_maturity, "time_to_maturity")
    if not isfinite(market_price):
        raise ValueError(f"market_price must be finite. Received {market_price}.")

    lower, upper = no_arbitrage_price_bounds(inputs, option_type)
    if market_price < lower - tolerance or market_price > upper + tolerance:
        raise ValueError(
            "market_price violates no-arbitrage bounds: "
            f"price={market_price}, lower={lower}, upper={upper}."
        )


def implied_volatility_bisection(
    inputs: BlackScholesInputs,
    option_type: OptionType,
    market_price: float,
    lower_vol: float = 1e-8,
    upper_vol: float = 5.0,
    tolerance: float = 1e-8,
    max_iterations: int = 200,
) -> ImpliedVolatilityResult:
    """Estimate implied volatility with the bisection method.

    Bisection is slower than Newton-Raphson but very robust when the market price
    is properly bracketed between low-vol and high-vol model prices.
    """
    validate_market_price(inputs, option_type, market_price)
    validate_positive(lower_vol, "lower_vol")
    validate_positive(upper_vol, "upper_vol")
    if upper_vol <= lower_vol:
        raise ValueError("upper_vol must be greater than lower_vol.")

    low_inputs = replace(inputs, volatility=lower_vol)
    high_inputs = replace(inputs, volatility=upper_vol)
    low_error = black_scholes_price(low_inputs, option_type) - market_price
    high_error = black_scholes_price(high_inputs, option_type) - market_price

    if abs(low_error) <= tolerance:
        return ImpliedVolatilityResult(lower_vol, True, 0, "bisection", low_error)
    if abs(high_error) <= tolerance:
        return ImpliedVolatilityResult(upper_vol, True, 0, "bisection", high_error)
    if low_error * high_error > 0:
        raise ValueError(
            "market_price is not bracketed by lower_vol and upper_vol model prices."
        )

    low = lower_vol
    high = upper_vol
    mid = 0.5 * (low + high)
    error = float("inf")

    for iteration in range(1, max_iterations + 1):
        mid = 0.5 * (low + high)
        mid_inputs = replace(inputs, volatility=mid)
        error = black_scholes_price(mid_inputs, option_type) - market_price

        if abs(error) <= tolerance or 0.5 * (high - low) <= tolerance:
            return ImpliedVolatilityResult(mid, True, iteration, "bisection", error)

        if low_error * error <= 0:
            high = mid
            high_error = error
        else:
            low = mid
            low_error = error

    return ImpliedVolatilityResult(mid, False, max_iterations, "bisection", error)


def implied_volatility_newton(
    inputs: BlackScholesInputs,
    option_type: OptionType,
    market_price: float,
    initial_vol: float = 0.20,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
    min_vol: float = 1e-8,
    max_vol: float = 5.0,
) -> ImpliedVolatilityResult:
    """Estimate implied volatility with Newton-Raphson iteration.

    Newton-Raphson updates volatility using:

    ``sigma_next = sigma - pricing_error / vega``

    It is fast near the solution, but can struggle when vega is tiny or the
    initial guess is poor.
    """
    validate_market_price(inputs, option_type, market_price)
    validate_positive(initial_vol, "initial_vol")
    validate_positive(min_vol, "min_vol")
    validate_positive(max_vol, "max_vol")
    if max_vol <= min_vol:
        raise ValueError("max_vol must be greater than min_vol.")

    sigma = min(max(initial_vol, min_vol), max_vol)
    error = float("inf")

    for iteration in range(1, max_iterations + 1):
        trial_inputs = replace(inputs, volatility=sigma)
        model_price = black_scholes_price(trial_inputs, option_type)
        error = model_price - market_price

        if abs(error) <= tolerance:
            return ImpliedVolatilityResult(sigma, True, iteration, "newton", error)

        option_vega = vega(trial_inputs)
        if abs(option_vega) < 1e-12:
            break

        sigma -= error / option_vega
        if sigma <= min_vol or sigma >= max_vol or not isfinite(sigma):
            break

    return ImpliedVolatilityResult(sigma, False, max_iterations, "newton", error)


def implied_volatility(
    inputs: BlackScholesInputs,
    option_type: OptionType,
    market_price: float,
    initial_vol: float = 0.20,
    tolerance: float = 1e-8,
    max_iterations: int = 100,
) -> ImpliedVolatilityResult:
    """Estimate implied volatility using Newton first, then Bisection fallback."""
    newton_result = implied_volatility_newton(
        inputs=inputs,
        option_type=option_type,
        market_price=market_price,
        initial_vol=initial_vol,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    if newton_result.converged:
        return newton_result

    return implied_volatility_bisection(
        inputs=inputs,
        option_type=option_type,
        market_price=market_price,
        tolerance=tolerance,
        max_iterations=max_iterations * 2,
    )


def extract_implied_volatility_table(
    option_chain: pd.DataFrame,
    default_spot: float | None = None,
    default_risk_free_rate: float | None = None,
    method: str = "bisection",
) -> pd.DataFrame:
    """Extract implied volatilities from a tidy option-chain table.

    Required columns:

    - ``strike``
    - ``time_to_maturity``
    - ``option_type``
    - ``market_price``

    Optional columns when defaults are supplied:

    - ``spot``
    - ``risk_free_rate``
    """
    required = {"strike", "time_to_maturity", "option_type", "market_price"}
    missing = required.difference(option_chain.columns)
    if missing:
        raise ValueError(f"option_chain is missing required columns: {sorted(missing)}")

    if "spot" not in option_chain.columns and default_spot is None:
        raise ValueError("default_spot is required when option_chain has no spot column.")
    if "risk_free_rate" not in option_chain.columns and default_risk_free_rate is None:
        raise ValueError(
            "default_risk_free_rate is required when option_chain has no risk_free_rate column."
        )

    rows: list[dict[str, object]] = []
    for _, row in option_chain.iterrows():
        option_type = OptionType(str(row["option_type"]).lower())
        inputs = BlackScholesInputs(
            spot=float(row["spot"]) if "spot" in option_chain.columns else float(default_spot),
            strike=float(row["strike"]),
            time_to_maturity=float(row["time_to_maturity"]),
            risk_free_rate=(
                float(row["risk_free_rate"])
                if "risk_free_rate" in option_chain.columns
                else float(default_risk_free_rate)
            ),
            volatility=0.20,
        )
        market_price = float(row["market_price"])

        if method == "newton":
            result = implied_volatility_newton(inputs, option_type, market_price)
        elif method == "bisection":
            result = implied_volatility_bisection(inputs, option_type, market_price)
        else:
            result = implied_volatility(inputs, option_type, market_price)

        output = row.to_dict()
        output.update(
            {
                "implied_volatility": result.implied_volatility,
                "iv_converged": result.converged,
                "iv_iterations": result.iterations,
                "iv_method": result.method,
                "iv_pricing_error": result.pricing_error,
            }
        )
        rows.append(output)

    return pd.DataFrame(rows)

