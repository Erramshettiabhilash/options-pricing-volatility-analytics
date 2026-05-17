"""Analytical Black-Scholes Greeks for European vanilla options."""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

from models.black_scholes import (
    BlackScholesInputs,
    d1,
    d2,
    standard_normal_cdf,
    standard_normal_pdf,
)
from utils.types import OptionType
from utils.validation import validate_positive


@dataclass(frozen=True)
class Greeks:
    """Container for first- and second-order Black-Scholes sensitivities.

    Values use the mathematical convention:

    - Vega is per 1.00 volatility change.
    - Rho is per 1.00 rate change.
    - Theta is annualized.
    """

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


def _validate_greek_inputs(inputs: BlackScholesInputs) -> None:
    """Validate inputs for Greeks that require a live option and positive vol."""
    inputs.validate()
    validate_positive(inputs.time_to_maturity, "time_to_maturity")
    validate_positive(inputs.volatility, "volatility")


def delta(inputs: BlackScholesInputs, option_type: OptionType) -> float:
    """Calculate Black-Scholes Delta.

    Delta measures sensitivity to the underlying spot price:

    ``Delta = dV / dS``
    """
    _validate_greek_inputs(inputs)
    d_1 = d1(inputs)

    if option_type == OptionType.CALL:
        return standard_normal_cdf(d_1)
    if option_type == OptionType.PUT:
        return standard_normal_cdf(d_1) - 1.0

    raise ValueError(f"Unsupported option type: {option_type}.")


def gamma(inputs: BlackScholesInputs) -> float:
    """Calculate Black-Scholes Gamma.

    Gamma measures curvature with respect to spot:

    ``Gamma = second derivative of V with respect to S``
    """
    _validate_greek_inputs(inputs)

    return standard_normal_pdf(d1(inputs)) / (
        inputs.spot * inputs.volatility * sqrt(inputs.time_to_maturity)
    )


def vega(inputs: BlackScholesInputs) -> float:
    """Calculate Black-Scholes Vega.

    Vega measures sensitivity to volatility:

    ``Vega = dV / d sigma``

    This value is per 1.00 volatility change. Divide by 100 for sensitivity to
    one volatility point.
    """
    _validate_greek_inputs(inputs)

    return inputs.spot * standard_normal_pdf(d1(inputs)) * sqrt(
        inputs.time_to_maturity
    )


def theta(inputs: BlackScholesInputs, option_type: OptionType) -> float:
    """Calculate annualized Black-Scholes Theta.

    Theta measures sensitivity to calendar time passing. The value returned here
    is annualized; divide by 365 or 252 for a daily convention.
    """
    _validate_greek_inputs(inputs)
    d_1 = d1(inputs)
    d_2 = d2(inputs)
    first_term = -(
        inputs.spot
        * standard_normal_pdf(d_1)
        * inputs.volatility
        / (2.0 * sqrt(inputs.time_to_maturity))
    )
    discounted_strike = inputs.strike * exp(
        -inputs.risk_free_rate * inputs.time_to_maturity
    )

    if option_type == OptionType.CALL:
        return first_term - inputs.risk_free_rate * discounted_strike * standard_normal_cdf(
            d_2
        )
    if option_type == OptionType.PUT:
        return first_term + inputs.risk_free_rate * discounted_strike * standard_normal_cdf(
            -d_2
        )

    raise ValueError(f"Unsupported option type: {option_type}.")


def rho(inputs: BlackScholesInputs, option_type: OptionType) -> float:
    """Calculate Black-Scholes Rho.

    Rho measures sensitivity to the continuously compounded risk-free rate:

    ``Rho = dV / dr``

    This value is per 1.00 rate change. Divide by 100 for sensitivity to one
    percentage point.
    """
    _validate_greek_inputs(inputs)
    d_2 = d2(inputs)
    discounted_strike_time = (
        inputs.strike
        * inputs.time_to_maturity
        * exp(-inputs.risk_free_rate * inputs.time_to_maturity)
    )

    if option_type == OptionType.CALL:
        return discounted_strike_time * standard_normal_cdf(d_2)
    if option_type == OptionType.PUT:
        return -discounted_strike_time * standard_normal_cdf(-d_2)

    raise ValueError(f"Unsupported option type: {option_type}.")


def calculate_greeks(inputs: BlackScholesInputs, option_type: OptionType) -> Greeks:
    """Calculate the main Black-Scholes Greeks for a European option."""
    return Greeks(
        delta=delta(inputs, option_type),
        gamma=gamma(inputs),
        vega=vega(inputs),
        theta=theta(inputs, option_type),
        rho=rho(inputs, option_type),
    )
