"""Black-Scholes pricing for European vanilla options.

This module implements the Black-Scholes model from first principles. It uses
only general-purpose scientific Python tools and does not depend on any
financial pricing library.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, pi, sqrt

from utils.types import OptionType
from utils.validation import validate_non_negative, validate_positive


SQRT_TWO = sqrt(2.0)
SQRT_TWO_PI = sqrt(2.0 * pi)


@dataclass(frozen=True)
class BlackScholesInputs:
    """Inputs required to price a European option under Black-Scholes.

    Attributes:
        spot: Current underlying price, usually written as S.
        strike: Option strike price, usually written as K.
        time_to_maturity: Time to expiry in years, usually written as T.
        risk_free_rate: Continuously compounded annual risk-free rate, r.
        volatility: Annualized volatility of the underlying, sigma.
    """

    spot: float
    strike: float
    time_to_maturity: float
    risk_free_rate: float
    volatility: float

    def validate(self) -> None:
        """Validate inputs for Black-Scholes pricing."""
        validate_positive(self.spot, "spot")
        validate_positive(self.strike, "strike")
        validate_non_negative(self.time_to_maturity, "time_to_maturity")
        validate_non_negative(self.volatility, "volatility")


def standard_normal_cdf(x: float) -> float:
    """Return the standard normal cumulative distribution function N(x).

    The implementation uses the error function identity:

    ``N(x) = 0.5 * (1 + erf(x / sqrt(2)))``
    """
    return 0.5 * (1.0 + erf(x / SQRT_TWO))


def standard_normal_pdf(x: float) -> float:
    """Return the standard normal probability density function phi(x)."""
    return exp(-0.5 * x**2) / SQRT_TWO_PI


def d1(inputs: BlackScholesInputs) -> float:
    """Calculate the Black-Scholes d1 term."""
    inputs.validate()
    validate_positive(inputs.time_to_maturity, "time_to_maturity")
    validate_positive(inputs.volatility, "volatility")

    numerator = log(inputs.spot / inputs.strike) + (
        inputs.risk_free_rate + 0.5 * inputs.volatility**2
    ) * inputs.time_to_maturity
    denominator = inputs.volatility * sqrt(inputs.time_to_maturity)

    return numerator / denominator


def d2(inputs: BlackScholesInputs) -> float:
    """Calculate the Black-Scholes d2 term."""
    return d1(inputs) - inputs.volatility * sqrt(inputs.time_to_maturity)


def intrinsic_value(spot: float, strike: float, option_type: OptionType) -> float:
    """Return the option payoff if expiry happens immediately."""
    validate_positive(spot, "spot")
    validate_positive(strike, "strike")

    if option_type == OptionType.CALL:
        return max(spot - strike, 0.0)
    if option_type == OptionType.PUT:
        return max(strike - spot, 0.0)

    raise ValueError(f"Unsupported option type: {option_type}.")


def deterministic_discounted_value(
    inputs: BlackScholesInputs,
    option_type: OptionType,
) -> float:
    """Price an option when volatility is zero and the terminal price is known."""
    inputs.validate()
    discount_factor = exp(-inputs.risk_free_rate * inputs.time_to_maturity)
    forward_terminal_spot = inputs.spot / discount_factor

    if option_type == OptionType.CALL:
        return discount_factor * max(forward_terminal_spot - inputs.strike, 0.0)
    if option_type == OptionType.PUT:
        return discount_factor * max(inputs.strike - forward_terminal_spot, 0.0)

    raise ValueError(f"Unsupported option type: {option_type}.")


def black_scholes_price(
    inputs: BlackScholesInputs,
    option_type: OptionType,
) -> float:
    """Price a European call or put using the Black-Scholes formula.

    Args:
        inputs: Black-Scholes market and contract inputs.
        option_type: ``OptionType.CALL`` or ``OptionType.PUT``.

    Returns:
        The theoretical no-arbitrage option price.
    """
    inputs.validate()

    if inputs.time_to_maturity == 0:
        return intrinsic_value(inputs.spot, inputs.strike, option_type)
    if inputs.volatility == 0:
        return deterministic_discounted_value(inputs, option_type)

    d_1 = d1(inputs)
    d_2 = d_1 - inputs.volatility * sqrt(inputs.time_to_maturity)
    discount_factor = exp(-inputs.risk_free_rate * inputs.time_to_maturity)

    if option_type == OptionType.CALL:
        return (
            inputs.spot * standard_normal_cdf(d_1)
            - inputs.strike * discount_factor * standard_normal_cdf(d_2)
        )
    if option_type == OptionType.PUT:
        return (
            inputs.strike * discount_factor * standard_normal_cdf(-d_2)
            - inputs.spot * standard_normal_cdf(-d_1)
        )

    raise ValueError(f"Unsupported option type: {option_type}.")
