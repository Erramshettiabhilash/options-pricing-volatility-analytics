"""SABR stochastic-volatility implied-volatility model.

The implementation uses Hagan's lognormal implied-volatility approximation,
which is the standard starting point for fitting volatility smiles in rates,
FX, and equity derivatives workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log, sqrt

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import least_squares

from utils.validation import validate_positive


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class SABRParameters:
    """SABR model parameters.

    Attributes:
        alpha: Initial volatility level.
        beta: Elasticity parameter, usually fixed during calibration.
        rho: Correlation between forward and volatility shocks.
        nu: Volatility of volatility.
    """

    alpha: float
    beta: float
    rho: float
    nu: float

    def validate(self) -> None:
        """Validate SABR parameter bounds."""
        validate_positive(self.alpha, "alpha")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError(f"beta must be between 0 and 1. Received {self.beta}.")
        if not -1.0 < self.rho < 1.0:
            raise ValueError(f"rho must be strictly between -1 and 1. Received {self.rho}.")
        if self.nu < 0.0:
            raise ValueError(f"nu must be non-negative. Received {self.nu}.")


@dataclass(frozen=True)
class SABRCalibrationResult:
    """Result from calibrating SABR to an implied-volatility smile."""

    parameters: SABRParameters
    rmse: float
    success: bool
    iterations: int
    calibration_table: pd.DataFrame


def _time_correction(
    forward: float,
    maturity: float,
    params: SABRParameters,
    fk_beta: float | None = None,
) -> float:
    """Return the Hagan time correction multiplier."""
    alpha = params.alpha
    beta = params.beta
    rho = params.rho
    nu = params.nu
    scale = forward ** (1.0 - beta) if fk_beta is None else fk_beta

    correction = (
        ((1.0 - beta) ** 2 / 24.0) * alpha**2 / scale**2
        + (rho * beta * nu * alpha) / (4.0 * scale)
        + ((2.0 - 3.0 * rho**2) / 24.0) * nu**2
    )

    return 1.0 + maturity * correction


def sabr_implied_volatility(
    forward: float,
    strike: float,
    maturity: float,
    params: SABRParameters,
) -> float:
    """Calculate SABR lognormal implied volatility with Hagan's approximation."""
    validate_positive(forward, "forward")
    validate_positive(strike, "strike")
    validate_positive(maturity, "maturity")
    params.validate()

    alpha = params.alpha
    beta = params.beta
    rho = params.rho
    nu = params.nu

    if abs(forward - strike) < 1e-12:
        forward_beta = forward ** (1.0 - beta)
        return alpha / forward_beta * _time_correction(
            forward,
            maturity,
            params,
            fk_beta=forward_beta,
        )

    log_fk = log(forward / strike)
    fk_beta = (forward * strike) ** ((1.0 - beta) / 2.0)
    log_adjustment = (
        1.0
        + ((1.0 - beta) ** 2 / 24.0) * log_fk**2
        + ((1.0 - beta) ** 4 / 1920.0) * log_fk**4
    )

    if nu == 0.0:
        z_over_x = 1.0
    else:
        z = (nu / alpha) * fk_beta * log_fk
        if abs(z) < 1e-8:
            z_over_x = 1.0 - 0.5 * rho * z
        else:
            x_z = log((sqrt(1.0 - 2.0 * rho * z + z**2) + z - rho) / (1.0 - rho))
            z_over_x = z / x_z

    return (
        alpha
        / (fk_beta * log_adjustment)
        * z_over_x
        * _time_correction(forward, maturity, params, fk_beta=fk_beta)
    )


def sabr_smile(
    forward: float,
    strikes: FloatArray,
    maturity: float,
    params: SABRParameters,
) -> FloatArray:
    """Calculate SABR implied volatilities for an array of strikes."""
    strikes = np.asarray(strikes, dtype=np.float64)
    if np.any(strikes <= 0):
        raise ValueError("strikes must be strictly positive.")

    return np.array(
        [
            sabr_implied_volatility(
                forward=forward,
                strike=float(strike),
                maturity=maturity,
                params=params,
            )
            for strike in strikes
        ],
        dtype=np.float64,
    )


def calibrate_sabr_smile(
    forward: float,
    strikes: FloatArray,
    market_volatilities: FloatArray,
    maturity: float,
    beta: float = 0.5,
    initial_alpha: float | None = None,
    initial_rho: float = -0.25,
    initial_nu: float = 0.50,
    weights: FloatArray | None = None,
) -> SABRCalibrationResult:
    """Calibrate alpha, rho, and nu to a market volatility smile.

    Beta is fixed because unconstrained four-parameter SABR calibration can be
    unstable for one maturity. This mirrors a common desk workflow.
    """
    validate_positive(forward, "forward")
    validate_positive(maturity, "maturity")
    if not 0.0 <= beta <= 1.0:
        raise ValueError(f"beta must be between 0 and 1. Received {beta}.")

    strikes = np.asarray(strikes, dtype=np.float64)
    market_volatilities = np.asarray(market_volatilities, dtype=np.float64)
    if strikes.shape != market_volatilities.shape:
        raise ValueError("strikes and market_volatilities must have the same shape.")
    if len(strikes) < 3:
        raise ValueError("At least three strikes are required to calibrate SABR.")
    if np.any(strikes <= 0) or np.any(market_volatilities <= 0):
        raise ValueError("strikes and market_volatilities must be strictly positive.")

    if weights is None:
        weights_array = np.ones_like(market_volatilities)
    else:
        weights_array = np.asarray(weights, dtype=np.float64)
        if weights_array.shape != market_volatilities.shape:
            raise ValueError("weights must have the same shape as market_volatilities.")
        if np.any(weights_array <= 0):
            raise ValueError("weights must be strictly positive.")

    if initial_alpha is None:
        atm_index = int(np.abs(strikes - forward).argmin())
        initial_alpha = float(market_volatilities[atm_index] * forward ** (1.0 - beta))

    def residuals(raw_params: FloatArray) -> FloatArray:
        alpha, rho, nu = raw_params
        params = SABRParameters(alpha=float(alpha), beta=beta, rho=float(rho), nu=float(nu))
        model_vols = sabr_smile(forward, strikes, maturity, params)
        return weights_array * (model_vols - market_volatilities)

    solution = least_squares(
        residuals,
        x0=np.array([initial_alpha, initial_rho, initial_nu], dtype=np.float64),
        bounds=(
            np.array([1e-8, -0.999, 1e-8], dtype=np.float64),
            np.array([10.0, 0.999, 10.0], dtype=np.float64),
        ),
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        max_nfev=2_000,
    )

    calibrated_params = SABRParameters(
        alpha=float(solution.x[0]),
        beta=beta,
        rho=float(solution.x[1]),
        nu=float(solution.x[2]),
    )
    model_vols = sabr_smile(forward, strikes, maturity, calibrated_params)
    errors = model_vols - market_volatilities
    table = pd.DataFrame(
        {
            "strike": strikes,
            "market_volatility": market_volatilities,
            "sabr_volatility": model_vols,
            "vol_error": errors,
        }
    )
    rmse = float(np.sqrt(np.mean(errors**2)))

    return SABRCalibrationResult(
        parameters=calibrated_params,
        rmse=rmse,
        success=bool(solution.success),
        iterations=int(solution.nfev),
        calibration_table=table,
    )


def calibrate_sabr_from_dataframe(
    smile: pd.DataFrame,
    forward: float,
    maturity: float,
    beta: float = 0.5,
    strike_col: str = "strike",
    volatility_col: str = "implied_volatility",
) -> SABRCalibrationResult:
    """Calibrate SABR directly from a tidy smile DataFrame."""
    required = {strike_col, volatility_col}
    missing = required.difference(smile.columns)
    if missing:
        raise ValueError(f"smile is missing required columns: {sorted(missing)}")

    sorted_smile = smile.sort_values(strike_col)

    return calibrate_sabr_smile(
        forward=forward,
        strikes=sorted_smile[strike_col].to_numpy(dtype=np.float64),
        market_volatilities=sorted_smile[volatility_col].to_numpy(dtype=np.float64),
        maturity=maturity,
        beta=beta,
    )

