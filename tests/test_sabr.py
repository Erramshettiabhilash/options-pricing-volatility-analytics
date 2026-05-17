"""Tests for SABR implied volatility and calibration."""

import numpy as np
import pandas as pd
import pytest

from models.sabr import (
    SABRParameters,
    calibrate_sabr_from_dataframe,
    calibrate_sabr_smile,
    sabr_implied_volatility,
    sabr_smile,
)


def test_sabr_implied_volatility_is_positive_atm_and_off_atm() -> None:
    """Hagan SABR implied volatility should be positive for valid inputs."""
    params = SABRParameters(alpha=0.20, beta=1.0, rho=-0.30, nu=0.50)

    atm = sabr_implied_volatility(100.0, 100.0, 1.0, params)
    otm = sabr_implied_volatility(100.0, 120.0, 1.0, params)

    assert atm > 0.0
    assert otm > 0.0


def test_sabr_smile_vectorizes_over_strikes() -> None:
    """SABR smile should return one model volatility per strike."""
    params = SABRParameters(alpha=2.0, beta=0.5, rho=-0.25, nu=0.70)
    strikes = np.array([80.0, 100.0, 120.0])

    vols = sabr_smile(100.0, strikes, 1.0, params)

    assert vols.shape == strikes.shape
    assert np.all(vols > 0)


def test_sabr_calibration_recovers_synthetic_parameters() -> None:
    """Calibration should recover parameters from a synthetic SABR smile."""
    true_params = SABRParameters(alpha=2.0, beta=0.5, rho=-0.35, nu=0.80)
    forward = 100.0
    maturity = 1.0
    strikes = np.array([70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0])
    market_vols = sabr_smile(forward, strikes, maturity, true_params)

    result = calibrate_sabr_smile(
        forward=forward,
        strikes=strikes,
        market_volatilities=market_vols,
        maturity=maturity,
        beta=true_params.beta,
        initial_alpha=1.5,
        initial_rho=-0.10,
        initial_nu=0.40,
    )

    assert result.success
    assert result.rmse < 1e-8
    assert result.parameters.alpha == pytest.approx(true_params.alpha, abs=1e-5)
    assert result.parameters.rho == pytest.approx(true_params.rho, abs=1e-5)
    assert result.parameters.nu == pytest.approx(true_params.nu, abs=1e-5)


def test_sabr_calibration_from_dataframe() -> None:
    """SABR calibration should accept a tidy market-smile DataFrame."""
    params = SABRParameters(alpha=2.2, beta=0.5, rho=-0.20, nu=0.65)
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    market_vols = sabr_smile(100.0, strikes, 0.75, params)
    smile = pd.DataFrame({"strike": strikes, "implied_volatility": market_vols})

    result = calibrate_sabr_from_dataframe(smile, forward=100.0, maturity=0.75, beta=0.5)

    assert result.success
    assert set(result.calibration_table.columns) == {
        "strike",
        "market_volatility",
        "sabr_volatility",
        "vol_error",
    }
    assert result.rmse < 1e-8


def test_sabr_rejects_invalid_parameters() -> None:
    """Invalid SABR parameters should fail loudly."""
    with pytest.raises(ValueError, match="alpha must be positive"):
        SABRParameters(alpha=0.0, beta=0.5, rho=0.0, nu=0.5).validate()
    with pytest.raises(ValueError, match="beta must be between 0 and 1"):
        SABRParameters(alpha=1.0, beta=1.5, rho=0.0, nu=0.5).validate()
    with pytest.raises(ValueError, match="rho must be strictly between -1 and 1"):
        SABRParameters(alpha=1.0, beta=0.5, rho=1.0, nu=0.5).validate()
    with pytest.raises(ValueError, match="nu must be non-negative"):
        SABRParameters(alpha=1.0, beta=0.5, rho=0.0, nu=-0.5).validate()

