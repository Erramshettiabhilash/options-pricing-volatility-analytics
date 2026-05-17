"""Volatility smile, skew, term-structure, and surface analytics."""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.interpolate import griddata

from analytics.implied_volatility import extract_implied_volatility_table


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class VolatilitySurfaceGrid:
    """Regular grid representation of an implied volatility surface."""

    strikes: FloatArray
    maturities: FloatArray
    implied_volatilities: FloatArray


def add_moneyness_columns(
    implied_vols: pd.DataFrame,
    spot_col: str = "spot",
    strike_col: str = "strike",
) -> pd.DataFrame:
    """Add moneyness and log-moneyness columns to an implied-vol table."""
    required = {spot_col, strike_col}
    missing = required.difference(implied_vols.columns)
    if missing:
        raise ValueError(f"implied_vols is missing required columns: {sorted(missing)}")

    output = implied_vols.copy()
    output["moneyness"] = output[strike_col] / output[spot_col]
    output["log_moneyness"] = np.log(output[strike_col] / output[spot_col])

    return output


def build_implied_volatility_surface(
    option_chain: pd.DataFrame,
    default_spot: float | None = None,
    default_risk_free_rate: float | None = None,
    method: str = "bisection",
) -> pd.DataFrame:
    """Extract implied volatilities and enrich them with moneyness columns."""
    implied_vols = extract_implied_volatility_table(
        option_chain=option_chain,
        default_spot=default_spot,
        default_risk_free_rate=default_risk_free_rate,
        method=method,
    )

    return add_moneyness_columns(implied_vols)


def smile_for_maturity(
    implied_vols: pd.DataFrame,
    maturity: float,
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Return the volatility smile for one maturity."""
    required = {"time_to_maturity", "strike", "implied_volatility"}
    missing = required.difference(implied_vols.columns)
    if missing:
        raise ValueError(f"implied_vols is missing required columns: {sorted(missing)}")

    mask = (implied_vols["time_to_maturity"] - maturity).abs() <= tolerance
    smile = implied_vols.loc[mask].sort_values("strike")
    if smile.empty:
        raise ValueError(f"No smile found for maturity={maturity}.")

    return smile.reset_index(drop=True)


def term_structure_at_moneyness(
    implied_vols: pd.DataFrame,
    target_moneyness: float = 1.0,
) -> pd.DataFrame:
    """Return nearest available IV by maturity at a target moneyness."""
    required = {"time_to_maturity", "moneyness", "implied_volatility"}
    missing = required.difference(implied_vols.columns)
    if missing:
        raise ValueError(f"implied_vols is missing required columns: {sorted(missing)}")
    if target_moneyness <= 0:
        raise ValueError("target_moneyness must be positive.")

    rows: list[pd.Series] = []
    for _, group in implied_vols.groupby("time_to_maturity"):
        nearest_index = (group["moneyness"] - target_moneyness).abs().idxmin()
        rows.append(group.loc[nearest_index])

    return pd.DataFrame(rows).sort_values("time_to_maturity").reset_index(drop=True)


def skew_slope_by_maturity(implied_vols: pd.DataFrame) -> pd.DataFrame:
    """Estimate smile skew as the slope of IV versus log-moneyness by maturity."""
    required = {"time_to_maturity", "log_moneyness", "implied_volatility"}
    missing = required.difference(implied_vols.columns)
    if missing:
        raise ValueError(f"implied_vols is missing required columns: {sorted(missing)}")

    rows: list[dict[str, float]] = []
    for maturity, group in implied_vols.groupby("time_to_maturity"):
        if len(group) < 2:
            raise ValueError("At least two strikes per maturity are required for skew.")

        x = group["log_moneyness"].to_numpy(dtype=float)
        y = group["implied_volatility"].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, deg=1)
        rows.append(
            {
                "time_to_maturity": float(maturity),
                "skew_slope": float(slope),
                "atm_intercept": float(intercept),
            }
        )

    return pd.DataFrame(rows).sort_values("time_to_maturity").reset_index(drop=True)


def pivot_volatility_surface(
    implied_vols: pd.DataFrame,
    x_col: str = "strike",
    y_col: str = "time_to_maturity",
    value_col: str = "implied_volatility",
) -> pd.DataFrame:
    """Pivot a tidy IV table into a maturity-by-strike surface matrix."""
    required = {x_col, y_col, value_col}
    missing = required.difference(implied_vols.columns)
    if missing:
        raise ValueError(f"implied_vols is missing required columns: {sorted(missing)}")

    return implied_vols.pivot_table(
        index=y_col,
        columns=x_col,
        values=value_col,
        aggfunc="mean",
    ).sort_index(axis=0).sort_index(axis=1)


def interpolate_volatility_surface(
    implied_vols: pd.DataFrame,
    strikes: FloatArray,
    maturities: FloatArray,
    method: str = "linear",
) -> VolatilitySurfaceGrid:
    """Interpolate an implied volatility surface onto a regular grid."""
    required = {"strike", "time_to_maturity", "implied_volatility"}
    missing = required.difference(implied_vols.columns)
    if missing:
        raise ValueError(f"implied_vols is missing required columns: {sorted(missing)}")
    if len(strikes) < 2 or len(maturities) < 2:
        raise ValueError("strikes and maturities must each contain at least two values.")

    strike_grid, maturity_grid = np.meshgrid(strikes, maturities)
    points = implied_vols[["strike", "time_to_maturity"]].to_numpy(dtype=float)
    values = implied_vols["implied_volatility"].to_numpy(dtype=float)
    interpolated = griddata(
        points,
        values,
        (strike_grid, maturity_grid),
        method=method,
    )

    if np.isnan(interpolated).any():
        nearest = griddata(points, values, (strike_grid, maturity_grid), method="nearest")
        interpolated = np.where(np.isnan(interpolated), nearest, interpolated)

    return VolatilitySurfaceGrid(
        strikes=np.asarray(strikes, dtype=np.float64),
        maturities=np.asarray(maturities, dtype=np.float64),
        implied_volatilities=np.asarray(interpolated, dtype=np.float64),
    )


def synthetic_equity_index_volatility(
    strike: float,
    maturity: float,
    spot: float = 100.0,
    base_volatility: float = 0.18,
    downside_skew: float = -0.10,
    smile_curvature: float = 0.18,
    term_slope: float = 0.025,
) -> float:
    """Create a smooth synthetic equity-index implied volatility."""
    if strike <= 0 or maturity <= 0 or spot <= 0:
        raise ValueError("strike, maturity, and spot must be positive.")

    log_moneyness = log(strike / spot)
    term_component = term_slope * (1.0 - np.exp(-maturity))
    skew_component = downside_skew * log_moneyness
    curvature_component = smile_curvature * log_moneyness**2

    return float(base_volatility + term_component + skew_component + curvature_component)
