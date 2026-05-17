"""Scenario and Monte Carlo stress testing for option portfolios."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from analytics.greeks import calculate_greeks
from models.black_scholes import BlackScholesInputs, black_scholes_price
from simulations.stochastic_processes import generate_gbm_paths
from utils.types import OptionType


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class StressScenario:
    """Market shock used for deterministic stress testing."""

    name: str
    spot_multiplier: float = 1.0
    volatility_multiplier: float = 1.0
    volatility_shift: float = 0.0
    rate_shift: float = 0.0
    time_shift: float = 0.0

    def validate(self) -> None:
        """Validate stress scenario inputs."""
        if self.spot_multiplier <= 0:
            raise ValueError("spot_multiplier must be positive.")
        if self.volatility_multiplier < 0:
            raise ValueError("volatility_multiplier must be non-negative.")
        if not self.name:
            raise ValueError("scenario name must be non-empty.")


def default_stress_scenarios() -> list[StressScenario]:
    """Return a practical starter set of option stress scenarios."""
    return [
        StressScenario(name="base"),
        StressScenario(name="market_crash", spot_multiplier=0.80, volatility_shift=0.20),
        StressScenario(name="flash_crash", spot_multiplier=0.65, volatility_shift=0.35),
        StressScenario(name="volatility_explosion", volatility_multiplier=2.0),
        StressScenario(name="vol_crush", volatility_multiplier=0.60),
        StressScenario(name="rate_jump", rate_shift=0.02),
        StressScenario(
            name="term_structure_inversion",
            volatility_shift=0.12,
            rate_shift=-0.01,
            time_shift=0.05,
        ),
    ]


def _validate_positions(positions: pd.DataFrame) -> None:
    """Validate required portfolio columns."""
    required = {"strike", "time_to_maturity", "volatility", "option_type", "quantity"}
    missing = required.difference(positions.columns)
    if missing:
        raise ValueError(f"positions is missing required columns: {sorted(missing)}")


def _shocked_inputs(
    row: pd.Series,
    spot: float,
    risk_free_rate: float,
    scenario: StressScenario,
    min_time_to_maturity: float = 1e-6,
    min_volatility: float = 1e-6,
) -> BlackScholesInputs:
    """Build shocked Black-Scholes inputs for one option row."""
    shocked_time = max(
        float(row["time_to_maturity"]) - scenario.time_shift,
        min_time_to_maturity,
    )
    shocked_volatility = max(
        float(row["volatility"]) * scenario.volatility_multiplier
        + scenario.volatility_shift,
        min_volatility,
    )

    return BlackScholesInputs(
        spot=spot * scenario.spot_multiplier,
        strike=float(row["strike"]),
        time_to_maturity=shocked_time,
        risk_free_rate=risk_free_rate + scenario.rate_shift,
        volatility=shocked_volatility,
    )


def revalue_portfolio(
    positions: pd.DataFrame,
    spot: float,
    risk_free_rate: float,
    scenario: StressScenario | None = None,
) -> pd.DataFrame:
    """Revalue each option position under a stress scenario."""
    _validate_positions(positions)
    active_scenario = scenario or StressScenario("base")
    active_scenario.validate()

    rows: list[dict[str, float | str]] = []
    for _, row in positions.iterrows():
        option_type = OptionType(str(row["option_type"]).lower())
        quantity = float(row["quantity"])
        inputs = _shocked_inputs(row, spot, risk_free_rate, active_scenario)
        price = black_scholes_price(inputs, option_type)
        greeks = calculate_greeks(inputs, option_type)
        rows.append(
            {
                "scenario": active_scenario.name,
                "option_type": option_type.value,
                "strike": inputs.strike,
                "time_to_maturity": inputs.time_to_maturity,
                "volatility": inputs.volatility,
                "quantity": quantity,
                "spot": inputs.spot,
                "risk_free_rate": inputs.risk_free_rate,
                "price": price,
                "market_value": quantity * price,
                "delta_exposure": quantity * greeks.delta,
                "gamma_exposure": quantity * greeks.gamma,
                "vega_exposure": quantity * greeks.vega,
                "theta_exposure": quantity * greeks.theta,
                "rho_exposure": quantity * greeks.rho,
            }
        )

    return pd.DataFrame(rows)


def portfolio_stress_test(
    positions: pd.DataFrame,
    spot: float,
    risk_free_rate: float,
    scenarios: list[StressScenario] | None = None,
) -> pd.DataFrame:
    """Run deterministic scenario stress tests on an option portfolio."""
    active_scenarios = scenarios or default_stress_scenarios()
    base = revalue_portfolio(positions, spot, risk_free_rate, StressScenario("base"))
    base_value = float(base["market_value"].sum())

    rows: list[dict[str, float | str]] = []
    for scenario in active_scenarios:
        scenario_values = revalue_portfolio(positions, spot, risk_free_rate, scenario)
        stressed_value = float(scenario_values["market_value"].sum())
        rows.append(
            {
                "scenario": scenario.name,
                "base_value": base_value,
                "stressed_value": stressed_value,
                "pnl": stressed_value - base_value,
                "delta_exposure": float(scenario_values["delta_exposure"].sum()),
                "gamma_exposure": float(scenario_values["gamma_exposure"].sum()),
                "vega_exposure": float(scenario_values["vega_exposure"].sum()),
                "theta_exposure": float(scenario_values["theta_exposure"].sum()),
                "rho_exposure": float(scenario_values["rho_exposure"].sum()),
            }
        )

    return pd.DataFrame(rows)


def monte_carlo_portfolio_stress(
    positions: pd.DataFrame,
    spot: float,
    risk_free_rate: float,
    horizon: float = 10 / 252,
    n_paths: int = 20_000,
    stress_volatility: float = 0.40,
    volatility_multiplier: float = 1.5,
    rate_shift: float = 0.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """Simulate short-horizon portfolio P&L under stressed spot paths."""
    _validate_positions(positions)
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if n_paths < 2:
        raise ValueError("n_paths must be at least 2.")
    if stress_volatility < 0:
        raise ValueError("stress_volatility must be non-negative.")
    if volatility_multiplier < 0:
        raise ValueError("volatility_multiplier must be non-negative.")

    base_value = float(
        revalue_portfolio(positions, spot, risk_free_rate, StressScenario("base"))[
            "market_value"
        ].sum()
    )
    spot_paths = generate_gbm_paths(
        spot=spot,
        drift=risk_free_rate,
        volatility=stress_volatility,
        time_to_maturity=horizon,
        n_steps=1,
        n_paths=n_paths,
        seed=seed,
    )
    terminal_spots = spot_paths[:, -1]
    pnl = np.zeros(n_paths, dtype=np.float64)

    for path_index, terminal_spot in enumerate(terminal_spots):
        path_value = 0.0
        for _, row in positions.iterrows():
            option_type = OptionType(str(row["option_type"]).lower())
            quantity = float(row["quantity"])
            remaining_time = max(float(row["time_to_maturity"]) - horizon, 1e-6)
            shocked_volatility = max(
                float(row["volatility"]) * volatility_multiplier,
                1e-6,
            )
            inputs = BlackScholesInputs(
                spot=float(terminal_spot),
                strike=float(row["strike"]),
                time_to_maturity=remaining_time,
                risk_free_rate=risk_free_rate + rate_shift,
                volatility=shocked_volatility,
            )
            path_value += quantity * black_scholes_price(inputs, option_type)
        pnl[path_index] = path_value - base_value

    return pd.DataFrame(
        {
            "terminal_spot": terminal_spots,
            "pnl": pnl,
            "base_value": base_value,
        }
    )


def stress_loss_summary(pnl: FloatArray | pd.Series) -> dict[str, float]:
    """Summarize a stress P&L distribution."""
    values = np.asarray(pnl, dtype=np.float64)
    if values.size < 2:
        raise ValueError("At least two P&L observations are required.")

    return {
        "mean_pnl": float(np.mean(values)),
        "std_pnl": float(np.std(values, ddof=1)),
        "median_pnl": float(np.median(values)),
        "p01_pnl": float(np.percentile(values, 1.0)),
        "p05_pnl": float(np.percentile(values, 5.0)),
        "p95_pnl": float(np.percentile(values, 95.0)),
        "expected_shortfall_5": float(np.mean(values[values <= np.percentile(values, 5.0)])),
        "worst_pnl": float(np.min(values)),
        "best_pnl": float(np.max(values)),
    }

