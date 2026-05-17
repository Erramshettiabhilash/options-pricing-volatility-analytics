"""Tests for option portfolio stress testing."""

import pandas as pd
import pytest

from analytics.stress_testing import (
    StressScenario,
    default_stress_scenarios,
    monte_carlo_portfolio_stress,
    portfolio_stress_test,
    revalue_portfolio,
    stress_loss_summary,
)


def _sample_positions() -> pd.DataFrame:
    return pd.DataFrame(
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
                "quantity": 5.0,
            },
        ]
    )


def test_revalue_portfolio_returns_option_level_stress_rows() -> None:
    """Repricing should return one stressed row per option position."""
    positions = _sample_positions()
    scenario = StressScenario("crash", spot_multiplier=0.8, volatility_shift=0.2)
    table = revalue_portfolio(positions, spot=100.0, risk_free_rate=0.03, scenario=scenario)

    assert len(table) == len(positions)
    assert table["scenario"].unique().tolist() == ["crash"]
    assert {"market_value", "delta_exposure", "vega_exposure"}.issubset(table.columns)
    assert (table["volatility"] > positions["volatility"].to_numpy()).all()


def test_portfolio_stress_test_reports_pnl_by_scenario() -> None:
    """Scenario stress testing should aggregate portfolio P&L and Greeks."""
    positions = _sample_positions()
    scenarios = [
        StressScenario("base"),
        StressScenario("vol_up", volatility_multiplier=1.5),
    ]
    stress = portfolio_stress_test(positions, 100.0, 0.03, scenarios)

    assert stress["scenario"].tolist() == ["base", "vol_up"]
    assert stress.loc[stress["scenario"] == "base", "pnl"].iloc[0] == pytest.approx(0.0)
    assert {"gamma_exposure", "theta_exposure", "rho_exposure"}.issubset(stress.columns)


def test_default_stress_scenarios_include_crash_and_vol_explosion() -> None:
    """The default scenario set should include common desk stresses."""
    names = {scenario.name for scenario in default_stress_scenarios()}

    assert "market_crash" in names
    assert "volatility_explosion" in names
    assert "flash_crash" in names


def test_monte_carlo_portfolio_stress_returns_pathwise_pnl() -> None:
    """Monte Carlo stress should return one P&L row per path."""
    paths = monte_carlo_portfolio_stress(
        _sample_positions(),
        spot=100.0,
        risk_free_rate=0.03,
        horizon=5 / 252,
        n_paths=500,
        stress_volatility=0.35,
        seed=4,
    )

    assert len(paths) == 500
    assert {"terminal_spot", "pnl", "base_value"}.issubset(paths.columns)
    assert paths["pnl"].std() > 0.0


def test_stress_loss_summary_reports_tail_metrics() -> None:
    """Loss summary should include tail and expected-shortfall metrics."""
    paths = monte_carlo_portfolio_stress(
        _sample_positions(),
        spot=100.0,
        risk_free_rate=0.03,
        n_paths=300,
        seed=7,
    )
    summary = stress_loss_summary(paths["pnl"])

    assert set(summary) == {
        "mean_pnl",
        "std_pnl",
        "median_pnl",
        "p01_pnl",
        "p05_pnl",
        "p95_pnl",
        "expected_shortfall_5",
        "worst_pnl",
        "best_pnl",
    }
    assert summary["expected_shortfall_5"] <= summary["p05_pnl"]


def test_invalid_stress_scenario_raises_value_error() -> None:
    """Invalid scenario shocks should fail clearly."""
    with pytest.raises(ValueError, match="spot_multiplier must be positive"):
        StressScenario("bad", spot_multiplier=0.0).validate()

