"""Run a compact end-to-end demo of the options analytics platform.

The script creates reproducible example artifacts under ``results/final_demo``.
It is intentionally small enough to run during an interview or project review.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analytics.implied_volatility import extract_implied_volatility_table
from analytics.reporting import build_step12_analytics_summary
from analytics.stress_testing import (
    default_stress_scenarios,
    monte_carlo_portfolio_stress,
    portfolio_stress_test,
    stress_loss_summary,
)
from analytics.volatility_surface import add_moneyness_columns, smile_for_maturity
from models.black_scholes import BlackScholesInputs, black_scholes_price
from models.sabr import calibrate_sabr_from_dataframe
from simulations.hedging import simulate_delta_hedge
from simulations.market_making import (
    MarketMakingConfig,
    aggregate_risk_report,
    market_making_risk_report,
    simulate_market_making_session,
    theta_decay_analysis,
)
from simulations.monte_carlo import monte_carlo_price
from utils.types import OptionType
from visualization.hedging import plot_hedge_pnl_distribution
from visualization.market_making import (
    plot_cumulative_spread_capture,
    plot_inventory_by_instrument,
    plot_theta_decay,
)
from visualization.payoffs import plot_option_payoff, plot_strategy_payoff
from visualization.stress_testing import plot_scenario_pnl
from visualization.volatility import plot_sabr_calibration, plot_volatility_smile


DATA_PATH = PROJECT_ROOT / "data" / "sample_option_chain.csv"
OUTPUT_DIR = PROJECT_ROOT / "results" / "final_demo"


def main() -> None:
    """Generate final demo tables and charts."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    option_chain = pd.read_csv(DATA_PATH)
    iv_table = extract_implied_volatility_table(option_chain, method="bisection")
    iv_table = add_moneyness_columns(iv_table)
    iv_table.to_csv(OUTPUT_DIR / "implied_volatility_table.csv", index=False)

    one_year_smile = smile_for_maturity(iv_table, maturity=1.0)
    sabr_result = calibrate_sabr_from_dataframe(
        smile=one_year_smile,
        forward=100.0 * 2.718281828459045**0.03,
        maturity=1.0,
        beta=0.5,
    )
    sabr_result.calibration_table.to_csv(
        OUTPUT_DIR / "sabr_calibration_table.csv",
        index=False,
    )

    inputs = BlackScholesInputs(
        spot=100.0,
        strike=100.0,
        time_to_maturity=1.0,
        risk_free_rate=0.05,
        volatility=0.20,
    )
    mc_result = monte_carlo_price(
        inputs,
        OptionType.CALL,
        n_paths=50_000,
        n_steps=252,
        seed=14,
    )
    hedge_result = simulate_delta_hedge(
        inputs,
        OptionType.CALL,
        n_paths=2_000,
        n_steps=126,
        option_position=-1.0,
        seed=14,
    )

    instruments = pd.DataFrame(
        [
            {
                "strike": 95.0,
                "time_to_maturity": 0.50,
                "volatility": 0.22,
                "option_type": "call",
            },
            {
                "strike": 100.0,
                "time_to_maturity": 1.00,
                "volatility": 0.20,
                "option_type": "call",
            },
            {
                "strike": 105.0,
                "time_to_maturity": 1.00,
                "volatility": 0.23,
                "option_type": "put",
            },
        ]
    )
    trades, inventory = simulate_market_making_session(
        instruments,
        spot=100.0,
        risk_free_rate=0.03,
        n_trades=75,
        config=MarketMakingConfig(max_inventory=40.0),
        seed=14,
    )
    risk_report = market_making_risk_report(inventory, 100.0, 0.03)
    theta_projection = theta_decay_analysis(risk_report, days=10)

    positions = inventory.copy()
    stress_table = portfolio_stress_test(
        positions,
        spot=100.0,
        risk_free_rate=0.03,
        scenarios=default_stress_scenarios(),
    )
    stress_paths = monte_carlo_portfolio_stress(
        positions,
        spot=100.0,
        risk_free_rate=0.03,
        n_paths=5_000,
        stress_volatility=0.40,
        volatility_multiplier=1.5,
        seed=14,
    )
    stress_summary = stress_loss_summary(stress_paths["pnl"])

    summary = build_step12_analytics_summary(
        monte_carlo_summary={
            "black_scholes_price": black_scholes_price(inputs, OptionType.CALL),
            "monte_carlo_price": mc_result.price,
            "monte_carlo_standard_error": mc_result.standard_error,
        },
        hedging_summary={
            "hedge_mean_pnl": hedge_result.summary["mean_pnl"],
            "hedge_std_pnl": hedge_result.summary["std_pnl"],
        },
        stress_summary={
            "stress_p05_pnl": stress_summary["p05_pnl"],
            "expected_shortfall_5": stress_summary["expected_shortfall_5"],
        },
    )
    summary.to_csv(OUTPUT_DIR / "headline_summary.csv", index=False)
    trades.to_csv(OUTPUT_DIR / "market_making_trades.csv", index=False)
    inventory.to_csv(OUTPUT_DIR / "market_making_inventory.csv", index=False)
    pd.DataFrame([aggregate_risk_report(risk_report)]).to_csv(
        OUTPUT_DIR / "aggregate_risk_report.csv",
        index=False,
    )
    stress_table.to_csv(OUTPUT_DIR / "scenario_stress_results.csv", index=False)

    fig, _ = plot_option_payoff(inputs, OptionType.CALL, title="Final Demo Long Call")
    fig.savefig(OUTPUT_DIR / "long_call_payoff.png", dpi=140)

    call_spread_short = BlackScholesInputs(100.0, 110.0, 1.0, 0.05, 0.20)
    fig, _ = plot_strategy_payoff(
        [
            (inputs, OptionType.CALL, 1.0),
            (call_spread_short, OptionType.CALL, -1.0),
        ],
        title="Final Demo Bull Call Spread",
    )
    fig.savefig(OUTPUT_DIR / "bull_call_spread.png", dpi=140)

    fig, _ = plot_volatility_smile(iv_table, "Final Demo Implied Volatility Smiles")
    fig.savefig(OUTPUT_DIR / "volatility_smiles.png", dpi=140)

    fig, _ = plot_sabr_calibration(sabr_result, "Final Demo SABR Calibration")
    fig.savefig(OUTPUT_DIR / "sabr_calibration.png", dpi=140)

    fig, _ = plot_hedge_pnl_distribution(
        hedge_result,
        "Final Demo Delta-Hedge P&L",
    )
    fig.savefig(OUTPUT_DIR / "delta_hedge_pnl.png", dpi=140)

    fig, _ = plot_cumulative_spread_capture(
        trades,
        "Final Demo Cumulative Spread Capture",
    )
    fig.savefig(OUTPUT_DIR / "spread_capture.png", dpi=140)

    fig, _ = plot_inventory_by_instrument(
        inventory,
        "Final Demo Market-Maker Inventory",
    )
    fig.savefig(OUTPUT_DIR / "inventory.png", dpi=140)

    fig, _ = plot_theta_decay(theta_projection, "Final Demo Theta Projection")
    fig.savefig(OUTPUT_DIR / "theta_projection.png", dpi=140)

    fig, _ = plot_scenario_pnl(stress_table, "Final Demo Scenario Stress P&L")
    fig.savefig(OUTPUT_DIR / "scenario_stress_pnl.png", dpi=140)

    print(f"Wrote final demo artifacts to {OUTPUT_DIR}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
