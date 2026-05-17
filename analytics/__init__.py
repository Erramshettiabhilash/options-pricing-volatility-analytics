"""Risk analytics, implied volatility, sensitivities, and stress testing."""

from analytics.greeks import (
    Greeks,
    calculate_greeks,
    delta,
    gamma,
    rho,
    theta,
    vega,
)
from analytics.implied_volatility import (
    ImpliedVolatilityResult,
    extract_implied_volatility_table,
    implied_volatility,
    implied_volatility_bisection,
    implied_volatility_newton,
    no_arbitrage_price_bounds,
    validate_market_price,
)
from analytics.volatility_surface import (
    VolatilitySurfaceGrid,
    add_moneyness_columns,
    build_implied_volatility_surface,
    interpolate_volatility_surface,
    pivot_volatility_surface,
    skew_slope_by_maturity,
    smile_for_maturity,
    synthetic_equity_index_volatility,
    term_structure_at_moneyness,
)
from analytics.stress_testing import (
    StressScenario,
    default_stress_scenarios,
    monte_carlo_portfolio_stress,
    portfolio_stress_test,
    revalue_portfolio,
    stress_loss_summary,
)
from analytics.reporting import (
    build_step12_analytics_summary,
    summarize_numeric_columns,
)

__all__ = [
    "Greeks",
    "ImpliedVolatilityResult",
    "VolatilitySurfaceGrid",
    "StressScenario",
    "add_moneyness_columns",
    "build_implied_volatility_surface",
    "build_step12_analytics_summary",
    "calculate_greeks",
    "default_stress_scenarios",
    "delta",
    "extract_implied_volatility_table",
    "gamma",
    "implied_volatility",
    "implied_volatility_bisection",
    "implied_volatility_newton",
    "interpolate_volatility_surface",
    "monte_carlo_portfolio_stress",
    "no_arbitrage_price_bounds",
    "pivot_volatility_surface",
    "portfolio_stress_test",
    "revalue_portfolio",
    "rho",
    "skew_slope_by_maturity",
    "smile_for_maturity",
    "synthetic_equity_index_volatility",
    "summarize_numeric_columns",
    "term_structure_at_moneyness",
    "theta",
    "stress_loss_summary",
    "validate_market_price",
    "vega",
]
