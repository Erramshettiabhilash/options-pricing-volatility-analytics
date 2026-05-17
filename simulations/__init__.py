"""Simulation engines for Monte Carlo pricing and dynamic hedging."""

from simulations.monte_carlo import (
    MonteCarloResult,
    european_payoff,
    monte_carlo_convergence,
    monte_carlo_price,
)
from simulations.hedging import (
    DeltaHedgingResult,
    gamma_vega_exposure_table,
    hedge_neutralizing_trades,
    simulate_delta_hedge,
)
from simulations.market_making import (
    MarketMakingConfig,
    OptionQuote,
    TradeExecution,
    aggregate_risk_report,
    execute_quote_trade,
    market_making_risk_report,
    quote_option,
    simulate_market_making_session,
    theta_decay_analysis,
)
from simulations.stochastic_processes import (
    calculate_log_returns,
    generate_brownian_motion,
    generate_gbm_paths,
    time_grid,
)

__all__ = [
    "MonteCarloResult",
    "DeltaHedgingResult",
    "MarketMakingConfig",
    "OptionQuote",
    "TradeExecution",
    "aggregate_risk_report",
    "calculate_log_returns",
    "european_payoff",
    "execute_quote_trade",
    "gamma_vega_exposure_table",
    "generate_brownian_motion",
    "generate_gbm_paths",
    "hedge_neutralizing_trades",
    "market_making_risk_report",
    "monte_carlo_convergence",
    "monte_carlo_price",
    "quote_option",
    "simulate_delta_hedge",
    "simulate_market_making_session",
    "theta_decay_analysis",
    "time_grid",
]
