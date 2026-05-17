# Architecture

This project is organized around the way a derivatives research platform is typically separated on a desk.

## Layers

```text
models/         Pricing and volatility models
analytics/      Risk, implied volatility, surfaces, stress, and reporting
simulations/    Monte Carlo, hedging, and market-making engines
visualization/  Research and reporting charts
utils/          Shared types and validation helpers
tests/          Numerical and behavioral checks
```

## Design Principles

- Model code should be deterministic and side-effect free.
- Analytics code should transform model outputs into risk insight.
- Simulation code may generate paths and pathwise P&L, but should expose reproducible seeds.
- Visualization code should consume tidy tables or typed result objects.
- Tests should protect pricing identities, known numerical values, convergence behavior, and edge cases.

## Core Data Flow

```text
Market inputs
    -> Black-Scholes / SABR models
    -> Greeks, IV, surfaces, stress analytics
    -> Monte Carlo / hedging / market-making simulations
    -> charts, tables, and reports
```

## Numerical Conventions

- Rates are continuously compounded annual rates.
- Volatility is annualized decimal volatility, so 20% is `0.20`.
- Time to maturity is measured in years.
- Vega and Rho are reported per `1.00` change in volatility or rate unless explicitly divided by 100.
- Theta is annualized unless a daily projection is explicitly requested.

## Extension Points

Good next extensions include:

- dividend yield support
- local volatility
- Heston stochastic volatility
- variance reduction for Monte Carlo
- volatility surface arbitrage checks
- portfolio-level margin and liquidity models

