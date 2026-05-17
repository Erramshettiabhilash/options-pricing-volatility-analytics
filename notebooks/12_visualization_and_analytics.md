# Step 12: Visualization & Analytics

This step turns the platform into something useful for research, reporting, and interviews.

Pricing code gives numbers. Visualization gives intuition.

## 1. What We Visualize

The platform now supports charts for:

- option payoff diagrams
- strategy payoff diagrams
- Greeks curves
- Monte Carlo paths
- payoff distributions
- Monte Carlo convergence
- volatility smiles
- volatility term structures
- volatility surfaces
- SABR calibration fits
- hedging P&L
- market-making inventory
- stress-test P&L

## 2. Why Payoff Diagrams Matter

Payoff diagrams show the contract's economic shape at expiry.

For a call:

```text
payoff = max(S_T - K, 0)
```

For a put:

```text
payoff = max(K - S_T, 0)
```

Profit includes premium:

```text
profit = payoff - premium paid
```

For short positions, the sign flips.

## 3. Greeks Curves

Greeks curves show how risk changes as spot moves.

This is more useful than a single Greek number because option risk is nonlinear.

## 4. Monte Carlo Charts

Monte Carlo visualizations answer:

- what paths were simulated?
- how noisy is the payoff?
- does the estimator converge to theory?

## 5. Volatility Charts

Volatility smiles and surfaces show how market-implied risk differs by strike and maturity.

These are essential for:

- calibration
- quote marking
- risk checks
- model comparison

## 6. Hedging and Market-Making Charts

Hedging charts show whether a strategy behaves under rebalancing.

Market-making charts show:

- inventory
- spread capture
- theta decay
- risk concentration

## 7. Stress Charts

Stress charts are for risk governance. They highlight where losses appear under large shocks.

Useful views:

- scenario P&L
- stressed Greek exposures
- Monte Carlo stress distribution
- expected shortfall

## 8. Code

New payoff visualizations live in:

```text
visualization/payoffs.py
```

Analytics summaries live in:

```text
analytics/reporting.py
```

The rest of the visualization stack lives across:

```text
visualization/foundations.py
visualization/greeks.py
visualization/monte_carlo.py
visualization/volatility.py
visualization/hedging.py
visualization/market_making.py
visualization/stress_testing.py
```

## 9. Professional Charting Rules

Good quant charts should:

- have clear titles
- label axes
- show units
- avoid visual clutter
- make benchmarks visible
- save reproducibly into `results/`

For interviews, charts should support the story:

```text
model -> risk -> simulation -> hedging -> stress
```

