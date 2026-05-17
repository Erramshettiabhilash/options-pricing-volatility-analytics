# Step 11: Stress Testing

Stress testing asks what happens when model assumptions are no longer calm.

Instead of asking:

```text
What is the fair value today?
```

we ask:

```text
What happens if spot crashes, volatility explodes, rates jump, or hedges break?
```

## 1. Why Stress Testing Matters

Greeks are local sensitivities. They work best for small moves.

Stress tests are scenario-based. They answer large-move questions:

- What if spot falls 20%?
- What if implied volatility doubles?
- What if rates jump?
- What if the term structure inverts?
- What if realized volatility is far above hedge volatility?

## 2. Scenario Analysis

A stress scenario shocks market inputs:

```text
spot -> spot * multiplier
volatility -> volatility * multiplier + shift
rate -> rate + shift
time_to_maturity -> time_to_maturity - time_shift
```

Then the portfolio is repriced under the shocked market.

```text
Stress P&L = stressed portfolio value - base portfolio value
```

## 3. Volatility Explosion

Short-option books are usually short vega. A volatility explosion can create large losses because implied vols rise across strikes and maturities.

## 4. Market Crash

Equity crashes usually combine:

```text
spot down
volatility up
liquidity worse
hedging harder
```

That combination is more dangerous than a spot shock alone.

## 5. Flash Crash

A flash crash represents a sudden large spot move where hedges cannot be rebalanced smoothly.

This exposes gamma risk and hedge slippage.

## 6. Term Structure Inversion

In stressed markets, short-dated volatility can rise above long-dated volatility.

This matters because short-dated option books can become extremely sensitive to near-term realized movement.

## 7. Monte Carlo Stress

Scenario tests are deterministic. Monte Carlo stress simulates many stressed spot outcomes and reprices the portfolio under each path.

Useful tail metrics:

```text
5% P&L
1% P&L
expected shortfall
worst simulated loss
```

Expected shortfall is the average loss inside the tail, which is often more informative than a single percentile.

## 8. Code

Stress analytics live in:

```text
analytics/stress_testing.py
```

Visualization:

```text
visualization/stress_testing.py
```

Main functions:

```text
StressScenario
default_stress_scenarios
revalue_portfolio
portfolio_stress_test
monte_carlo_portfolio_stress
stress_loss_summary
```

## 9. Professional Interpretation

A desk does not trust only one number. It wants:

- scenario P&L
- tail loss distribution
- stressed Greeks
- hedge breakdown assumptions
- liquidity and transaction-cost overlays

Stress testing is how an option platform moves from pricing into risk governance.

