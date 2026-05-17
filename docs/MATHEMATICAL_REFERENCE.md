# Mathematical Reference

## Black-Scholes

For a European option:

```text
d1 = [log(S / K) + (r + 0.5 sigma^2)T] / [sigma sqrt(T)]
d2 = d1 - sigma sqrt(T)
```

Call:

```text
C = S N(d1) - K exp(-rT) N(d2)
```

Put:

```text
P = K exp(-rT) N(-d2) - S N(-d1)
```

## Greeks

```text
Call Delta = N(d1)
Put Delta  = N(d1) - 1
Gamma      = phi(d1) / [S sigma sqrt(T)]
Vega       = S phi(d1) sqrt(T)
```

Theta and Rho are implemented analytically in `analytics/greeks.py`.

## Risk-Neutral Monte Carlo

Risk-neutral GBM:

```text
dS_t = r S_t dt + sigma S_t dW_t
```

Monte Carlo price:

```text
V_0 = exp(-rT) average(payoff(S_T))
```

Standard error:

```text
SE = standard_deviation(discounted payoffs) / sqrt(number of paths)
```

## Implied Volatility

Implied volatility solves:

```text
BlackScholesPrice(sigma) - MarketPrice = 0
```

Newton update:

```text
sigma_next = sigma - pricing_error / Vega
```

## SABR

SABR dynamics:

```text
dF_t      = alpha_t F_t^beta dW_t
dalpha_t = nu alpha_t dZ_t
corr(dW_t, dZ_t) = rho
```

Parameter interpretation:

```text
alpha = volatility level
beta  = elasticity
rho   = skew
nu    = volatility of volatility
```

## Stress Testing

Scenario P&L:

```text
stress P&L = stressed portfolio value - base portfolio value
```

Expected shortfall:

```text
ES_5 = average P&L conditional on being in the worst 5% of outcomes
```

