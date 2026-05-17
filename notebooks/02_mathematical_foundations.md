# Step 2: Mathematical Foundations

This step builds the intuition behind the models we will code later. Options are contracts on uncertain future prices, so before pricing options we need a disciplined language for uncertainty through time.

## 1. Stochastic Processes

A stochastic process is a variable that evolves through time with randomness.

In finance, the stock price is not just one random number. It is a full path:

```text
S_0 -> S_1 -> S_2 -> ... -> S_T
```

Why it matters:

- An option payoff depends on the future value of the underlying.
- A hedge depends on how risk changes through time.
- A volatility model is a model of how uncertainty evolves.

Mathematically, a stochastic process is often written:

```text
X_t, for t >= 0
```

The subscript `t` means the value changes with time.

## 2. Brownian Motion

Brownian Motion is the core random building block used in Black-Scholes and many derivatives models.

A standard Brownian Motion `W_t` has three important properties:

1. It starts at zero: `W_0 = 0`
2. Its increments are independent.
3. Its increments are normally distributed:

```text
W_t - W_s ~ Normal(0, t - s), where t > s
```

Intuition:

Brownian Motion is a mathematically clean model for cumulative random shocks. Each short time step adds a small unpredictable move.

Simple simulation idea:

```text
dW = sqrt(dt) * Z
Z ~ Normal(0, 1)
W_t = cumulative sum of dW
```

Finance meaning:

Brownian Motion is not the stock price itself. It is the random shock driving the stock price.

## 3. Drift vs Diffusion

A common continuous-time model looks like:

```text
dS_t = drift term + diffusion term
```

More specifically:

```text
dS_t = mu * S_t * dt + sigma * S_t * dW_t
```

The two pieces mean different things:

- `mu * S_t * dt`: expected directional growth over time.
- `sigma * S_t * dW_t`: random uncertainty around that growth.

Trading intuition:

- Drift is the long-run trend assumption.
- Diffusion is the risk that creates option value.

For short-dated options, volatility usually matters far more than real-world drift.

## 4. Geometric Brownian Motion

Geometric Brownian Motion, or GBM, is the classic stock-price model behind Black-Scholes.

The model is:

```text
dS_t = mu * S_t * dt + sigma * S_t * dW_t
```

Its exact solution is:

```text
S_t = S_0 * exp((mu - 0.5 * sigma^2) * t + sigma * W_t)
```

Why use GBM?

- Prices stay positive.
- Log returns are normally distributed.
- It creates a closed-form solution for European options under Black-Scholes assumptions.

Finance meaning:

GBM says percentage returns are random, not dollar changes. A stock at 100 can move more dollars than a stock at 10 under the same volatility.

## 5. Log Returns

Simple return:

```text
R_t = (S_t - S_{t-1}) / S_{t-1}
```

Log return:

```text
r_t = log(S_t / S_{t-1})
```

Why quants like log returns:

- They add cleanly through time:

```text
log(S_T / S_0) = log(S_1 / S_0) + ... + log(S_T / S_{T-1})
```

- They fit naturally with GBM.
- They make continuous compounding mathematically clean.

Practical note:

For small daily returns, simple returns and log returns are close. For large moves, they differ.

## 6. Volatility

Volatility measures the scale of uncertainty in returns.

In this project, volatility is annualized:

```text
sigma = annualized standard deviation of returns
```

If daily log-return volatility is `sigma_daily`, then approximate annualized volatility is:

```text
sigma_annual = sigma_daily * sqrt(252)
```

Why `sqrt(252)`?

Variance scales linearly with time, and volatility is the square root of variance.

Trading intuition:

Volatility is the price of uncertainty. Higher volatility increases the value of both calls and puts because optionality benefits from larger possible moves.

## 7. Risk-Neutral Pricing

This is one of the most important ideas in derivatives.

Under risk-neutral pricing, we price options as discounted expected payoffs under a special probability measure:

```text
Option Price = exp(-rT) * E_Q[Payoff(S_T)]
```

`Q` is the risk-neutral measure.

The key modeling switch:

```text
Real-world drift:       mu
Risk-neutral drift:     r
```

Why this matters:

In Black-Scholes pricing, the option price does not depend on the investor's expected stock return `mu`. It depends on:

- current spot
- strike
- time to maturity
- risk-free rate
- volatility

Intuition:

If an option could be perfectly replicated by dynamically trading stock and cash, then its price must equal the cost of that replicating strategy. Otherwise, arbitrage would exist.

That replication argument removes the need to forecast the stock's real-world expected return.

## 8. Visual Mental Model

Think of the modeling stack like this:

```text
Random shocks
    Brownian Motion: W_t

Stock price dynamics
    GBM: dS_t = mu S_t dt + sigma S_t dW_t

Return measurement
    Log returns: log(S_t / S_{t-1})

Option valuation
    Risk-neutral expectation: exp(-rT) * E_Q[payoff]

Trading risk
    Greeks measure how option value changes when inputs move
```

## 9. Code in This Step

The reusable code for this step lives in:

```text
simulations/stochastic_processes.py
visualization/foundations.py
```

Example:

```python
from simulations.stochastic_processes import (
    calculate_log_returns,
    generate_gbm_paths,
    time_grid,
)

times = time_grid(time_to_maturity=1.0, n_steps=252)
paths = generate_gbm_paths(
    spot=100.0,
    drift=0.08,
    volatility=0.20,
    time_to_maturity=1.0,
    n_steps=252,
    n_paths=1_000,
    seed=42,
)
log_returns = calculate_log_returns(paths)
```

This is the foundation for Monte Carlo pricing later. In risk-neutral Monte Carlo, we will replace `drift=mu` with `drift=r`.

