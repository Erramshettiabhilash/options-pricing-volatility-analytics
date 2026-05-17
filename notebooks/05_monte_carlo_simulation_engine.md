# Step 5: Monte Carlo Simulation Engine

Monte Carlo pricing estimates option value by simulating many possible future paths, calculating the payoff on each path, and discounting the average payoff back to today.

## 1. Why Monte Carlo Matters

Black-Scholes gives a beautiful closed-form solution for simple European options. Real desks often need more:

- path-dependent payoffs
- exotic options
- stress scenarios
- stochastic volatility
- portfolio-level simulation

Monte Carlo is flexible because it prices from simulated scenarios rather than from a closed-form formula.

## 2. Risk-Neutral Simulation

Under Black-Scholes assumptions, the real-world stock model is:

```text
dS_t = mu S_t dt + sigma S_t dW_t
```

For pricing, we simulate under the risk-neutral measure:

```text
dS_t = r S_t dt + sigma S_t dW_t
```

The exact GBM solution is:

```text
S_T = S_0 * exp((r - 0.5 sigma^2)T + sigma W_T)
```

This is the same stochastic model from Step 2, but with drift `r` instead of `mu`.

## 3. Monte Carlo Pricing Formula

For each simulated terminal price:

```text
Call payoff = max(S_T - K, 0)
Put payoff  = max(K - S_T, 0)
```

Then:

```text
Monte Carlo Price = exp(-rT) * average(payoff)
```

The estimator is random because every simulation sample is random.

## 4. Simulation Error

Monte Carlo does not produce one exact answer. It produces an estimate with error.

The standard error is:

```text
SE = standard_deviation(discounted payoffs) / sqrt(number of paths)
```

Approximate 95% confidence interval:

```text
price +/- 1.96 * SE
```

Why this matters:

If the Black-Scholes price is inside the Monte Carlo confidence interval, the simulation is behaving sensibly.

## 5. Convergence

Monte Carlo error shrinks slowly:

```text
error scale ≈ 1 / sqrt(N)
```

That means getting 10 times less error usually requires about 100 times more paths.

This is why variance reduction matters.

## 6. Variance Reduction Intuition

Variance reduction means getting a more accurate estimate with the same number of paths.

Common methods:

- antithetic variates
- control variates
- stratified sampling
- quasi-random Sobol sequences

Example intuition:

If one simulation uses a positive random shock, antithetic sampling also uses the matching negative shock. This balances the sample and often reduces noise.

We will keep Step 5 simple and use plain vectorized Monte Carlo. Later, control variates can use Black-Scholes itself as a benchmark.

## 7. Code

The engine lives in:

```text
simulations/monte_carlo.py
```

Example:

```python
from models.black_scholes import BlackScholesInputs, black_scholes_price
from simulations.monte_carlo import monte_carlo_price
from utils.types import OptionType

inputs = BlackScholesInputs(
    spot=100.0,
    strike=100.0,
    time_to_maturity=1.0,
    risk_free_rate=0.05,
    volatility=0.20,
)

mc = monte_carlo_price(
    inputs=inputs,
    option_type=OptionType.CALL,
    n_paths=100_000,
    n_steps=252,
    seed=42,
)

bs = black_scholes_price(inputs, OptionType.CALL)

print(mc.price)
print(mc.standard_error)
print(bs)
```

## 8. Professional Interpretation

A Monte Carlo result should always report:

- price estimate
- standard error
- confidence interval
- model assumptions
- number of paths
- random seed for reproducibility

Without those details, a simulation result is hard to audit.

