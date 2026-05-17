# Step 3: Black-Scholes Model

Black-Scholes is the classic closed-form pricing model for European options. It matters because it gives us a clean benchmark for pricing, Greeks, implied volatility, and Monte Carlo validation.

## 1. What We Are Pricing

A European call gives the right, but not the obligation, to buy the underlying at strike `K` at expiry.

```text
Call payoff = max(S_T - K, 0)
```

A European put gives the right, but not the obligation, to sell the underlying at strike `K` at expiry.

```text
Put payoff = max(K - S_T, 0)
```

The option value today is not just the expected payoff. It is the discounted risk-neutral expected payoff:

```text
Option Price = exp(-rT) * E_Q[Payoff(S_T)]
```

## 2. Model Parameters

Black-Scholes uses five core inputs:

```text
S      current spot price
K      strike price
T      time to maturity in years
r      continuously compounded risk-free rate
sigma  annualized volatility
```

Finance meaning:

- `S`: where the underlying is trading now.
- `K`: the exercise level written into the contract.
- `T`: how much time uncertainty has to unfold.
- `r`: the cash discounting and forward-pricing rate.
- `sigma`: the scale of future uncertainty.

Volatility is usually the most important pricing input because options benefit from dispersion.

## 3. The Formula

For a European call:

```text
C = S * N(d1) - K * exp(-rT) * N(d2)
```

For a European put:

```text
P = K * exp(-rT) * N(-d2) - S * N(-d1)
```

where:

```text
d1 = [log(S / K) + (r + 0.5 * sigma^2)T] / [sigma * sqrt(T)]
d2 = d1 - sigma * sqrt(T)
```

`N(x)` is the standard normal cumulative distribution function.

## 4. Why N(d1) and N(d2) Matter

The terms `N(d1)` and `N(d2)` are not arbitrary magic.

`N(d2)` is closely related to the risk-neutral probability that the option finishes in the money.

For a call:

```text
N(d2) ≈ risk-neutral probability that S_T > K
```

`N(d1)` appears because the expected stock value conditional on exercise needs a volatility-adjusted weighting. For a call, `N(d1)` is also the Black-Scholes delta, which we will use in Step 4.

Trading intuition:

- `N(d2)` behaves like exercise probability.
- `N(d1)` behaves like hedge ratio.

That is why the call price looks like:

```text
expected stock received - present value of expected strike paid
```

## 5. Put-Call Parity

For European options without dividends:

```text
C - P = S - K * exp(-rT)
```

This is a no-arbitrage relationship. If our call and put prices violate it, something is wrong in either the model implementation or the inputs.

## 6. Implementation

The code lives in:

```text
models/black_scholes.py
```

Example:

```python
from models.black_scholes import BlackScholesInputs, black_scholes_price
from utils.types import OptionType

inputs = BlackScholesInputs(
    spot=100.0,
    strike=100.0,
    time_to_maturity=1.0,
    risk_free_rate=0.05,
    volatility=0.20,
)

call_price = black_scholes_price(inputs, OptionType.CALL)
put_price = black_scholes_price(inputs, OptionType.PUT)

print(call_price)  # 10.4506
print(put_price)   # 5.5735
```

## 7. Validation Checks

Good quant code should handle edge cases deliberately:

- `spot > 0`
- `strike > 0`
- `time_to_maturity >= 0`
- `volatility >= 0`
- at expiry, price equals intrinsic value
- with zero volatility, price becomes discounted deterministic payoff
- call and put prices satisfy put-call parity

These checks are covered in:

```text
tests/test_black_scholes.py
```

