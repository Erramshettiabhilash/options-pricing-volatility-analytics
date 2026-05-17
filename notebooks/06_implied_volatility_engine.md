# Step 6: Implied Volatility Engine

Implied volatility is the volatility input that makes a model price match the market price.

Black-Scholes answers:

```text
Given sigma, what is the option price?
```

Implied volatility answers:

```text
Given the market price, what sigma is the market implying?
```

## 1. Why Implied Volatility Matters

Volatility is not directly observable like spot or strike. We infer it from option prices.

Traders quote options in implied volatility because:

- it normalizes option prices across strikes and maturities
- it reveals market expectations of future uncertainty
- it makes rich/cheap comparisons easier
- it exposes smile, skew, and term-structure behavior

In practice, market makers often think in volatility first and price second.

## 2. The Root-Finding Problem

We want to solve:

```text
BlackScholesPrice(sigma) - MarketPrice = 0
```

There is no closed-form inverse for Black-Scholes volatility, so we use numerical methods.

## 3. Newton-Raphson Method

Newton-Raphson uses the slope of the pricing function with respect to volatility. That slope is Vega.

Update rule:

```text
sigma_next = sigma - (ModelPrice - MarketPrice) / Vega
```

Why it is useful:

- very fast near the solution
- usually converges in a few iterations for liquid at-the-money options

Why it can fail:

- vega can be tiny for deep ITM or OTM options
- a poor starting guess can jump outside realistic volatility bounds
- bad market prices can violate arbitrage bounds

## 4. Bisection Method

Bisection brackets the solution between a low volatility and high volatility.

Algorithm:

```text
1. Choose low_vol and high_vol
2. Price at the midpoint
3. Keep the half interval that contains the root
4. Repeat until the pricing error is small
```

Why it is useful:

- slower than Newton
- much more robust
- excellent fallback method

## 5. No-Arbitrage Bounds

Before solving IV, we check whether the market price is sensible.

For calls:

```text
max(S - K exp(-rT), 0) <= C <= S
```

For puts:

```text
max(K exp(-rT) - S, 0) <= P <= K exp(-rT)
```

If a price violates these bounds, implied volatility is not meaningful under this model.

## 6. Volatility Smile

If Black-Scholes assumptions held perfectly, every strike and maturity would imply the same volatility.

Real markets do not behave that way.

When we compute IV across strikes, we often see:

```text
Strike low  -> higher IV
ATM strike  -> lower IV
Strike high -> different IV
```

This pattern is called the volatility smile or skew.

Market interpretation:

- equity index OTM puts often have high IV because crash protection is in demand
- skew reflects asymmetric downside risk
- smile shape reveals supply, demand, jumps, leverage effects, and tail risk

## 7. Code

The implied volatility engine lives in:

```text
analytics/implied_volatility.py
```

Example:

```python
from analytics.implied_volatility import implied_volatility
from models.black_scholes import BlackScholesInputs
from utils.types import OptionType

inputs = BlackScholesInputs(
    spot=100.0,
    strike=100.0,
    time_to_maturity=1.0,
    risk_free_rate=0.05,
    volatility=0.20,
)

result = implied_volatility(
    inputs=inputs,
    option_type=OptionType.CALL,
    market_price=10.4505835722,
)

print(result.implied_volatility)  # approximately 0.20
```

## 8. Professional Workflow

A good IV extraction pipeline should:

- validate prices against no-arbitrage bounds
- use Newton for speed
- fall back to Bisection for robustness
- report convergence status and pricing error
- store results in a tidy table for smile and surface analysis

This is the bridge into Step 7, where we build volatility smiles, skews, term structures, and 3D volatility surfaces.

