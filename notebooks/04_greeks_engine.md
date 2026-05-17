# Step 4: Greeks Engine

Greeks turn an option price into a risk report. A price tells you what the option is worth now. Greeks tell you how that value changes when the market moves.

## 1. Why Greeks Matter

Options are nonlinear instruments. A stock position has simple directional exposure: if you own 100 shares, you gain roughly 100 dollars for every 1 dollar increase in the stock.

An option is different. Its exposure changes with:

- spot price
- volatility
- time
- interest rates
- moneyness

Greeks measure those sensitivities.

## 2. Delta

Delta is sensitivity to the underlying price:

```text
Delta = dV / dS
```

For Black-Scholes:

```text
Call Delta = N(d1)
Put Delta  = N(d1) - 1
```

Trading intuition:

- Call delta is between 0 and 1.
- Put delta is between -1 and 0.
- A call with delta 0.60 behaves like roughly 60 shares per 100 option contracts, before contract multiplier.

Delta is the core hedge ratio for directional risk.

## 3. Gamma

Gamma is the rate of change of delta:

```text
Gamma = d²V / dS²
```

For calls and puts:

```text
Gamma = phi(d1) / (S * sigma * sqrt(T))
```

Trading intuition:

Gamma tells you how unstable your delta hedge is. High gamma means your hedge ratio changes quickly when spot moves.

Market makers watch gamma carefully because a short-gamma book needs frequent rebalancing and can lose money in fast markets.

## 4. Vega

Vega is sensitivity to volatility:

```text
Vega = dV / d sigma
```

Black-Scholes:

```text
Vega = S * phi(d1) * sqrt(T)
```

Trading intuition:

Long options are usually long vega. If implied volatility rises, long calls and long puts become more valuable.

Professional convention:

The mathematical formula gives vega per 1.00 volatility change. Traders often quote vega per one volatility point, so they divide this number by 100.

## 5. Theta

Theta is sensitivity to time passing:

```text
Theta = dV / dt
```

Call theta:

```text
Theta_call =
-(S * phi(d1) * sigma) / (2 * sqrt(T))
- rK exp(-rT) N(d2)
```

Put theta:

```text
Theta_put =
-(S * phi(d1) * sigma) / (2 * sqrt(T))
+ rK exp(-rT) N(-d2)
```

Trading intuition:

Theta is time decay. Long options usually lose time value as expiry approaches. Short option sellers often collect theta but take gamma and vega risk.

This project reports annualized theta. Divide by 365 or 252 for a daily convention.

## 6. Rho

Rho is sensitivity to interest rates:

```text
Rho = dV / dr
```

Call rho:

```text
Rho_call = K T exp(-rT) N(d2)
```

Put rho:

```text
Rho_put = -K T exp(-rT) N(-d2)
```

Trading intuition:

Rates affect the present value of the strike. Calls typically have positive rho; puts typically have negative rho.

## 7. Code

The Greeks engine lives in:

```text
analytics/greeks.py
```

Example:

```python
from analytics.greeks import calculate_greeks
from models.black_scholes import BlackScholesInputs
from utils.types import OptionType

inputs = BlackScholesInputs(
    spot=100.0,
    strike=100.0,
    time_to_maturity=1.0,
    risk_free_rate=0.05,
    volatility=0.20,
)

greeks = calculate_greeks(inputs, OptionType.CALL)
print(greeks)
```

Expected benchmark:

```text
Delta = 0.636831
Gamma = 0.018762
Vega  = 37.524035
Theta = -6.414028
Rho   = 53.232482
```

## 8. Hedging Interpretation

If a market maker sells one call with delta 0.64, they are short approximately 0.64 shares of directional exposure. To delta hedge, they buy 0.64 shares.

But if spot moves, gamma changes delta. If implied volatility moves, vega changes the option value. If time passes, theta changes the book. That is why a derivatives desk needs a full Greeks engine, not just a pricing function.

