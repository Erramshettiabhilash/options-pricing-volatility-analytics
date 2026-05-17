# Step 9: Dynamic Greeks Hedging

Dynamic hedging turns option risk into trading actions.

An option price is useful, but a market maker also needs to know how to hedge the risk created by that option.

## 1. Why Market Makers Hedge

If a market maker sells a call, they receive premium but become short optionality.

That short call has:

```text
negative delta
negative gamma
negative vega
positive theta
```

To reduce directional exposure, the market maker buys stock against the short call delta.

## 2. Delta Hedging

Delta measures option sensitivity to spot:

```text
Delta = dV / dS
```

For an option position with quantity `q`, the stock hedge is:

```text
stock hedge = -q * option delta
```

Example:

If you are short one call:

```text
q = -1
call delta = 0.60
stock hedge = -(-1) * 0.60 = +0.60 shares
```

You buy stock to offset the short call's directional exposure.

## 3. Why Hedging Is Dynamic

Delta changes when spot moves. That change is Gamma.

High gamma means the hedge must be rebalanced frequently.

In continuous-time Black-Scholes with no transaction costs and correct volatility, delta hedging perfectly replicates the option. In real life, hedging is discrete and imperfect.

## 4. Hedge P&L

The hedge simulation tracks:

```text
option position
stock hedge
cash account
transaction costs
terminal option payoff
```

Final hedge P&L:

```text
cash + stock value + option settlement
```

For a short option, option settlement is negative when the option finishes in the money.

## 5. Hedge Errors

Hedge errors come from:

- discrete rebalancing
- realized volatility differing from implied volatility
- transaction costs
- jumps and gaps
- model misspecification
- liquidity constraints

This is why real hedging is risk management, not magic.

## 6. Gamma and Vega Hedging

Delta can be hedged with stock.

Gamma and Vega usually require other options.

A two-option hedge can solve:

```text
current gamma + q1 * hedge1_gamma + q2 * hedge2_gamma = 0
current vega  + q1 * hedge1_vega  + q2 * hedge2_vega  = 0
```

This project includes a simple linear solver for those hedge quantities.

## 7. Code

The hedging engine lives in:

```text
simulations/hedging.py
```

Example:

```python
from models.black_scholes import BlackScholesInputs
from simulations.hedging import simulate_delta_hedge
from utils.types import OptionType

inputs = BlackScholesInputs(
    spot=100.0,
    strike=100.0,
    time_to_maturity=1.0,
    risk_free_rate=0.05,
    volatility=0.20,
)

result = simulate_delta_hedge(
    inputs=inputs,
    option_type=OptionType.CALL,
    n_paths=5_000,
    n_steps=252,
    option_position=-1.0,
    seed=42,
)

print(result.summary)
```

## 8. Professional Interpretation

A good hedging report should show:

- mean hedge P&L
- hedge P&L dispersion
- tail losses
- rebalancing frequency
- transaction costs
- realized versus hedge volatility

The desk question is not just whether the model price is right. It is whether the hedge behaves under realistic market movement.

