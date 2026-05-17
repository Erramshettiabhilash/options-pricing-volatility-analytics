# Step 8: SABR Stochastic Volatility Model

SABR is a stochastic-volatility model used to fit and interpret implied-volatility smiles.

The name comes from:

```text
Stochastic Alpha Beta Rho
```

It models both the forward price and its volatility as random processes.

## 1. SABR Dynamics

The SABR model is:

```text
dF_t     = alpha_t * F_t^beta * dW_t
dalpha_t = nu * alpha_t * dZ_t
corr(dW_t, dZ_t) = rho
```

Where:

```text
F_t      forward price
alpha_t  stochastic volatility level
beta     elasticity parameter
rho      correlation between price and volatility shocks
nu       volatility of volatility
```

## 2. Parameter Intuition

`alpha` controls the general volatility level.

`beta` controls how volatility scales with the forward. Common choices:

```text
beta = 0      normal-like dynamics
beta = 0.5    square-root style elasticity
beta = 1      lognormal-style dynamics
```

`rho` controls skew. Negative `rho` usually creates equity-style downside skew: when price falls, volatility rises.

`nu` controls smile curvature. Higher `nu` means volatility itself is more volatile, creating a stronger smile.

## 3. Hagan Implied Volatility Approximation

Instead of simulating SABR directly, desks often use Hagan's approximation to map SABR parameters into Black implied volatility.

That gives:

```text
SABR parameters -> implied volatility by strike
```

This is fast enough for calibration and quoting.

## 4. Calibration

Calibration means finding SABR parameters that best match market implied volatilities.

In this project we calibrate:

```text
alpha, rho, nu
```

while holding:

```text
beta fixed
```

Why fix beta?

One smile often does not contain enough stable information to estimate all four SABR parameters. Fixing beta is a common professional workflow.

The optimization minimizes:

```text
SABR_IV(K_i) - Market_IV(K_i)
```

across strikes.

## 5. SABR vs Black-Scholes

Black-Scholes assumes one volatility for all strikes.

SABR can create a full smile:

```text
low strikes  -> higher/lower IV depending on rho
ATM          -> alpha-driven level
wings        -> controlled by nu
```

That makes SABR more realistic for market smiles.

## 6. Code

Model code lives in:

```text
models/sabr.py
```

Visualization support:

```text
visualization/volatility.py
```

Example:

```python
from models.sabr import calibrate_sabr_from_dataframe

result = calibrate_sabr_from_dataframe(
    smile=market_smile,
    forward=100.0,
    maturity=1.0,
    beta=0.5,
)

print(result.parameters)
print(result.rmse)
```

## 7. Desk Interpretation

After calibration, the parameters tell a story:

- high `alpha`: elevated volatility level
- negative `rho`: downside skew
- high `nu`: pronounced smile curvature
- stable low RMSE: SABR explains the market smile well

This calibrated smile will be useful later for market making, stress testing, and scenario analysis.

