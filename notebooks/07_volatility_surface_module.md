# Step 7: Volatility Surface Module

A volatility surface extends implied volatility analysis across both strike and maturity.

In Step 6, we extracted implied volatility from one option price. In Step 7, we organize many implied volatilities into a surface:

```text
IV = f(strike, maturity)
```

This is closer to how an institutional options desk thinks about the market.

## 1. Smile

A volatility smile is implied volatility across strikes for one maturity.

```text
strike -> implied volatility
```

If Black-Scholes were perfectly true, all strikes would imply the same volatility. Real markets usually show a curve.

## 2. Skew

Skew is the slope of implied volatility across strikes or moneyness.

Equity index markets often show negative skew:

```text
lower strike puts -> higher implied volatility
```

Why?

- investors buy downside protection
- crashes are asymmetric
- spot and volatility often move in opposite directions
- market makers charge more for tail risk

## 3. Term Structure

Term structure is implied volatility across maturities, usually around a fixed moneyness such as ATM.

```text
maturity -> implied volatility
```

It tells us whether short-dated or long-dated uncertainty is expensive.

Examples:

- event risk can lift short-dated IV
- macro uncertainty can lift long-dated IV
- crisis markets can invert the term structure

## 4. 3D Volatility Surface

The surface combines smile and term structure:

```text
x-axis: strike
y-axis: maturity
z-axis: implied volatility
```

Desks use volatility surfaces for:

- marking option books
- interpolating missing market quotes
- risk reports
- scenario analysis
- model calibration
- market making quotes

## 5. Moneyness and Log-Moneyness

Moneyness:

```text
K / S
```

Log-moneyness:

```text
log(K / S)
```

Log-moneyness is often cleaner for modeling because it is symmetric around ATM when `K = S`.

## 6. Why OTM Puts Often Have Higher IV

For equity markets, out-of-the-money puts are crash insurance. Demand for that protection pushes put prices up.

Since implied volatility is the volatility that explains the price, higher put prices become higher implied volatilities.

This is why equity index smiles often look more like a skew than a symmetric smile.

## 7. Code

Analytics live in:

```text
analytics/volatility_surface.py
```

Visualization lives in:

```text
visualization/volatility.py
```

Main functions:

```text
build_implied_volatility_surface
add_moneyness_columns
smile_for_maturity
term_structure_at_moneyness
skew_slope_by_maturity
pivot_volatility_surface
interpolate_volatility_surface
```

## 8. Professional Workflow

A production volatility-surface workflow usually does this:

```text
1. Load option chain
2. Validate prices
3. Extract implied volatility
4. Convert strikes to moneyness/log-moneyness
5. Inspect smile and skew by maturity
6. Inspect term structure
7. Interpolate onto a regular grid
8. Use the surface for marking, risk, calibration, and quoting
```

This step prepares us for Step 8: SABR stochastic volatility calibration.

