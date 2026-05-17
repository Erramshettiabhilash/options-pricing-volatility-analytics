# Step 10: Options Market Making

Options market making combines pricing, risk, inventory, and execution.

The simplified loop is:

```text
price -> quote -> trade -> inventory -> risk -> adjust quote
```

## 1. Fair Value vs Quote

Black-Scholes gives fair value. A market maker quotes around it:

```text
bid < fair value < ask
```

The spread compensates for hedging cost, inventory risk, gamma risk, vega risk, and adverse selection.

## 2. Spread Capture

If a customer buys from the market maker, the customer pays the ask.

```text
spread capture = ask - fair value
```

If a customer sells to the market maker, the customer receives the bid.

```text
spread capture = fair value - bid
```

Spread capture is compensation for risk. It is not guaranteed profit.

## 3. Inventory Management

Long inventory shifts quotes lower to encourage selling inventory.

Short inventory shifts quotes higher to make new shorts more expensive and encourage customers to sell options back.

## 4. Gamma Exposure Monitoring

Gamma measures how quickly delta changes. Short-gamma books are dangerous in fast markets because the hedge has to be chased.

The quoting engine widens spreads when inventory creates larger gamma exposure.

## 5. Theta Decay

Theta measures time decay. Short option positions often have positive book theta because:

```text
short quantity * negative option theta = positive theta exposure
```

That theta is only earned if realized movement and hedging costs do not overwhelm it.

## 6. Adverse Selection

Adverse selection means customer flow may be informed or arrive just before market movement.

When order flow looks toxic, the engine widens the spread.

## 7. Code

The market-making engine lives in:

```text
simulations/market_making.py
```

Main functions:

```text
quote_option
execute_quote_trade
market_making_risk_report
aggregate_risk_report
theta_decay_analysis
simulate_market_making_session
```

Visualization:

```text
visualization/market_making.py
```

This is simplified, but it captures the desk workflow that connects model value to executable quotes.

