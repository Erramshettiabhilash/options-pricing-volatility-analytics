# Final Project Report

## Project Statement

This project implements a complete options pricing and volatility analytics platform from scratch in Python.

It covers:

- Black-Scholes pricing
- analytical Greeks
- Monte Carlo pricing
- implied volatility extraction
- volatility smiles and surfaces
- SABR stochastic volatility calibration
- dynamic delta hedging
- simplified options market making
- deterministic and Monte Carlo stress testing
- professional visualizations and reporting

## What Was Built

### Pricing Models

- European call and put pricing
- manual standard normal CDF/PDF
- expiry and zero-volatility edge cases
- no-arbitrage validation through put-call parity tests

### Risk Analytics

- Delta, Gamma, Vega, Theta, and Rho
- implied volatility via Newton-Raphson and Bisection
- volatility smile, skew, term structure, and surface analytics
- stress testing and reporting summaries

### Simulation Engines

- Brownian Motion and GBM path generation
- Monte Carlo option pricing
- dynamic delta hedging with cash-account accounting
- options market-making session simulation
- Monte Carlo portfolio stress testing

### Visualization

- payoff diagrams
- Greeks curves
- Monte Carlo convergence
- volatility smiles and surfaces
- SABR calibration fit
- hedging P&L
- inventory and spread capture
- scenario stress P&L

## Interview Talking Points

You can describe the platform as:

> I implemented a modular derivatives analytics platform from scratch, including Black-Scholes pricing, Greeks, Monte Carlo pricing, implied volatility solvers, volatility surface analytics, SABR calibration, dynamic hedging, options market making, and stress testing.

Strong technical points to emphasize:

- risk-neutral pricing separates real-world drift from pricing drift
- Greeks convert prices into hedgeable risk sensitivities
- implied volatility is an inverse pricing problem
- volatility surfaces reveal market skew and term structure
- SABR turns a market smile into interpretable stochastic-volatility parameters
- hedging errors arise from discrete rebalancing, costs, and model mismatch
- market-making spreads compensate for inventory, gamma, vega, and adverse selection
- stress tests complement local Greeks because large shocks are nonlinear

## Verification

The final verification commands are:

```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m compileall analytics models simulations utils visualization tests examples
.\.venv\Scripts\python examples\run_end_to_end_demo.py
```

