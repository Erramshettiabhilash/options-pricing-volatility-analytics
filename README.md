# Options Pricing & Volatility Analytics Platform

An institutional-style derivatives analytics project built from scratch in Python.

This repository grows step by step into a complete options pricing and volatility analytics toolkit covering:

- Black-Scholes pricing
- Greeks
- Monte Carlo simulation
- Implied volatility
- Volatility smiles, skews, and surfaces
- SABR stochastic volatility
- Dynamic hedging
- Options market making
- Stress testing
- Professional visualization and reporting

## Current Status
 The platform includes modular pricing, analytics, simulation, visualization, tests, project metadata, architecture documentation, sample data, and reproducible final demo outputs.



### Tools

- **Python**: the core programming language for numerical research and production analytics.
- **NumPy**: fast vectorized arrays for simulations, payoffs, and numerical calculations.
- **SciPy**: numerical optimization, probability distributions, and root-finding.
- **Pandas**: market data tables, option chains, experiment outputs, and scenario results.
- **Matplotlib**: static research plots for reports and notebooks.
- **Plotly**: interactive volatility surfaces, dashboards, and exploratory analytics.
- **Pytest**: repeatable unit tests so formulas and numerical methods stay correct as the project grows.

### Setup

From the project root:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Folder Structure

```text
data/             Market data, option chains, and sample datasets.
models/           Pricing models such as Black-Scholes and SABR.
analytics/        Greeks, implied volatility, stress testing, and sensitivity analysis.
simulations/      Monte Carlo engines and dynamic hedging simulations.
visualization/    Matplotlib and Plotly charting utilities.
utils/            Shared types, validation, math helpers, and constants.
notebooks/        Research notebooks for experiments and explanation.
results/          Generated charts, tables, and experiment outputs.
tests/            Unit tests and numerical validation checks.
```

## Roadmap

The project is intentionally modular: formulas live in `models`, risk calculations live in `analytics`, simulations live in `simulations`, and charts live in `visualization`. This mirrors a real derivatives research workflow where pricing, calibration, hedging, and reporting are separate but connected layers.

## Quality Checks

Run the full test suite:

```bash
python -m pytest
```

Run a syntax/import compilation pass:

```bash
python -m compileall analytics models simulations utils visualization tests
```

Architecture notes are in:

```text
docs/ARCHITECTURE.md
```

## End-to-End Demo

Run the final demo:

```bash
python examples/run_end_to_end_demo.py
```

It reads:

```text
data/sample_option_chain.csv
```

and writes a compact tracked example report to:

```text
results/final_demo/
```

## Documentation

- `docs/ARCHITECTURE.md`: project architecture and conventions
- `docs/MATHEMATICAL_REFERENCE.md`: core formulas
- `docs/FINAL_REPORT.md`: final project summary and interview talking points
- `CONTRIBUTING.md`: development and testing checklist

## Final Claim

By completing this project, you can confidently say:

> I implemented a complete options pricing and volatility analytics platform from scratch including Black-Scholes pricing, Greeks computation, Monte Carlo simulation, implied volatility estimation, volatility surfaces, SABR stochastic volatility modeling, dynamic hedging systems, options market making, and stress testing.



REPOSITORY CLONE 

https://github.com/Erramshettiabhilash/options-pricing-volatility-analytics
