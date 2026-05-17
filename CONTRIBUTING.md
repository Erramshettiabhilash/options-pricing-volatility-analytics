# Contributing

This is a quant research codebase, so correctness matters more than cleverness.

## Development Checklist

Before submitting or presenting changes:

```powershell
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m compileall analytics models simulations utils visualization tests
```

## Code Style

- Prefer small, typed functions.
- Use dataclasses for structured model inputs and result objects.
- Validate financial inputs at module boundaries.
- Keep pricing logic out of visualization code.
- Keep notebook explanations synchronized with production modules.
- Do not use financial pricing libraries such as QuantLib or Mibian.

## Testing Expectations

Tests should cover:

- known analytical benchmarks
- no-arbitrage identities
- input validation
- numerical stability
- reproducibility with fixed seeds
- convergence behavior for iterative methods

