"""Reusable reporting summaries for the analytics platform."""

from __future__ import annotations

import pandas as pd


def summarize_numeric_columns(
    table: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Return compact summary statistics for selected numeric columns."""
    missing = set(columns).difference(table.columns)
    if missing:
        raise ValueError(f"table is missing required columns: {sorted(missing)}")

    summary = table[columns].agg(["mean", "std", "min", "median", "max"]).T
    summary.index.name = "metric"

    return summary.reset_index()


def build_step12_analytics_summary(
    monte_carlo_summary: dict[str, float],
    hedging_summary: dict[str, float],
    stress_summary: dict[str, float],
) -> pd.DataFrame:
    """Combine headline metrics from pricing, hedging, and stress analytics."""
    rows = []
    for section, values in [
        ("monte_carlo", monte_carlo_summary),
        ("hedging", hedging_summary),
        ("stress", stress_summary),
    ]:
        for metric, value in values.items():
            rows.append({"section": section, "metric": metric, "value": float(value)})

    return pd.DataFrame(rows)

