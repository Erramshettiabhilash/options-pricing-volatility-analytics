"""Input validation helpers used across pricing and analytics modules."""

from __future__ import annotations


def validate_positive(value: float, name: str) -> None:
    """Raise a ValueError when a numeric input is not strictly positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive. Received {value}.")


def validate_non_negative(value: float, name: str) -> None:
    """Raise a ValueError when a numeric input is negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative. Received {value}.")


def validate_probability(value: float, name: str) -> None:
    """Raise a ValueError when a numeric input is not in the closed interval [0, 1]."""
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1. Received {value}.")

