"""Shared type definitions for option analytics."""

from enum import StrEnum


class OptionType(StrEnum):
    """Supported vanilla European option types."""

    CALL = "call"
    PUT = "put"

