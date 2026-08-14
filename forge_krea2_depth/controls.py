"""Strict validation for Forge UI and always-on API control values."""

from __future__ import annotations

import math
import numbers


def require_checkbox(value, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"Krea 2 Depth {name} must be true or false.")
    return value


def require_strength(value) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError("Krea 2 Depth strength must be a number from 0 to 2.")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 2.0:
        raise ValueError("Krea 2 Depth strength must be a finite number from 0 to 2.")
    return result


def require_resolution(value) -> int:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(
            "Krea 2 Depth resolution must be a whole number from 256 to 2048."
        )
    result = float(value)
    if (
        not math.isfinite(result)
        or not result.is_integer()
        or not 256 <= result <= 2048
    ):
        raise ValueError(
            "Krea 2 Depth resolution must be a whole number from 256 to 2048."
        )
    return int(result)


def require_preprocessor(value) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Krea 2 Depth preprocessor must be a non-empty name.")
    return value
