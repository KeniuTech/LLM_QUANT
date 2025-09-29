"""Reusable quantitative indicator helpers."""
from __future__ import annotations

from statistics import pstdev
from typing import Iterable, Sequence


def _to_float_list(values: Iterable[object]) -> list[float]:
    cleaned: list[float] = []
    for value in values:
        try:
            cleaned.append(float(value))
        except (TypeError, ValueError):
            continue
    return cleaned


def momentum(series: Sequence[object], window: int) -> float:
    """Return simple momentum ratio over ``window`` periods.

    ``series`` is expected to be ordered from most recent to oldest. ``0.0`` is
    returned when insufficient history or the denominator is invalid.
    """

    if window <= 0:
        return 0.0
    numeric = _to_float_list(series)
    if len(numeric) < window:
        return 0.0
    latest = numeric[0]
    past = numeric[window - 1]
    if past == 0.0:
        return 0.0
    try:
        return (latest / past) - 1.0
    except ZeroDivisionError:
        return 0.0


def volatility(series: Sequence[object], window: int) -> float:
    """Compute population standard deviation of simple returns."""

    if window <= 1:
        return 0.0
    numeric = _to_float_list(series)
    if len(numeric) < 2:
        return 0.0
    limit = min(window, len(numeric) - 1)
    returns: list[float] = []
    for idx in range(limit):
        current = numeric[idx]
        previous = numeric[idx + 1]
        if previous == 0.0:
            continue
        returns.append((current / previous) - 1.0)
    if len(returns) < 2:
        return 0.0
    return float(pstdev(returns))


def rolling_mean(series: Sequence[object], window: int) -> float:
    """Return the arithmetic mean over the latest ``window`` observations."""

    if window <= 0:
        return 0.0
    numeric = _to_float_list(series)
    if not numeric:
        return 0.0
    subset = numeric[: min(window, len(numeric))]
    if not subset:
        return 0.0
    return float(sum(subset) / len(subset))


def normalize(value: object, *, factor: float | None = None, clamp: tuple[float, float] = (0.0, 1.0)) -> float:
    """Clamp ``value`` into the ``clamp`` interval after optional scaling."""

    if clamp[0] > clamp[1]:
        raise ValueError("clamp minimum cannot exceed maximum")
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return clamp[0]
    if factor and factor > 0:
        numeric = numeric / factor
    return max(clamp[0], min(clamp[1], numeric))
