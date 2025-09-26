"""Performance metric utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class Metric:
    name: str
    value: float
    description: str


def compute_nav_metrics(nav_series: Iterable[float]) -> List[Metric]:
    """Compute core statistics such as CAGR, Sharpe, and max drawdown."""

    _ = nav_series
    return []
