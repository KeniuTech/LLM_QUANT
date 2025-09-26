"""Feature engineering for signals and indicator computation."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, List


@dataclass
class FactorSpec:
    name: str
    window: int


@dataclass
class FactorResult:
    ts_code: str
    trade_date: date
    values: dict


DEFAULT_FACTORS: List[FactorSpec] = [
    FactorSpec("mom_20", 20),
    FactorSpec("mom_60", 60),
    FactorSpec("volat_20", 20),
    FactorSpec("turn_20", 20),
]


def compute_factors(trade_date: date, factors: Iterable[FactorSpec] = DEFAULT_FACTORS) -> List[FactorResult]:
    """Calculate factor values for the requested date.

    This function should join historical price data, apply rolling windows, and
    persist results into an factors table. The implementation is left as future
    work.
    """

    _ = trade_date, factors
    raise NotImplementedError
