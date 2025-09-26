"""Factory helpers to construct the agent ensemble."""
from __future__ import annotations

from typing import Dict, List

from .base import Agent
from .liquidity import LiquidityAgent
from .macro import MacroAgent
from .momentum import MomentumAgent
from .news import NewsAgent
from .risk import RiskAgent
from .value import ValueAgent


def default_agents() -> List[Agent]:
    return [
        MomentumAgent(),
        ValueAgent(),
        NewsAgent(),
        LiquidityAgent(),
        MacroAgent(),
        RiskAgent(),
    ]


def weight_map(raw: Dict[str, float]) -> Dict[str, float]:
    total = sum(raw.values())
    if total == 0:
        return raw
    return {name: weight / total for name, weight in raw.items()}
