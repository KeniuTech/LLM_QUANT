"""Agent abstractions for the multi-agent decision engine."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Mapping


class AgentAction(str, Enum):
    SELL = "SELL"
    HOLD = "HOLD"
    BUY_S = "BUY_S"
    BUY_M = "BUY_M"
    BUY_L = "BUY_L"


@dataclass
class AgentContext:
    ts_code: str
    trade_date: str
    features: Mapping[str, float]


class Agent:
    """Base class for all decision agents."""

    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    def score(self, context: AgentContext, action: AgentAction) -> float:
        """Return a normalized utility value in [0,1] for the proposed action."""

        raise NotImplementedError

    def feasible(self, context: AgentContext, action: AgentAction) -> bool:
        """Optional hook for agents with veto power (defaults to True)."""

        _ = context, action
        return True


UtilityMatrix = Dict[AgentAction, Dict[str, float]]
