"""Risk agent acts as leader with veto rights."""
from __future__ import annotations

from .base import Agent, AgentAction, AgentContext


class RiskAgent(Agent):
    def __init__(self) -> None:
        super().__init__(name="A_risk")

    def score(self, context: AgentContext, action: AgentAction) -> float:
        # Base risk agent is neutral unless penalties are triggered.
        penalty = context.features.get("risk_penalty", 0.0)
        if action is AgentAction.SELL:
            return min(1.0, 0.6 + penalty)
        if action is AgentAction.HOLD:
            return 0.5
        return max(0.0, 1.0 - penalty)

    def feasible(self, context: AgentContext, action: AgentAction) -> bool:
        if action is AgentAction.SELL:
            return True
        if context.features.get("is_suspended", False):
            return False
        if context.features.get("limit_up", False) and action not in (AgentAction.SELL, AgentAction.HOLD):
            return False
        if context.features.get("position_limit", False) and action in (AgentAction.BUY_M, AgentAction.BUY_L):
            return False
        return True
