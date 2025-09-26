"""Liquidity and transaction cost agent."""
from __future__ import annotations

from .base import Agent, AgentAction, AgentContext


class LiquidityAgent(Agent):
    def __init__(self) -> None:
        super().__init__(name="A_liq")

    def score(self, context: AgentContext, action: AgentAction) -> float:
        liq = context.features.get("liquidity_score", 0.5)
        cost = context.features.get("cost_penalty", 0.0)
        penalty = cost
        if action is AgentAction.SELL:
            return min(1.0, liq + penalty)
        if action is AgentAction.HOLD:
            return 0.4 + 0.2 * liq
        scale = {AgentAction.BUY_S: 0.5, AgentAction.BUY_M: 0.75, AgentAction.BUY_L: 1.0}.get(action, 0.0)
        return max(0.0, liq * scale - penalty)
