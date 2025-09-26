"""Momentum oriented agent."""
from __future__ import annotations

from math import tanh

from .base import Agent, AgentAction, AgentContext


def _sigmoid(x: float) -> float:
    return 0.5 * (tanh(x) + 1)


class MomentumAgent(Agent):
    def __init__(self) -> None:
        super().__init__(name="A_mom")

    def score(self, context: AgentContext, action: AgentAction) -> float:
        mom20 = context.features.get("mom_20", 0.0)
        mom60 = context.features.get("mom_60", 0.0)
        strength = _sigmoid(0.5 * mom20 + 0.5 * mom60)
        if action is AgentAction.SELL:
            return 1 - strength
        if action is AgentAction.HOLD:
            return 0.5
        if action is AgentAction.BUY_S:
            return strength * 0.6
        if action is AgentAction.BUY_M:
            return strength * 0.8
        return strength
