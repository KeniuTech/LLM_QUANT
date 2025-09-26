"""Macro and industry regime agent."""
from __future__ import annotations

from .base import Agent, AgentAction, AgentContext


class MacroAgent(Agent):
    def __init__(self) -> None:
        super().__init__(name="A_macro")

    def score(self, context: AgentContext, action: AgentAction) -> float:
        industry_heat = context.features.get("industry_heat", 0.5)
        relative_strength = context.features.get("industry_relative_mom", 0.0)
        raw = min(1.0, max(0.0, industry_heat * 0.6 + relative_strength * 0.4))
        if action is AgentAction.SELL:
            return 1 - raw
        if action is AgentAction.HOLD:
            return 0.5
        if action is AgentAction.BUY_S:
            return raw * 0.6
        if action is AgentAction.BUY_M:
            return raw * 0.8
        return raw
