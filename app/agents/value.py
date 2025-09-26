"""Value and quality filtering agent."""
from __future__ import annotations

from .base import Agent, AgentAction, AgentContext


class ValueAgent(Agent):
    def __init__(self) -> None:
        super().__init__(name="A_val")

    def score(self, context: AgentContext, action: AgentAction) -> float:
        pe = context.features.get("pe_percentile", 0.5)
        pb = context.features.get("pb_percentile", 0.5)
        roe = context.features.get("roe_percentile", 0.5)
        # Lower valuation percentiles and higher quality percentiles add value.
        raw = max(0.0, (1 - pe) * 0.4 + (1 - pb) * 0.3 + roe * 0.3)
        raw = min(raw, 1.0)
        if action is AgentAction.SELL:
            return 1 - raw
        if action is AgentAction.HOLD:
            return 0.5
        if action is AgentAction.BUY_S:
            return raw * 0.7
        if action is AgentAction.BUY_M:
            return raw * 0.85
        return raw
