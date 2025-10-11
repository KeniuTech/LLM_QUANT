"""Value and quality filtering agent."""
from __future__ import annotations

from typing import Mapping

from .base import Agent, AgentAction, AgentContext


class ValueAgent(Agent):
    def __init__(self) -> None:
        super().__init__(name="A_val")

    def score(self, context: AgentContext, action: AgentAction) -> float:
        pe_score = context.features.get("valuation_pe_score", 0.0)
        pb_score = context.features.get("valuation_pb_score", 0.0)
        # 多因子组合尚未落地，这里兼容扩展因子（若存在则优先使用）
        scope_values = {}
        if isinstance(context.raw, Mapping):
            scope_values = context.raw.get("scope_values", {}) or {}
        multi_score = context.features.get("val_multiscore")
        if multi_score is None:
            multi_score = scope_values.get("factors.val_multiscore")
        if multi_score is not None:
            raw = float(max(0.0, min(1.0, multi_score)))
        else:
            raw = max(0.0, min(1.0, 0.6 * pe_score + 0.4 * pb_score))
        if action is AgentAction.SELL:
            return 1 - raw
        if action is AgentAction.HOLD:
            return 0.5
        if action is AgentAction.BUY_S:
            return raw * 0.7
        if action is AgentAction.BUY_M:
            return raw * 0.85
        return raw
