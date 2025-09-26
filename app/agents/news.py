"""News and sentiment aware agent."""
from __future__ import annotations

from .base import Agent, AgentAction, AgentContext


class NewsAgent(Agent):
    def __init__(self) -> None:
        super().__init__(name="A_news")

    def score(self, context: AgentContext, action: AgentAction) -> float:
        heat = context.features.get("news_heat", 0.0)
        sentiment = context.features.get("news_sentiment", 0.0)
        positive = max(0.0, sentiment)
        negative = max(0.0, -sentiment)
        buy_score = min(1.0, heat * positive)
        sell_score = min(1.0, heat * negative)
        if action is AgentAction.SELL:
            return sell_score
        if action is AgentAction.HOLD:
            return 0.3 + 0.4 * (1 - heat)
        if action is AgentAction.BUY_S:
            return 0.5 * buy_score
        if action is AgentAction.BUY_M:
            return 0.75 * buy_score
        return buy_score
