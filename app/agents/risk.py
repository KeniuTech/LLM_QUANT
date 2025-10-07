"""Risk agent acts as leader with veto rights."""
from __future__ import annotations

from .base import Agent, AgentAction, AgentContext


class RiskRecommendation:
    """Represents structured recommendation from the risk agent."""

    __slots__ = ("status", "reason", "recommended_action", "notes")

    def __init__(
        self,
        *,
        status: str,
        reason: str,
        recommended_action: AgentAction | None = None,
        notes: dict | None = None,
    ) -> None:
        self.status = status
        self.reason = reason
        self.recommended_action = recommended_action
        self.notes = notes or {}

    def to_dict(self) -> dict:
        payload = {
            "status": self.status,
            "reason": self.reason,
            "notes": dict(self.notes),
        }
        if self.recommended_action is not None:
            payload["recommended_action"] = self.recommended_action.value
        return payload


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

    def assess(
        self,
        context: AgentContext,
        decision_action: AgentAction,
        conflict_flag: bool,
    ) -> RiskRecommendation:
        features = dict(context.features or {})
        risk_penalty = float(features.get("risk_penalty") or 0.0)

        if bool(features.get("is_suspended")):
            return RiskRecommendation(
                status="blocked",
                reason="suspended",
                recommended_action=AgentAction.HOLD,
                notes={"trigger": "is_suspended"},
            )

        if bool(features.get("limit_up")) and decision_action in {
            AgentAction.BUY_S,
            AgentAction.BUY_M,
            AgentAction.BUY_L,
        }:
            return RiskRecommendation(
                status="blocked",
                reason="limit_up",
                recommended_action=AgentAction.HOLD,
                notes={"trigger": "limit_up"},
            )

        if bool(features.get("position_limit")) and decision_action in {
            AgentAction.BUY_M,
            AgentAction.BUY_L,
        }:
            return RiskRecommendation(
                status="pending_review",
                reason="position_limit",
                recommended_action=AgentAction.BUY_S,
                notes={"trigger": "position_limit"},
            )

        if risk_penalty >= 0.9 and decision_action in {
            AgentAction.BUY_S,
            AgentAction.BUY_M,
            AgentAction.BUY_L,
        }:
            return RiskRecommendation(
                status="blocked",
                reason="risk_penalty_extreme",
                recommended_action=AgentAction.HOLD,
                notes={"risk_penalty": risk_penalty},
            )
        if risk_penalty >= 0.7 and decision_action in {
            AgentAction.BUY_S,
            AgentAction.BUY_M,
            AgentAction.BUY_L,
        }:
            return RiskRecommendation(
                status="pending_review",
                reason="risk_penalty_high",
                recommended_action=AgentAction.HOLD,
                notes={"risk_penalty": risk_penalty},
            )

        if conflict_flag:
            return RiskRecommendation(
                status="pending_review",
                reason="conflict_threshold",
            )

        return RiskRecommendation(status="ok", reason="clear")
