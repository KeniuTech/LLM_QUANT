"""Tests for RiskAgent assessment and risk evaluation pipeline."""
from __future__ import annotations

from app.agents.base import AgentAction, AgentContext
from app.agents.game import _evaluate_risk
from app.agents.risk import RiskAgent, RiskRecommendation


def _make_context(**features: float | bool) -> AgentContext:
    return AgentContext(
        ts_code="000001.SZ",
        trade_date="2025-01-01",
        features=features,
        market_snapshot={},
        raw={},
    )


def test_risk_agent_ok_status() -> None:
    agent = RiskAgent()
    context = _make_context(risk_penalty=0.1)
    recommendation = agent.assess(context, AgentAction.BUY_S, conflict_flag=False)
    assert recommendation.status == "ok"
    assert recommendation.reason == "clear"


def test_risk_agent_blocked_on_limit_up() -> None:
    agent = RiskAgent()
    context = _make_context(limit_up=True)
    recommendation = agent.assess(context, AgentAction.BUY_M, conflict_flag=False)
    assert recommendation.status == "blocked"
    assert recommendation.reason == "limit_up"
    assert recommendation.recommended_action == AgentAction.HOLD


def test_risk_agent_pending_on_conflict() -> None:
    agent = RiskAgent()
    context = _make_context()
    recommendation = agent.assess(context, AgentAction.HOLD, conflict_flag=True)
    assert recommendation.status == "pending_review"
    assert recommendation.reason == "conflict_threshold"


def test_evaluate_risk_external_alerts() -> None:
    agent = RiskAgent()
    context = AgentContext(
        ts_code="000002.SZ",
        trade_date="2025-01-01",
        features={"risk_penalty": 0.1},
        market_snapshot={},
        raw={"risk_alerts": ["sudden_news"]},
    )
    assessment = _evaluate_risk(
        context=context,
        action=AgentAction.HOLD,
        department_votes={"buy": 0.6},
        conflict_flag=False,
        risk_agent=agent,
    )
    assert assessment.status == "pending_review"
    assert assessment.reason == "external_alert"
    assert assessment.recommended_action == AgentAction.HOLD
    assert "external_alerts" in assessment.notes
