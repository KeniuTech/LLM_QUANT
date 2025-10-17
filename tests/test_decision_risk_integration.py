"""Integration-style tests for risk-aware decision execution."""
from __future__ import annotations

from datetime import date

import pytest

from app.agents.base import AgentAction, AgentContext
from app.agents.game import decide
from app.agents.registry import default_agents
from app.agents.risk import RiskAgent, RiskRecommendation
from app.backtest.engine import BacktestEngine, BacktestResult, BtConfig, PortfolioState


def _make_context(features: dict, *, alerts: list[str] | None = None) -> AgentContext:
    raw = {
        "scope_values": {"daily.close": 100.0},
    }
    if alerts:
        raw["risk_alerts"] = alerts
    return AgentContext(
        ts_code="000001.SZ",
        trade_date="2025-01-10",
        features=features,
        market_snapshot={},
        raw=raw,
    )


class _StubRiskAgent(RiskAgent):
    def __init__(self, recommendation: RiskRecommendation) -> None:
        super().__init__()
        self._recommendation = recommendation

    def assess(
        self,
        context: AgentContext,
        decision_action: AgentAction,
        conflict_flag: bool,
    ) -> RiskRecommendation:
        return self._recommendation


def test_decide_adjusts_execution_on_risk_recommendation(monkeypatch):
    agents = default_agents()
    recommendation = RiskRecommendation(
        status="blocked",
        reason="risk_penalty_extreme",
        recommended_action=AgentAction.HOLD,
        notes={"risk_penalty": 0.95},
    )
    stub_risk = _StubRiskAgent(recommendation)
    agents = [stub_risk if isinstance(agent, RiskAgent) else agent for agent in agents]

    context = _make_context({"risk_penalty": 0.95})

    decision = decide(
        context,
        agents,
        weights={agent.name: 1.0 for agent in agents},
        department_manager=None,
    )
    assert decision.requires_review is True
    assert decision.risk_assessment
    assert decision.risk_assessment.status == "blocked"
    assert decision.risk_assessment.recommended_action == AgentAction.HOLD

    execution_rounds = [round for round in decision.rounds if round.agenda == "execution_summary"]
    assert execution_rounds
    execution_notes = execution_rounds[0].notes
    assert execution_notes.get("execution_status") == "risk_adjusted"
    assert execution_rounds[0].outcome == AgentAction.HOLD.value


def test_backtest_engine_applies_risk_adjusted_execution(monkeypatch):
    cfg = BtConfig(
        id="risk-test",
        name="risk-test",
        start_date=date(2025, 1, 10),
        end_date=date(2025, 1, 10),
        universe=["000001.SZ"],
        params={},
    )
    engine = BacktestEngine(cfg)
    state = PortfolioState(cash=100_000.0)
    result = BacktestResult()

    context = _make_context({"risk_penalty": 0.95})
    recommendation = RiskRecommendation(
        status="blocked",
        reason="risk_penalty_extreme",
        recommended_action=AgentAction.HOLD,
        notes={"risk_penalty": 0.95},
    )

    agents = [
        _StubRiskAgent(recommendation) if isinstance(agent, RiskAgent) else agent
        for agent in engine.agents
    ]
    engine.agents = agents
    engine.department_manager = None

    decision = decide(
        context,
        engine.agents,
        engine.weights,
        department_manager=None,
    )

    engine._apply_portfolio_updates(
        date(2025, 1, 10),
        state,
        [("000001.SZ", context, decision)],
        result,
    )

    assert not state.holdings
    assert not result.trades
    assert result.nav_series[0]["nav"] == pytest.approx(100_000.0)


def test_decide_records_suspension_risk_round():
    agents = default_agents()
    context = _make_context({"is_suspended": True})

    decision = decide(
        context,
        agents,
        weights={agent.name: 1.0 for agent in agents},
        department_manager=None,
    )

    assert decision.requires_review is True
    assert decision.risk_assessment
    assert decision.risk_assessment.status == "blocked"
    assert decision.risk_assessment.reason == "suspended"

    risk_rounds = [summary for summary in decision.rounds if summary.agenda == "risk_review"]
    assert risk_rounds
    notes = risk_rounds[0].notes
    assert notes.get("status") == "blocked"
    assert notes.get("reason") == "suspended"
