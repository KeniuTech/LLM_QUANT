"""Backtest engine skeleton for daily bar simulation."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Iterable, List, Mapping

from app.agents.base import AgentContext
from app.agents.game import Decision, decide
from app.agents.registry import default_agents
from app.utils.db import db_session


@dataclass
class BtConfig:
    id: str
    name: str
    start_date: date
    end_date: date
    universe: List[str]
    params: Dict[str, float]
    method: str = "nash"


@dataclass
class PortfolioState:
    cash: float = 1_000_000.0
    holdings: Dict[str, float] = field(default_factory=dict)


@dataclass
class BacktestResult:
    nav_series: List[Dict[str, float]] = field(default_factory=list)
    trades: List[Dict[str, str]] = field(default_factory=list)


class BacktestEngine:
    """Runs the multi-agent game inside a daily event-driven loop."""

    def __init__(self, cfg: BtConfig) -> None:
        self.cfg = cfg
        self.agents = default_agents()
        self.weights = {agent.name: 1.0 for agent in self.agents}

    def load_market_data(self, trade_date: date) -> Mapping[str, Dict[str, float]]:
        """Load per-stock feature vectors. Replace with real data access."""

        _ = trade_date
        return {}

    def simulate_day(self, trade_date: date, state: PortfolioState) -> List[Decision]:
        feature_map = self.load_market_data(trade_date)
        decisions: List[Decision] = []
        for ts_code, features in feature_map.items():
            context = AgentContext(ts_code=ts_code, trade_date=trade_date.isoformat(), features=features)
            decision = decide(context, self.agents, self.weights, method=self.cfg.method)
            decisions.append(decision)
            self.record_agent_state(context, decision)
        # TODO: translate decisions into fills, holdings, and NAV updates.
        _ = state
        return decisions

    def record_agent_state(self, context: AgentContext, decision: Decision) -> None:
        payload = {
            "trade_date": context.trade_date,
            "ts_code": context.ts_code,
            "action": decision.action.value,
            "confidence": decision.confidence,
        }
        _ = payload
        # Implementation should persist into agent_utils and bt_trades.

    def run(self) -> BacktestResult:
        state = PortfolioState()
        result = BacktestResult()
        current = self.cfg.start_date
        while current <= self.cfg.end_date:
            decisions = self.simulate_day(current, state)
            _ = decisions
            current = date.fromordinal(current.toordinal() + 1)
        return result


def run_backtest(cfg: BtConfig) -> BacktestResult:
    engine = BacktestEngine(cfg)
    result = engine.run()
    with db_session() as conn:
        _ = conn
        # Implementation should persist bt_nav, bt_trades, and bt_report rows.
    return result
