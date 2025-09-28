"""Backtest engine skeleton for daily bar simulation."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Iterable, List, Mapping

from app.agents.base import AgentContext
from app.agents.departments import DepartmentManager
from app.agents.game import Decision, decide
from app.agents.registry import default_agents
from app.utils.config import get_config
from app.utils.db import db_session
from app.utils.logging import get_logger


LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "backtest"}


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
        app_cfg = get_config()
        weight_config = app_cfg.agent_weights.as_dict() if app_cfg.agent_weights else {}
        if weight_config:
            self.weights = weight_config
        else:
            self.weights = {agent.name: 1.0 for agent in self.agents}
        self.department_manager = (
            DepartmentManager(app_cfg) if app_cfg.departments else None
        )

    def load_market_data(self, trade_date: date) -> Mapping[str, Dict[str, float]]:
        """Load per-stock feature vectors. Replace with real data access."""

        _ = trade_date
        return {}

    def simulate_day(self, trade_date: date, state: PortfolioState) -> List[Decision]:
        feature_map = self.load_market_data(trade_date)
        decisions: List[Decision] = []
        for ts_code, features in feature_map.items():
            context = AgentContext(ts_code=ts_code, trade_date=trade_date.isoformat(), features=features)
            decision = decide(
                context,
                self.agents,
                self.weights,
                method=self.cfg.method,
                department_manager=self.department_manager,
            )
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
            "department_votes": decision.department_votes,
            "requires_review": decision.requires_review,
            "departments": {
                code: dept.to_dict()
                for code, dept in decision.department_decisions.items()
            },
        }
        combined_weights = dict(self.weights)
        if self.department_manager:
            for code, agent in self.department_manager.agents.items():
                key = f"dept_{code}"
                combined_weights[key] = agent.settings.weight

        feasible_json = json.dumps(
            [action.value for action in decision.feasible_actions],
            ensure_ascii=False,
        )
        rows = []
        for agent_name, weight in combined_weights.items():
            action_scores = {
                action.value: float(decision.utilities.get(action, {}).get(agent_name, 0.0))
                for action in decision.utilities.keys()
            }
            best_action = decision.action.value
            if action_scores:
                best_action = max(action_scores.items(), key=lambda item: item[1])[0]
            metadata: Dict[str, object] = {}
            if agent_name.startswith("dept_"):
                dept_code = agent_name.split("dept_", 1)[-1]
                dept_decision = decision.department_decisions.get(dept_code)
                if dept_decision:
                    metadata = {
                        "_summary": dept_decision.summary,
                        "_signals": dept_decision.signals,
                        "_risks": dept_decision.risks,
                        "_confidence": dept_decision.confidence,
                    }
            payload_json = {**action_scores, **metadata}
            rows.append(
                (
                    context.trade_date,
                    context.ts_code,
                    agent_name,
                    best_action,
                    json.dumps(payload_json, ensure_ascii=False),
                    feasible_json,
                    float(weight),
                )
            )

        global_payload = {
            "_confidence": decision.confidence,
            "_target_weight": decision.target_weight,
            "_department_votes": decision.department_votes,
            "_requires_review": decision.requires_review,
        }
        rows.append(
            (
                context.trade_date,
                context.ts_code,
                "global",
                decision.action.value,
                json.dumps(global_payload, ensure_ascii=False),
                feasible_json,
                1.0,
            )
        )

        try:
            with db_session() as conn:
                conn.executemany(
                    """
                    INSERT OR REPLACE INTO agent_utils
                    (trade_date, ts_code, agent, action, utils, feasible, weight)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
        except Exception:
            LOGGER.exception("写入 agent_utils 失败", extra=LOG_EXTRA)
        _ = payload
        # TODO: persist payload into bt_trades / audit tables when schema is ready.

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
