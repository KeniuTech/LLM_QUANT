"""Backtest engine skeleton for daily bar simulation."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Mapping

from app.agents.base import AgentContext
from app.agents.departments import DepartmentManager
from app.agents.game import Decision, decide
from app.agents.registry import default_agents
from app.utils.data_access import DataBroker
from app.utils.config import get_config
from app.utils.db import db_session
from app.utils.logging import get_logger


LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "backtest"}


def _compute_momentum(values: List[float], window: int) -> float:
    if window <= 0 or len(values) < window:
        return 0.0
    latest = values[0]
    past = values[window - 1]
    if past is None or past == 0:
        return 0.0
    try:
        return (latest / past) - 1.0
    except ZeroDivisionError:
        return 0.0


def _compute_volatility(values: List[float], window: int) -> float:
    if len(values) < 2 or window <= 1:
        return 0.0
    limit = min(window, len(values) - 1)
    returns: List[float] = []
    for idx in range(limit):
        current = values[idx]
        previous = values[idx + 1]
        if previous is None or previous == 0:
            continue
        returns.append((current / previous) - 1.0)
    if len(returns) < 2:
        return 0.0
    return float(pstdev(returns))


def _normalize(value: Any, factor: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if factor <= 0:
        return max(0.0, min(1.0, numeric))
    return max(0.0, min(1.0, numeric / factor))


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
        self.data_broker = DataBroker()
        department_scope: set[str] = set()
        for settings in app_cfg.departments.values():
            department_scope.update(settings.data_scope)
        base_scope = {
            "daily.close",
            "daily.open",
            "daily.high",
            "daily.low",
            "daily.pct_chg",
            "daily.vol",
            "daily.amount",
            "daily_basic.turnover_rate",
            "daily_basic.turnover_rate_f",
            "daily_basic.volume_ratio",
            "stk_limit.up_limit",
            "stk_limit.down_limit",
        }
        self.required_fields = sorted(base_scope | department_scope)

    def load_market_data(self, trade_date: date) -> Mapping[str, Dict[str, Any]]:
        """Load per-stock feature vectors and context slices for the trade date."""

        trade_date_str = trade_date.strftime("%Y%m%d")
        feature_map: Dict[str, Dict[str, Any]] = {}
        universe = self.cfg.universe or []
        for ts_code in universe:
            scope_values = self.data_broker.fetch_latest(
                ts_code,
                trade_date_str,
                self.required_fields,
            )

            closes = self.data_broker.fetch_series(
                "daily",
                "close",
                ts_code,
                trade_date_str,
                window=60,
            )
            close_values = [value for _date, value in closes]
            mom20 = _compute_momentum(close_values, 20)
            mom60 = _compute_momentum(close_values, 60)
            volat20 = _compute_volatility(close_values, 20)

            turnover_series = self.data_broker.fetch_series(
                "daily_basic",
                "turnover_rate",
                ts_code,
                trade_date_str,
                window=20,
            )
            turnover_values = [value for _date, value in turnover_series]
            turn20 = mean(turnover_values) if turnover_values else 0.0

            liquidity_score = _normalize(turn20, factor=20.0)
            cost_penalty = _normalize(scope_values.get("daily_basic.volume_ratio", 0.0), factor=50.0)

            latest_close = scope_values.get("daily.close", 0.0)
            latest_pct = scope_values.get("daily.pct_chg", 0.0)
            latest_turnover = scope_values.get("daily_basic.turnover_rate", 0.0)

            up_limit = scope_values.get("stk_limit.up_limit")
            limit_up = False
            if up_limit and latest_close:
                limit_up = latest_close >= up_limit * 0.999

            down_limit = scope_values.get("stk_limit.down_limit")
            limit_down = False
            if down_limit and latest_close:
                limit_down = latest_close <= down_limit * 1.001

            is_suspended = self.data_broker.fetch_flags(
                "suspend",
                ts_code,
                trade_date_str,
                "suspend_date <= ? AND (resume_date IS NULL OR resume_date > ?)",
                (trade_date_str, trade_date_str),
            )

            features = {
                "mom_20": mom20,
                "mom_60": mom60,
                "volat_20": volat20,
                "turn_20": turn20,
                "liquidity_score": liquidity_score,
                "cost_penalty": cost_penalty,
                "news_heat": scope_values.get("news.heat_score", 0.0),
                "news_sentiment": scope_values.get("news.sentiment_index", 0.0),
                "industry_heat": scope_values.get("macro.industry_heat", 0.0),
                "industry_relative_mom": scope_values.get(
                    "macro.relative_strength",
                    scope_values.get("index.performance_peers", 0.0),
                ),
                "risk_penalty": min(1.0, volat20 * 5.0),
                "is_suspended": is_suspended,
                "limit_up": limit_up,
                "limit_down": limit_down,
                "position_limit": False,
            }

            market_snapshot = {
                "close": latest_close,
                "pct_chg": latest_pct,
                "turnover_rate": latest_turnover,
                "volume": scope_values.get("daily.vol", 0.0),
                "amount": scope_values.get("daily.amount", 0.0),
                "up_limit": up_limit,
                "down_limit": down_limit,
            }

            raw_payload = {
                "scope_values": scope_values,
                "close_series": closes,
                "turnover_series": turnover_series,
                "required_fields": self.required_fields,
            }

            feature_map[ts_code] = {
                "features": features,
                "market_snapshot": market_snapshot,
                "raw": raw_payload,
            }

        return feature_map

    def simulate_day(self, trade_date: date, state: PortfolioState) -> List[Decision]:
        feature_map = self.load_market_data(trade_date)
        decisions: List[Decision] = []
        for ts_code, payload in feature_map.items():
            features = payload.get("features", {})
            market_snapshot = payload.get("market_snapshot", {})
            raw = payload.get("raw", {})
            context = AgentContext(
                ts_code=ts_code,
                trade_date=trade_date.isoformat(),
                features=features,
                market_snapshot=market_snapshot,
                raw=raw,
            )
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
                    if dept_decision.supplements:
                        metadata["_supplements"] = dept_decision.supplements
                    if dept_decision.dialogue:
                        metadata["_dialogue"] = dept_decision.dialogue
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
            "_scope_values": context.raw.get("scope_values", {}),
            "_close_series": context.raw.get("close_series", []),
            "_turnover_series": context.raw.get("turnover_series", []),
            "_department_supplements": {
                code: dept.supplements
                for code, dept in decision.department_decisions.items()
                if dept.supplements
            },
            "_department_dialogue": {
                code: dept.dialogue
                for code, dept in decision.department_decisions.items()
                if dept.dialogue
            },
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
