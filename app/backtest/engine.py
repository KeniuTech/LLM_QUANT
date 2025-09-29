"""Backtest engine skeleton for daily bar simulation."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from statistics import mean, pstdev
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from app.agents.base import AgentAction, AgentContext
from app.agents.departments import DepartmentManager
from app.agents.game import Decision, decide
from app.llm.metrics import record_decision as metrics_record_decision
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
    cost_basis: Dict[str, float] = field(default_factory=dict)
    opened_dates: Dict[str, str] = field(default_factory=dict)
    realized_pnl: float = 0.0


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

    def simulate_day(
        self,
        trade_date: date,
        state: PortfolioState,
        decision_callback: Optional[Callable[[str, date, AgentContext, Decision], None]] = None,
    ) -> List[tuple[str, AgentContext, Decision]]:
        feature_map = self.load_market_data(trade_date)
        records: List[tuple[str, AgentContext, Decision]] = []
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
            try:
                metrics_record_decision(
                    ts_code=ts_code,
                    trade_date=context.trade_date,
                    action=decision.action.value,
                    confidence=decision.confidence,
                    summary=_extract_summary(decision),
                    source="backtest",
                    departments={
                        code: dept.to_dict()
                        for code, dept in decision.department_decisions.items()
                    },
                )
            except Exception:  # noqa: BLE001
                LOGGER.debug("记录决策指标失败", extra=LOG_EXTRA)
            records.append((ts_code, context, decision))
            self.record_agent_state(context, decision)
            if decision_callback:
                try:
                    decision_callback(ts_code, trade_date, context, decision)
                except Exception:  # noqa: BLE001
                    LOGGER.exception("决策回调执行失败", extra=LOG_EXTRA)
        # TODO: translate decisions into fills, holdings, and NAV updates.
        _ = state
        return records

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
                    if dept_decision.telemetry:
                        metadata["_telemetry"] = dept_decision.telemetry
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
            "_department_telemetry": {
                code: dept.telemetry
                for code, dept in decision.department_decisions.items()
                if dept.telemetry
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

        try:
            self._record_investment_candidate(context, decision)
        except Exception:  # noqa: BLE001
            LOGGER.exception("写入 investment_pool 失败", extra=LOG_EXTRA)

    def _apply_portfolio_updates(
        self,
        trade_date: date,
        state: PortfolioState,
        records: List[tuple[str, AgentContext, Decision]],
        result: BacktestResult,
    ) -> None:
        trade_date_str = trade_date.isoformat()
        price_map: Dict[str, float] = {}
        decisions_map: Dict[str, Decision] = {}
        for ts_code, context, decision in records:
            scope_values = context.raw.get("scope_values") if context.raw else {}
            if not isinstance(scope_values, Mapping):
                scope_values = {}
            price = scope_values.get("daily.close") or scope_values.get("close")
            if price is None:
                continue
            try:
                price = float(price)
            except (TypeError, ValueError):
                continue
            price_map[ts_code] = price
            decisions_map[ts_code] = decision

        if not price_map and state.holdings:
            trade_date_compact = trade_date.strftime("%Y%m%d")
            for ts_code in state.holdings.keys():
                fetched = self.data_broker.fetch_latest(ts_code, trade_date_compact, ["daily.close"])
                price = fetched.get("daily.close")
                if price:
                    price_map[ts_code] = float(price)

        portfolio_value_before = state.cash
        for ts_code, qty in state.holdings.items():
            price = price_map.get(ts_code)
            if price is None:
                continue
            portfolio_value_before += qty * price

        if portfolio_value_before <= 0:
            portfolio_value_before = state.cash or 1.0

        trades_records: List[Dict[str, Any]] = []
        for ts_code, decision in decisions_map.items():
            price = price_map.get(ts_code)
            if price is None or price <= 0:
                continue
            current_qty = state.holdings.get(ts_code, 0.0)
            desired_qty = current_qty
            if decision.action is AgentAction.SELL:
                desired_qty = 0.0
            elif decision.action is AgentAction.HOLD:
                desired_qty = current_qty
            else:
                target_weight = max(decision.target_weight, 0.0)
                desired_value = target_weight * portfolio_value_before
                if desired_value > 0:
                    desired_qty = desired_value / price
                else:
                    desired_qty = current_qty

            delta = desired_qty - current_qty
            if abs(delta) < 1e-6:
                continue

            if delta > 0:
                cost = delta * price
                if cost > state.cash:
                    affordable_qty = state.cash / price if price > 0 else 0.0
                    delta = max(0.0, affordable_qty)
                    cost = delta * price
                    desired_qty = current_qty + delta
                if delta <= 0:
                    continue
                total_cost = state.cost_basis.get(ts_code, 0.0) * current_qty + cost
                new_qty = current_qty + delta
                state.cost_basis[ts_code] = total_cost / new_qty if new_qty > 0 else 0.0
                state.cash -= cost
                state.holdings[ts_code] = new_qty
                state.opened_dates.setdefault(ts_code, trade_date_str)
                trades_records.append(
                    {
                        "trade_date": trade_date_str,
                        "ts_code": ts_code,
                        "action": "buy",
                        "quantity": float(delta),
                        "price": price,
                        "value": cost,
                        "confidence": decision.confidence,
                        "target_weight": decision.target_weight,
                    }
                )
            else:
                sell_qty = abs(delta)
                if sell_qty > current_qty:
                    sell_qty = current_qty
                    delta = -sell_qty
                proceeds = sell_qty * price
                cost_basis = state.cost_basis.get(ts_code, 0.0)
                realized = (price - cost_basis) * sell_qty
                state.cash += proceeds
                state.realized_pnl += realized
                new_qty = current_qty + delta
                if new_qty <= 1e-6:
                    state.holdings.pop(ts_code, None)
                    state.cost_basis.pop(ts_code, None)
                    state.opened_dates.pop(ts_code, None)
                else:
                    state.holdings[ts_code] = new_qty
                trades_records.append(
                    {
                        "trade_date": trade_date_str,
                        "ts_code": ts_code,
                        "action": "sell",
                        "quantity": float(sell_qty),
                        "price": price,
                        "value": proceeds,
                        "confidence": decision.confidence,
                        "target_weight": decision.target_weight,
                        "realized_pnl": realized,
                    }
                )

        market_value = 0.0
        unrealized_pnl = 0.0
        for ts_code, qty in state.holdings.items():
            price = price_map.get(ts_code)
            if price is None:
                continue
            market_value += qty * price
            cost_basis = state.cost_basis.get(ts_code, 0.0)
            unrealized_pnl += (price - cost_basis) * qty

        nav = state.cash + market_value
        result.nav_series.append(
            {
                "trade_date": trade_date_str,
                "nav": nav,
                "cash": state.cash,
                "market_value": market_value,
                "realized_pnl": state.realized_pnl,
                "unrealized_pnl": unrealized_pnl,
            }
        )
        if trades_records:
            result.trades.extend(trades_records)

        try:
            self._persist_portfolio(
                trade_date_str,
                state,
                market_value,
                unrealized_pnl,
                trades_records,
                price_map,
                decisions_map,
            )
        except Exception:  # noqa: BLE001
            LOGGER.exception("持仓数据写入失败", extra=LOG_EXTRA)

    def _record_investment_candidate(
        self, context: AgentContext, decision: Decision
    ) -> None:
        status = _candidate_status(decision.action, decision.requires_review)
        summary = _extract_summary(decision)
        if not summary:
            collected_signals: List[str] = []
            for dept in decision.department_decisions.values():
                collected_signals.extend(dept.signals)
            summary = "；".join(str(sig) for sig in collected_signals[:3])

        metadata = {
            "target_weight": decision.target_weight,
            "feasible_actions": [action.value for action in decision.feasible_actions],
            "department_votes": decision.department_votes,
            "requires_review": decision.requires_review,
            "confidence": decision.confidence,
        }
        if decision.department_decisions:
            metadata["departments"] = {
                code: dept.to_dict()
                for code, dept in decision.department_decisions.items()
            }

        with db_session() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO investment_pool
                (trade_date, ts_code, score, status, rationale, tags, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    context.trade_date,
                    context.ts_code,
                    float(decision.confidence or 0.0),
                    status,
                    summary or None,
                    json.dumps(_department_tags(decision), ensure_ascii=False),
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )

    def _persist_portfolio(
        self,
        trade_date: str,
        state: PortfolioState,
        market_value: float,
        unrealized_pnl: float,
        trades: List[Dict[str, Any]],
        price_map: Dict[str, float],
        decisions_map: Dict[str, Decision],
    ) -> None:
        holdings_rows: List[tuple] = []
        for ts_code, qty in state.holdings.items():
            price = price_map.get(ts_code)
            market_val = qty * price if price is not None else None
            cost_basis = state.cost_basis.get(ts_code, 0.0)
            unrealized = (price - cost_basis) * qty if price is not None else None
            decision = decisions_map.get(ts_code)
            target_weight = decision.target_weight if decision else None
            metadata = {
                "last_action": decision.action.value if decision else None,
                "confidence": decision.confidence if decision else None,
            }
            holdings_rows.append(
                (
                    ts_code,
                    state.opened_dates.get(ts_code, trade_date),
                    None,
                    qty,
                    cost_basis,
                    price,
                    market_val,
                    state.realized_pnl,
                    unrealized,
                    target_weight,
                    "open",
                    None,
                    json.dumps(metadata, ensure_ascii=False),
                )
            )

        snapshot_metadata = {
            "holdings": len(state.holdings),
        }

        with db_session() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO portfolio_snapshots
                (trade_date, total_value, cash, invested_value, unrealized_pnl, realized_pnl, net_flow, exposure, notes, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trade_date,
                    market_value + state.cash,
                    state.cash,
                    market_value,
                    unrealized_pnl,
                    state.realized_pnl,
                    None,
                    None,
                    None,
                    json.dumps(snapshot_metadata, ensure_ascii=False),
                ),
            )

            conn.execute("DELETE FROM portfolio_positions")
            if holdings_rows:
                conn.executemany(
                    """
                    INSERT INTO portfolio_positions
                    (ts_code, opened_date, closed_date, quantity, cost_price, market_price, market_value, realized_pnl, unrealized_pnl, target_weight, status, notes, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    holdings_rows,
                )

            if trades:
                conn.executemany(
                    """
                    INSERT INTO portfolio_trades
                    (trade_date, ts_code, action, quantity, price, fee, order_id, source, notes, metadata)
                    VALUES (?, ?, ?, ?, ?, 0, NULL, 'backtest', NULL, ?)
                    """,
                    [
                        (
                            trade["trade_date"],
                            trade["ts_code"],
                            trade["action"],
                            trade["quantity"],
                            trade["price"],
                            json.dumps(trade, ensure_ascii=False),
                        )
                        for trade in trades
                    ],
                )

    def run(
        self,
        decision_callback: Optional[Callable[[str, date, AgentContext, Decision], None]] = None,
    ) -> BacktestResult:
        state = PortfolioState()
        result = BacktestResult()
        current = self.cfg.start_date
        while current <= self.cfg.end_date:
            records = self.simulate_day(current, state, decision_callback)
            self._apply_portfolio_updates(current, state, records, result)
            current = date.fromordinal(current.toordinal() + 1)
        return result


def run_backtest(
    cfg: BtConfig,
    *,
    decision_callback: Optional[Callable[[str, date, AgentContext, Decision], None]] = None,
) -> BacktestResult:
    engine = BacktestEngine(cfg)
    result = engine.run(decision_callback=decision_callback)
    with db_session() as conn:
        _ = conn
        # Implementation should persist bt_nav, bt_trades, and bt_report rows.
    return result


def _candidate_status(action: AgentAction, requires_review: bool) -> str:
    mapping = {
        AgentAction.SELL: "exit",
        AgentAction.HOLD: "watch",
        AgentAction.BUY_S: "buy_s",
        AgentAction.BUY_M: "buy_m",
        AgentAction.BUY_L: "buy_l",
    }
    base = mapping.get(action, "candidate")
    if requires_review:
        return f"{base}_review"
    return base
def _extract_summary(decision: Decision) -> str:
    for dept_decision in decision.department_decisions.values():
        summary = getattr(dept_decision, "summary", "")
        if summary:
            return str(summary)
    return ""


def _department_tags(decision: Decision) -> List[str]:
    tags: List[str] = []
    for code, dept in decision.department_decisions.items():
        action = getattr(dept, "action", None)
        if action is None:
            continue
        tags.append(f"{code}:{action.value}")
    return sorted(set(tags))
