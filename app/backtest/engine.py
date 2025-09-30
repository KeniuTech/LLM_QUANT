"""Backtest engine skeleton for daily bar simulation."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
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
from app.core.indicators import momentum, normalize, rolling_mean, volatility


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
    cost_basis: Dict[str, float] = field(default_factory=dict)
    opened_dates: Dict[str, str] = field(default_factory=dict)
    realized_pnl: float = 0.0


@dataclass
class BacktestResult:
    nav_series: List[Dict[str, float]] = field(default_factory=list)
    trades: List[Dict[str, str]] = field(default_factory=list)
    risk_events: List[Dict[str, object]] = field(default_factory=list)


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
        params = cfg.params or {}
        self.risk_params = {
            "max_position_weight": float(params.get("max_position_weight", 0.2)),
            "max_daily_turnover_ratio": float(params.get("max_daily_turnover_ratio", 0.25)),
            "fee_rate": float(params.get("fee_rate", 0.0005)),
            "slippage_bps": float(params.get("slippage_bps", 10.0)),
        }
        self._fee_rate = max(self.risk_params["fee_rate"], 0.0)
        self._slippage_rate = max(self.risk_params["slippage_bps"], 0.0) / 10_000.0
        self._turnover_cap = max(self.risk_params["max_daily_turnover_ratio"], 0.0)
        self._buy_actions = {
            AgentAction.BUY_S,
            AgentAction.BUY_M,
            AgentAction.BUY_L,
        }
        self._sell_actions = {AgentAction.SELL}
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
            "factors.mom_20",
            "factors.mom_60",
            "factors.volat_20",
            "factors.turn_20",
            "news.sentiment_index",
            "news.heat_score",
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
            close_values = [value for _date, value in closes if value is not None]

            mom20 = scope_values.get("factors.mom_20")
            if mom20 is None and len(close_values) >= 20:
                mom20 = momentum(close_values, 20)

            mom60 = scope_values.get("factors.mom_60")
            if mom60 is None and len(close_values) >= 60:
                mom60 = momentum(close_values, 60)

            volat20 = scope_values.get("factors.volat_20")
            if volat20 is None and len(close_values) >= 2:
                volat20 = volatility(close_values, 20)

            turnover_series = self.data_broker.fetch_series(
                "daily_basic",
                "turnover_rate",
                ts_code,
                trade_date_str,
                window=20,
            )
            turnover_values = [value for _date, value in turnover_series if value is not None]

            turn20 = scope_values.get("factors.turn_20")
            if turn20 is None and turnover_values:
                turn20 = rolling_mean(turnover_values, 20)

            if mom20 is None:
                mom20 = 0.0
            if mom60 is None:
                mom60 = 0.0
            if volat20 is None:
                volat20 = 0.0
            if turn20 is None:
                turn20 = 0.0

            liquidity_score = normalize(turn20, factor=20.0)
            cost_penalty = normalize(
                scope_values.get("daily_basic.volume_ratio", 0.0),
                factor=50.0,
            )

            sentiment_index = scope_values.get("news.sentiment_index", 0.0)
            heat_score = scope_values.get("news.heat_score", 0.0)
            scope_values.setdefault("news.sentiment_index", sentiment_index)
            scope_values.setdefault("news.heat_score", heat_score)

            scope_values.setdefault("factors.mom_20", mom20)
            scope_values.setdefault("factors.mom_60", mom60)
            scope_values.setdefault("factors.volat_20", volat20)
            scope_values.setdefault("factors.turn_20", turn20)
            if scope_values.get("macro.industry_heat") is None:
                scope_values["macro.industry_heat"] = 0.5
            if scope_values.get("macro.relative_strength") is None:
                peer_strength = scope_values.get("index.performance_peers")
                if peer_strength is None:
                    peer_strength = 0.5
                scope_values["macro.relative_strength"] = peer_strength
            scope_values.setdefault(
                "index.performance_peers",
                scope_values.get("macro.relative_strength", 0.5),
            )

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
                "news_heat": heat_score,
                "news_sentiment": sentiment_index,
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
        feature_cache: Dict[str, Mapping[str, Any]] = {}
        for ts_code, context, decision in records:
            features = context.features or {}
            if not isinstance(features, Mapping):
                features = {}
            feature_cache[ts_code] = features
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
            for ts_code in list(state.holdings.keys()):
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

        daily_turnover = 0.0
        executed_trades: List[Dict[str, Any]] = []
        risk_events: List[Dict[str, Any]] = []

        def _record_risk(ts_code: str, reason: str, decision: Decision, extra: Optional[Dict[str, Any]] = None) -> None:
            payload = {
                "trade_date": trade_date_str,
                "ts_code": ts_code,
                "action": decision.action.value,
                "target_weight": decision.target_weight,
                "confidence": decision.confidence,
                "reason": reason,
            }
            if extra:
                payload.update(extra)
            risk_events.append(payload)

        for ts_code, decision in decisions_map.items():
            price = price_map.get(ts_code)
            if price is None or price <= 0:
                continue
            features = feature_cache.get(ts_code, {})
            current_qty = state.holdings.get(ts_code, 0.0)
            liquidity_score = float(features.get("liquidity_score") or 0.0)
            risk_penalty = float(features.get("risk_penalty") or 0.0)
            is_suspended = bool(features.get("is_suspended"))
            limit_up = bool(features.get("limit_up"))
            limit_down = bool(features.get("limit_down"))
            position_limit = bool(features.get("position_limit"))

            if is_suspended:
                _record_risk(ts_code, "suspended", decision)
                continue
            if decision.action in self._buy_actions:
                if limit_up:
                    _record_risk(ts_code, "limit_up", decision)
                    continue
                if position_limit:
                    _record_risk(ts_code, "position_limit", decision)
                    continue
                if risk_penalty >= 0.95:
                    _record_risk(ts_code, "risk_penalty", decision, {"risk_penalty": risk_penalty})
                    continue
            if decision.action in self._sell_actions and limit_down:
                _record_risk(ts_code, "limit_down", decision)
                continue

            effective_weight = max(decision.target_weight, 0.0)
            if decision.action in self._buy_actions:
                capped_weight = min(effective_weight, self.risk_params["max_position_weight"])
                effective_weight = capped_weight * max(0.0, 1.0 - risk_penalty)
            elif decision.action in self._sell_actions:
                effective_weight = 0.0

            desired_qty = current_qty
            if decision.action in self._sell_actions:
                desired_qty = 0.0
            elif decision.action in self._buy_actions or effective_weight >= 0.0:
                desired_value = max(effective_weight, 0.0) * portfolio_value_before
                desired_qty = desired_value / price if price > 0 else current_qty

            delta = desired_qty - current_qty
            if abs(delta) < 1e-6:
                continue

            if delta > 0 and self._turnover_cap > 0:
                liquidity_scalar = max(liquidity_score, 0.1)
                max_trade_value = self._turnover_cap * portfolio_value_before * liquidity_scalar
                if max_trade_value > 0 and delta * price > max_trade_value:
                    delta = max_trade_value / price
                    desired_qty = current_qty + delta

            if delta > 0:
                trade_price = price * (1.0 + self._slippage_rate)
                per_share_cost = trade_price * (1.0 + self._fee_rate)
                if per_share_cost <= 0:
                    _record_risk(ts_code, "invalid_price", decision)
                    continue
                max_affordable = state.cash / per_share_cost if per_share_cost > 0 else 0.0
                if delta > max_affordable:
                    if max_affordable <= 1e-6:
                        _record_risk(ts_code, "insufficient_cash", decision)
                        continue
                    delta = max_affordable
                    desired_qty = current_qty + delta

                trade_value = delta * trade_price
                fee = trade_value * self._fee_rate
                total_cash_needed = trade_value + fee
                if total_cash_needed <= 0:
                    _record_risk(ts_code, "invalid_trade", decision)
                    continue

                previous_cost = state.cost_basis.get(ts_code, 0.0) * current_qty
                new_qty = current_qty + delta
                state.cost_basis[ts_code] = (
                    (previous_cost + trade_value + fee) / new_qty if new_qty > 0 else 0.0
                )
                state.cash -= total_cash_needed
                state.holdings[ts_code] = new_qty
                state.opened_dates.setdefault(ts_code, trade_date_str)
                daily_turnover += trade_value
                executed_trades.append(
                    {
                        "trade_date": trade_date_str,
                        "ts_code": ts_code,
                        "action": "buy",
                        "quantity": float(delta),
                        "price": trade_price,
                        "base_price": price,
                        "value": trade_value,
                        "fee": fee,
                        "slippage": trade_price - price,
                        "confidence": decision.confidence,
                        "target_weight": decision.target_weight,
                        "effective_weight": effective_weight,
                        "risk_penalty": risk_penalty,
                        "liquidity_score": liquidity_score,
                        "status": "executed",
                    }
                )
            else:
                sell_qty = min(abs(delta), current_qty)
                if sell_qty <= 1e-6:
                    continue
                trade_price = price * (1.0 - self._slippage_rate)
                trade_price = max(trade_price, 0.0)
                gross_value = sell_qty * trade_price
                fee = gross_value * self._fee_rate
                proceeds = gross_value - fee
                cost_basis = state.cost_basis.get(ts_code, 0.0)
                realized = (trade_price - cost_basis) * sell_qty - fee
                state.cash += proceeds
                state.realized_pnl += realized
                new_qty = current_qty - sell_qty
                if new_qty <= 1e-6:
                    state.holdings.pop(ts_code, None)
                    state.cost_basis.pop(ts_code, None)
                    state.opened_dates.pop(ts_code, None)
                else:
                    state.holdings[ts_code] = new_qty
                daily_turnover += gross_value
                executed_trades.append(
                    {
                        "trade_date": trade_date_str,
                        "ts_code": ts_code,
                        "action": "sell",
                        "quantity": float(sell_qty),
                        "price": trade_price,
                        "base_price": price,
                        "value": gross_value,
                        "fee": fee,
                        "slippage": price - trade_price,
                        "confidence": decision.confidence,
                        "target_weight": decision.target_weight,
                        "effective_weight": effective_weight,
                        "risk_penalty": risk_penalty,
                        "liquidity_score": liquidity_score,
                        "realized_pnl": realized,
                        "status": "executed",
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
                "turnover": daily_turnover,
            }
        )
        if executed_trades:
            result.trades.extend(executed_trades)
        if risk_events:
            result.risk_events.extend(risk_events)

        try:
            self._persist_portfolio(
                trade_date_str,
                state,
                market_value,
                unrealized_pnl,
                executed_trades,
                price_map,
                decisions_map,
                daily_turnover,
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
        daily_turnover: float,
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
            "turnover_value": daily_turnover,
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
                    VALUES (?, ?, ?, ?, ?, ?, NULL, 'backtest', NULL, ?)
                    """,
                    [
                        (
                            trade["trade_date"],
                            trade["ts_code"],
                            trade["action"],
                            trade["quantity"],
                            trade["price"],
                            trade.get("fee", 0.0),
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
    _persist_backtest_results(cfg, result)
    return result


def _persist_backtest_results(cfg: BtConfig, result: BacktestResult) -> None:
    """Persist backtest configuration, NAV path, trades and summary metrics."""

    nav_rows: List[tuple] = []
    trade_rows: List[tuple] = []
    risk_rows: List[tuple] = []
    summary_payload: Dict[str, object] = {}
    turnover_sum = 0.0

    if result.nav_series:
        first_nav = float(result.nav_series[0].get("nav", 0.0) or 0.0)
        peak_nav = first_nav
        prev_nav: Optional[float] = None
        max_drawdown = 0.0
        for entry in result.nav_series:
            trade_date = str(entry.get("trade_date", ""))
            nav_val = float(entry.get("nav", 0.0) or 0.0)
            cash = float(entry.get("cash", 0.0) or 0.0)
            market_value = float(entry.get("market_value", 0.0) or 0.0)
            realized = float(entry.get("realized_pnl", 0.0) or 0.0)
            unrealized = float(entry.get("unrealized_pnl", 0.0) or 0.0)
            turnover = float(entry.get("turnover", 0.0) or 0.0)

            if nav_val > peak_nav:
                peak_nav = nav_val
            drawdown = (peak_nav - nav_val) / peak_nav if peak_nav else 0.0
            max_drawdown = max(max_drawdown, drawdown)

            if prev_nav is None or prev_nav == 0.0:
                ret_val = 0.0
            else:
                ret_val = (nav_val / prev_nav) - 1.0
            prev_nav = nav_val

            info_payload = {
                "cash": cash,
                "market_value": market_value,
                "realized_pnl": realized,
                "unrealized_pnl": unrealized,
                "turnover": turnover,
            }
            turnover_sum += turnover
            nav_rows.append(
                (
                    cfg.id,
                    trade_date,
                    nav_val,
                    float(ret_val),
                    None,
                    None,
                    float(drawdown),
                    json.dumps(info_payload, ensure_ascii=False),
                )
            )

        last_nav = float(result.nav_series[-1].get("nav", 0.0) or 0.0)
        total_return = (last_nav / first_nav - 1.0) if first_nav else 0.0
        summary_payload.update(
            {
                "start_nav": first_nav,
                "end_nav": last_nav,
                "total_return": total_return,
                "max_drawdown": max_drawdown,
                "days": len(result.nav_series),
            }
        )
        if turnover_sum:
            summary_payload["total_turnover"] = turnover_sum
            summary_payload["avg_turnover"] = turnover_sum / max(len(result.nav_series), 1)

    if result.trades:
        for trade in result.trades:
            trade_date = str(trade.get("trade_date", ""))
            ts_code = str(trade.get("ts_code", ""))
            side = str(trade.get("action", "")).lower()
            price = float(trade.get("price", 0.0) or 0.0)
            qty = float(trade.get("quantity", 0.0) or 0.0)
            reason_payload = {
                "confidence": trade.get("confidence"),
                "target_weight": trade.get("target_weight"),
                "value": trade.get("value"),
                "fee": trade.get("fee"),
                "slippage": trade.get("slippage"),
                "risk_penalty": trade.get("risk_penalty"),
                "liquidity_score": trade.get("liquidity_score"),
            }
            trade_rows.append(
                (
                    cfg.id,
                    ts_code,
                    trade_date,
                    side,
                    price,
                    qty,
                    json.dumps(reason_payload, ensure_ascii=False),
                )
            )
        summary_payload["trade_count"] = len(trade_rows)

    if result.risk_events:
        summary_payload["risk_events"] = len(result.risk_events)
        breakdown: Dict[str, int] = {}
        for event in result.risk_events:
            reason = str(event.get("reason") or "unknown")
            breakdown[reason] = breakdown.get(reason, 0) + 1
            risk_rows.append(
                (
                    cfg.id,
                    str(event.get("trade_date", "")),
                    str(event.get("ts_code", "")),
                    reason,
                    str(event.get("action", "")),
                    float(event.get("target_weight", 0.0) or 0.0),
                    float(event.get("confidence", 0.0) or 0.0),
                    json.dumps(event, ensure_ascii=False),
                )
            )
        summary_payload["risk_breakdown"] = breakdown

    cfg_payload = {
        "id": cfg.id,
        "name": cfg.name,
        "start_date": cfg.start_date.isoformat(),
        "end_date": cfg.end_date.isoformat(),
        "universe": cfg.universe,
        "params": cfg.params,
        "method": cfg.method,
    }

    with db_session() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO bt_config (id, name, start_date, end_date, universe, params)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                cfg.id,
                cfg.name,
                cfg.start_date.isoformat(),
                cfg.end_date.isoformat(),
                ",".join(cfg.universe),
                json.dumps(cfg.params, ensure_ascii=False),
            ),
        )

        conn.execute("DELETE FROM bt_nav WHERE cfg_id = ?", (cfg.id,))
        conn.execute("DELETE FROM bt_trades WHERE cfg_id = ?", (cfg.id,))
        conn.execute("DELETE FROM bt_risk_events WHERE cfg_id = ?", (cfg.id,))
        conn.execute("DELETE FROM bt_report WHERE cfg_id = ?", (cfg.id,))

        if nav_rows:
            conn.executemany(
                """
                INSERT INTO bt_nav (cfg_id, trade_date, nav, ret, pos_count, turnover, dd, info)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                nav_rows,
            )

        if trade_rows:
            conn.executemany(
                """
                INSERT INTO bt_trades (cfg_id, ts_code, trade_date, side, price, qty, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                trade_rows,
            )

        if risk_rows:
            conn.executemany(
                """
                INSERT INTO bt_risk_events (cfg_id, trade_date, ts_code, reason, action, target_weight, confidence, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                risk_rows,
            )

        summary_payload.setdefault("universe", cfg.universe)
        summary_payload.setdefault("method", cfg.method)
        conn.execute(
            """
            INSERT OR REPLACE INTO bt_report (cfg_id, summary)
            VALUES (?, ?)
            """,
            (cfg.id, json.dumps(summary_payload, ensure_ascii=False, default=str)),
        )


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
