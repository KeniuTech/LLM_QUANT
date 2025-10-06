"""Reinforcement-learning style environment wrapping the backtest engine."""
from __future__ import annotations

import json
import math
import copy
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .engine import BacktestEngine, BacktestResult, BtConfig
from app.agents.game import Decision
from app.agents.registry import weight_map
from app.utils.db import db_session
from app.utils.logging import get_logger

LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "decision_env"}


@dataclass(frozen=True)
class ParameterSpec:
    """Defines how an action dimension maps to strategy parameters or behaviors."""

    name: str
    target: str
    minimum: float = 0.0
    maximum: float = 1.0
    values: Optional[Sequence[Any]] = None

    def clamp(self, value: float) -> float:
        clipped = max(0.0, min(1.0, float(value)))
        return self.minimum + clipped * (self.maximum - self.minimum)

    def resolve(self, value: float) -> Any:
        if self.values is not None:
            if not self.values:
                raise ValueError(f"ParameterSpec {self.name} configured with empty values list")
            clipped = max(0.0, min(1.0, float(value)))
            index = int(round(clipped * (len(self.values) - 1)))
            return self.values[index]
        return self.clamp(value)


@dataclass
class EpisodeMetrics:
    total_return: float
    max_drawdown: float
    volatility: float
    nav_series: List[Dict[str, float]]
    trades: List[Dict[str, object]]
    turnover: float
    turnover_value: float
    trade_count: int
    risk_count: int
    risk_breakdown: Dict[str, int]

    @property
    def sharpe_like(self) -> float:
        if self.volatility <= 1e-9:
            return 0.0
        return self.total_return / self.volatility


class DecisionEnv:
    """Thin RL-friendly wrapper that evaluates parameter actions via backtest."""

    def __init__(
        self,
        *,
        bt_config: BtConfig,
        parameter_specs: Sequence[ParameterSpec],
        baseline_weights: Mapping[str, float],
        reward_fn: Optional[Callable[[EpisodeMetrics], float]] = None,
        disable_departments: bool = False,
    ) -> None:
        self._template_cfg = bt_config
        self._specs = list(parameter_specs)
        self._baseline_weights = dict(baseline_weights)
        self._reward_fn = reward_fn or self._default_reward
        self._last_metrics: Optional[EpisodeMetrics] = None
        self._last_action: Optional[Tuple[float, ...]] = None
        self._last_department_controls: Optional[Dict[str, Dict[str, Any]]] = None
        self._episode = 0
        self._disable_departments = bool(disable_departments)

    @property
    def action_dim(self) -> int:
        return len(self._specs)

    @property
    def last_department_controls(self) -> Optional[Dict[str, Dict[str, Any]]]:
        return self._last_department_controls

    def reset(self) -> Dict[str, float]:
        self._episode += 1
        self._last_metrics = None
        self._last_action = None
        self._last_department_controls = None
        return {
            "episode": float(self._episode),
            "baseline_return": 0.0,
        }

    def step(self, action: Sequence[float]) -> Tuple[Dict[str, float], float, bool, Dict[str, object]]:
        if len(action) != self.action_dim:
            raise ValueError(f"expected action length {self.action_dim}, got {len(action)}")
        action_array = [float(val) for val in action]
        self._last_action = tuple(action_array)

        weights, department_controls = self._prepare_actions(action_array)
        LOGGER.info(
            "episode=%s action=%s weights=%s controls=%s",
            self._episode,
            action_array,
            weights,
            department_controls,
            extra=LOG_EXTRA,
        )

        cfg = replace(self._template_cfg)
        engine = BacktestEngine(cfg)
        engine.weights = weight_map(weights)
        if self._disable_departments:
            engine.department_manager = None
            applied_controls: Dict[str, Dict[str, Any]] = {}
        else:
            applied_controls = self._apply_department_controls(engine, department_controls)

        self._clear_portfolio_records()

        try:
            result = engine.run()
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("backtest failed under action", extra={**LOG_EXTRA, "error": str(exc)})
            info = {"error": str(exc)}
            return {"failure": 1.0}, -1.0, True, info

        snapshots, trades_override = self._fetch_portfolio_records()
        metrics = self._compute_metrics(
            result,
            nav_override=snapshots if snapshots else None,
            trades_override=trades_override if trades_override else None,
        )
        reward = float(self._reward_fn(metrics))
        self._last_metrics = metrics

        observation = {
            "total_return": metrics.total_return,
            "max_drawdown": metrics.max_drawdown,
            "volatility": metrics.volatility,
            "sharpe_like": metrics.sharpe_like,
            "turnover": metrics.turnover,
            "turnover_value": metrics.turnover_value,
            "trade_count": float(metrics.trade_count),
            "risk_count": float(metrics.risk_count),
        }
        info = {
            "nav_series": metrics.nav_series,
            "trades": metrics.trades,
            "weights": weights,
            "risk_breakdown": metrics.risk_breakdown,
            "risk_events": getattr(result, "risk_events", []),
            "portfolio_snapshots": snapshots,
            "portfolio_trades": trades_override,
            "department_controls": applied_controls,
        }
        self._last_department_controls = applied_controls
        return observation, reward, True, info

    def _prepare_actions(
        self,
        action: Sequence[float],
    ) -> Tuple[Dict[str, float], Dict[str, Dict[str, Any]]]:
        weights = dict(self._baseline_weights)
        department_controls: Dict[str, Dict[str, Any]] = {}
        for idx, spec in enumerate(self._specs):
            try:
                resolved = spec.resolve(action[idx])
            except ValueError as exc:
                LOGGER.warning("参数 %s 解析失败：%s", spec.name, exc, extra=LOG_EXTRA)
                continue
            if spec.target.startswith("agent_weights."):
                agent_name = spec.target.split(".", 1)[1]
                try:
                    weights[agent_name] = float(resolved)
                except (TypeError, ValueError):
                    LOGGER.debug(
                        "spec %s produced non-numeric weight %s; skipping",
                        spec.name,
                        resolved,
                        extra=LOG_EXTRA,
                    )
                continue
            if spec.target.startswith("department."):
                target_path = spec.target.split(".")[1:]
                if len(target_path) < 2:
                    LOGGER.debug("未识别的部门目标：%s", spec.target, extra=LOG_EXTRA)
                    continue
                dept_code = target_path[0]
                field = ".".join(target_path[1:])
                dept_controls = department_controls.setdefault(dept_code, {})
                dept_controls[field] = resolved
                continue
            else:
                LOGGER.debug("暂未支持的参数目标：%s", spec.target, extra=LOG_EXTRA)
        return weights, department_controls

    def _apply_department_controls(
        self,
        engine: BacktestEngine,
        controls: Mapping[str, Mapping[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        manager = getattr(engine, "department_manager", None)
        if not manager or not getattr(manager, "agents", None):
            return {}

        applied: Dict[str, Dict[str, Any]] = {}
        for dept_code, payload in controls.items():
            agent = manager.agents.get(dept_code)
            if not agent or not isinstance(payload, Mapping):
                continue

            applied_fields: Dict[str, Any] = {}

            # Ensure mutable settings clone to avoid global side-effects
            try:
                original_settings = agent.settings
                cloned_settings = replace(original_settings)
                cloned_settings.llm = copy.deepcopy(original_settings.llm)
                agent.settings = cloned_settings
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning(
                    "复制部门 %s 配置失败：%s",
                    dept_code,
                    exc,
                    extra=LOG_EXTRA,
                )
                continue

            for raw_field, value in payload.items():
                field = raw_field.lower()
                if field == "function_policy":
                    field = "tool_choice"
                if field in {"prompt", "instruction"}:
                    agent.settings.prompt = str(value)
                    applied_fields[field] = agent.settings.prompt
                    continue
                if field == "description":
                    agent.settings.description = str(value)
                    applied_fields[field] = agent.settings.description
                    continue
                if field in {"prompt_template_id", "prompt_template"}:
                    agent.settings.prompt_template_id = str(value)
                    applied_fields["prompt_template_id"] = agent.settings.prompt_template_id
                    continue
                if field == "prompt_template_version":
                    agent.settings.prompt_template_version = str(value)
                    applied_fields["prompt_template_version"] = agent.settings.prompt_template_version
                    continue
                if field in {"temperature", "llm.temperature"}:
                    try:
                        temperature = max(0.0, min(2.0, float(value)))
                        agent.settings.llm.primary.temperature = temperature
                        applied_fields["temperature"] = temperature
                    except (TypeError, ValueError):
                        LOGGER.debug(
                            "无效的温度值 %s for %s",
                            value,
                            dept_code,
                            extra=LOG_EXTRA,
                        )
                    continue
                if field in {"tool_choice", "tool_strategy"}:
                    try:
                        agent.tool_choice = value
                        applied_fields["tool_choice"] = agent.tool_choice
                    except ValueError:
                        LOGGER.debug(
                            "部门 %s 工具策略 %s 无效",
                            dept_code,
                            value,
                            extra=LOG_EXTRA,
                        )
                    continue
                if field == "max_rounds":
                    try:
                        agent.max_rounds = value
                        applied_fields["max_rounds"] = agent.max_rounds
                    except ValueError:
                        LOGGER.debug(
                            "部门 %s max_rounds %s 无效",
                            dept_code,
                            value,
                            extra=LOG_EXTRA,
                        )
                    continue
                if field == "prompt_template_override":
                    agent.settings.prompt = str(value)
                    applied_fields["prompt"] = agent.settings.prompt
                    continue
                LOGGER.debug(
                    "部门 %s 未识别的控制项 %s",
                    dept_code,
                    raw_field,
                    extra=LOG_EXTRA,
                )

            if applied_fields:
                applied[dept_code] = applied_fields

        return applied

    def _compute_metrics(
        self,
        result: BacktestResult,
        *,
        nav_override: Optional[List[Dict[str, Any]]] = None,
        trades_override: Optional[List[Dict[str, Any]]] = None,
    ) -> EpisodeMetrics:
        nav_series = nav_override if nav_override is not None else result.nav_series or []
        trades = trades_override if trades_override is not None else result.trades

        if not nav_series:
            risk_breakdown: Dict[str, int] = {}
            for event in getattr(result, "risk_events", []) or []:
                reason = str(event.get("reason") or "unknown")
                risk_breakdown[reason] = risk_breakdown.get(reason, 0) + 1
            return EpisodeMetrics(
                total_return=0.0,
                max_drawdown=0.0,
                volatility=0.0,
                nav_series=[],
                trades=trades or [],
                turnover=0.0,
                turnover_value=0.0,
                trade_count=len(trades or []),
                risk_count=len(getattr(result, "risk_events", []) or []),
                risk_breakdown=risk_breakdown,
            )

        nav_values = [row.get("nav", 0.0) for row in nav_series]
        if not nav_values or nav_values[0] == 0:
            base_nav = nav_values[0] if nav_values else 1.0
        else:
            base_nav = nav_values[0]

        returns = [(nav / base_nav) - 1.0 for nav in nav_values]
        total_return = returns[-1]

        peak = nav_values[0]
        max_drawdown = 0.0
        for nav in nav_values:
            if nav > peak:
                peak = nav
            drawdown = (peak - nav) / peak if peak else 0.0
            max_drawdown = max(max_drawdown, drawdown)

        diffs = [nav_values[idx] - nav_values[idx - 1] for idx in range(1, len(nav_values))]
        if diffs:
            mean_diff = sum(diffs) / len(diffs)
            variance = sum((diff - mean_diff) ** 2 for diff in diffs) / len(diffs)
            volatility = math.sqrt(variance) / base_nav
        else:
            volatility = 0.0

        turnover_value = 0.0
        turnover_ratios: List[float] = []
        for row in nav_series:
            turnover_raw = float(row.get("turnover", 0.0) or 0.0)
            turnover_value += turnover_raw
            ratio = row.get("turnover_ratio")
            if ratio is not None:
                try:
                    turnover_ratios.append(float(ratio))
                    continue
                except (TypeError, ValueError):
                    turnover_ratios.append(0.0)
                    continue
            nav_val = float(row.get("nav", 0.0) or 0.0)
            if nav_val > 0:
                turnover_ratios.append(turnover_raw / nav_val)
            else:
                turnover_ratios.append(0.0)
        avg_turnover_ratio = sum(turnover_ratios) / len(turnover_ratios) if turnover_ratios else 0.0
        risk_events = getattr(result, "risk_events", []) or []
        risk_breakdown: Dict[str, int] = {}
        for event in risk_events:
            reason = str(event.get("reason") or "unknown")
            risk_breakdown[reason] = risk_breakdown.get(reason, 0) + 1

        return EpisodeMetrics(
            total_return=float(total_return),
            max_drawdown=float(max_drawdown),
            volatility=volatility,
            nav_series=nav_series,
            trades=trades or [],
            turnover=float(avg_turnover_ratio),
            turnover_value=float(turnover_value),
            trade_count=len(trades or []),
            risk_count=len(risk_events),
            risk_breakdown=risk_breakdown,
        )

    @staticmethod
    def _default_reward(metrics: EpisodeMetrics) -> float:
        risk_penalty = 0.05 * metrics.risk_count
        turnover_penalty = 0.1 * metrics.turnover
        penalty = 0.5 * metrics.max_drawdown + risk_penalty + turnover_penalty
        return metrics.total_return - penalty

    @property
    def last_metrics(self) -> Optional[EpisodeMetrics]:
        return self._last_metrics

    @property
    def last_action(self) -> Optional[Tuple[float, ...]]:
        return self._last_action

    def _clear_portfolio_records(self) -> None:
        start = self._template_cfg.start_date.isoformat()
        end = self._template_cfg.end_date.isoformat()
        try:
            with db_session() as conn:
                conn.execute("DELETE FROM portfolio_positions")
                conn.execute(
                    "DELETE FROM portfolio_snapshots WHERE trade_date BETWEEN ? AND ?",
                    (start, end),
                )
                conn.execute(
                    "DELETE FROM portfolio_trades WHERE trade_date BETWEEN ? AND ?",
                    (start, end),
                )
        except Exception:  # noqa: BLE001
            LOGGER.exception("清理投资组合记录失败", extra=LOG_EXTRA)

    def _fetch_portfolio_records(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        start = self._template_cfg.start_date.isoformat()
        end = self._template_cfg.end_date.isoformat()
        snapshots: List[Dict[str, Any]] = []
        trades: List[Dict[str, Any]] = []
        try:
            with db_session(read_only=True) as conn:
                snapshot_rows = conn.execute(
                    """
                    SELECT trade_date, total_value, cash, invested_value,
                           unrealized_pnl, realized_pnl, net_flow, exposure, metadata
                    FROM portfolio_snapshots
                    WHERE trade_date BETWEEN ? AND ?
                    ORDER BY trade_date
                    """,
                    (start, end),
                ).fetchall()
                trade_rows = conn.execute(
                    """
                    SELECT id, trade_date, ts_code, action, quantity, price, fee, source, metadata
                    FROM portfolio_trades
                    WHERE trade_date BETWEEN ? AND ?
                    ORDER BY trade_date, id
                    """,
                    (start, end),
                ).fetchall()
        except Exception:  # noqa: BLE001
            LOGGER.exception("读取投资组合记录失败", extra=LOG_EXTRA)
            return snapshots, trades

        for row in snapshot_rows:
            metadata = self._loads(row["metadata"], {})
            snapshots.append(
                {
                    "trade_date": row["trade_date"],
                    "nav": float(row["total_value"] or 0.0),
                    "cash": float(row["cash"] or 0.0),
                    "market_value": float(row["invested_value"] or 0.0),
                    "unrealized_pnl": float(row["unrealized_pnl"] or 0.0),
                    "realized_pnl": float(row["realized_pnl"] or 0.0),
                    "net_flow": float(row["net_flow"] or 0.0),
                    "exposure": float(row["exposure"] or 0.0),
                    "turnover": float(metadata.get("turnover_value", 0.0) or 0.0),
                    "turnover_ratio": float(metadata.get("turnover_ratio", 0.0) or 0.0),
                    "holdings": metadata.get("holdings"),
                    "trade_count": metadata.get("trade_count"),
                }
            )

        for row in trade_rows:
            metadata = self._loads(row["metadata"], {})
            trades.append(
                {
                    "id": row["id"],
                    "trade_date": row["trade_date"],
                    "ts_code": row["ts_code"],
                    "action": row["action"],
                    "quantity": float(row["quantity"] or 0.0),
                    "price": float(row["price"] or 0.0),
                    "fee": float(row["fee"] or 0.0),
                    "source": row["source"],
                    "metadata": metadata,
                }
            )

        return snapshots, trades

    @staticmethod
    def _loads(payload: Any, default: Any) -> Any:
        if not payload:
            return default
        if isinstance(payload, (dict, list)):
            return payload
        if isinstance(payload, str):
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return default
        return default
