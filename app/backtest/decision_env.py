"""Reinforcement-learning style environment wrapping the backtest engine."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import math

from .engine import BacktestEngine, BacktestResult, BtConfig
from app.agents.game import Decision
from app.agents.registry import weight_map
from app.utils.logging import get_logger

LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "decision_env"}


@dataclass(frozen=True)
class ParameterSpec:
    """Defines how a scalar action dimension maps to strategy parameters."""

    name: str
    target: str
    minimum: float = 0.0
    maximum: float = 1.0

    def clamp(self, value: float) -> float:
        clipped = max(0.0, min(1.0, float(value)))
        return self.minimum + clipped * (self.maximum - self.minimum)


@dataclass
class EpisodeMetrics:
    total_return: float
    max_drawdown: float
    volatility: float
    nav_series: List[Dict[str, float]]
    trades: List[Dict[str, object]]
    turnover: float
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
        self._episode = 0
        self._disable_departments = bool(disable_departments)

    @property
    def action_dim(self) -> int:
        return len(self._specs)

    def reset(self) -> Dict[str, float]:
        self._episode += 1
        self._last_metrics = None
        self._last_action = None
        return {
            "episode": float(self._episode),
            "baseline_return": 0.0,
        }

    def step(self, action: Sequence[float]) -> Tuple[Dict[str, float], float, bool, Dict[str, object]]:
        if len(action) != self.action_dim:
            raise ValueError(f"expected action length {self.action_dim}, got {len(action)}")
        action_array = [float(val) for val in action]
        self._last_action = tuple(action_array)

        weights = self._build_weights(action_array)
        LOGGER.info("episode=%s action=%s weights=%s", self._episode, action_array, weights, extra=LOG_EXTRA)

        cfg = replace(self._template_cfg)
        engine = BacktestEngine(cfg)
        engine.weights = weight_map(weights)
        if self._disable_departments:
            engine.department_manager = None

        try:
            result = engine.run()
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception("backtest failed under action", extra={**LOG_EXTRA, "error": str(exc)})
            info = {"error": str(exc)}
            return {"failure": 1.0}, -1.0, True, info

        metrics = self._compute_metrics(result)
        reward = float(self._reward_fn(metrics))
        self._last_metrics = metrics

        observation = {
            "total_return": metrics.total_return,
            "max_drawdown": metrics.max_drawdown,
            "volatility": metrics.volatility,
            "sharpe_like": metrics.sharpe_like,
            "turnover": metrics.turnover,
            "trade_count": float(metrics.trade_count),
            "risk_count": float(metrics.risk_count),
        }
        info = {
            "nav_series": metrics.nav_series,
            "trades": metrics.trades,
            "weights": weights,
            "risk_breakdown": metrics.risk_breakdown,
            "risk_events": getattr(result, "risk_events", []),
        }
        return observation, reward, True, info

    def _build_weights(self, action: Sequence[float]) -> Dict[str, float]:
        weights = dict(self._baseline_weights)
        for idx, spec in enumerate(self._specs):
            value = spec.clamp(action[idx])
            if spec.target.startswith("agent_weights."):
                agent_name = spec.target.split(".", 1)[1]
                weights[agent_name] = value
            else:
                LOGGER.debug("暂未支持的参数目标：%s", spec.target, extra=LOG_EXTRA)
        return weights

    def _compute_metrics(self, result: BacktestResult) -> EpisodeMetrics:
        nav_series = result.nav_series or []
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
                trades=result.trades,
                turnover=0.0,
                trade_count=len(result.trades or []),
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

        turnover = sum(float(row.get("turnover", 0.0) or 0.0) for row in nav_series)
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
            trades=result.trades,
            turnover=float(turnover),
            trade_count=len(result.trades or []),
            risk_count=len(risk_events),
            risk_breakdown=risk_breakdown,
        )

    @staticmethod
    def _default_reward(metrics: EpisodeMetrics) -> float:
        risk_penalty = 0.05 * metrics.risk_count
        turnover_penalty = 0.00001 * metrics.turnover
        penalty = 0.5 * metrics.max_drawdown + risk_penalty + turnover_penalty
        return metrics.total_return - penalty

    @property
    def last_metrics(self) -> Optional[EpisodeMetrics]:
        return self._last_metrics

    @property
    def last_action(self) -> Optional[Tuple[float, ...]]:
        return self._last_action
