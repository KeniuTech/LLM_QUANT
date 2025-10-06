"""Optimization utilities for DecisionEnv-based parameter tuning."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from app.backtest.decision_env import DecisionEnv, EpisodeMetrics
from app.backtest.decision_env import ParameterSpec
from app.utils.logging import get_logger
from app.utils.tuning import log_tuning_result

LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "decision_bandit"}


@dataclass
class BanditConfig:
    """Configuration for epsilon-greedy bandit optimization."""

    experiment_id: str
    strategy: str = "epsilon_greedy"
    episodes: int = 20
    epsilon: float = 0.2
    seed: int | None = None


@dataclass
class BanditEpisode:
    action: Dict[str, float]
    resolved_action: Dict[str, Any]
    reward: float
    metrics: EpisodeMetrics
    observation: Dict[str, float]
    weights: Mapping[str, float] | None = None
    department_controls: Mapping[str, Mapping[str, Any]] | None = None


@dataclass
class BanditSummary:
    episodes: List[BanditEpisode] = field(default_factory=list)

    @property
    def best_episode(self) -> BanditEpisode | None:
        if not self.episodes:
            return None
        return max(self.episodes, key=lambda item: item.reward)

    @property
    def average_reward(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(item.reward for item in self.episodes) / len(self.episodes)


class EpsilonGreedyBandit:
    """Simple epsilon-greedy tuner using DecisionEnv as the reward oracle."""

    def __init__(self, env: DecisionEnv, config: BanditConfig) -> None:
        self.env = env
        self.config = config
        self._random = random.Random(config.seed)
        self._specs: List[ParameterSpec] = list(getattr(env, "_specs", []))
        if not self._specs:
            raise ValueError("DecisionEnv does not expose parameter specs")
        self._value_estimates: Dict[Tuple[float, ...], float] = {}
        self._counts: Dict[Tuple[float, ...], int] = {}
        self._history = BanditSummary()

    def run(self) -> BanditSummary:
        for episode in range(1, self.config.episodes + 1):
            action = self._select_action()
            self.env.reset()
            done = False
            cumulative_reward = 0.0
            obs = {}
            info: Dict[str, Any] = {}
            while not done:
                obs, reward, done, info = self.env.step(action)
                cumulative_reward += reward

            metrics = self.env.last_metrics
            if metrics is None:
                raise RuntimeError("DecisionEnv did not populate last_metrics")
            key = tuple(action)
            old_estimate = self._value_estimates.get(key, 0.0)
            count = self._counts.get(key, 0) + 1
            self._counts[key] = count
            self._value_estimates[key] = old_estimate + (cumulative_reward - old_estimate) / count

            action_payload = self._raw_action_mapping(action)
            resolved_action = self._resolved_action_mapping(action)
            metrics_payload = _metrics_to_dict(metrics)
            department_controls = info.get("department_controls")
            if department_controls:
                metrics_payload["department_controls"] = department_controls
            metrics_payload["resolved_action"] = resolved_action
            try:
                log_tuning_result(
                    experiment_id=self.config.experiment_id,
                    strategy=self.config.strategy,
                    action=action_payload,
                    reward=cumulative_reward,
                    metrics=metrics_payload,
                    weights=info.get("weights"),
                )
            except Exception:  # noqa: BLE001
                LOGGER.exception("failed to log tuning result", extra=LOG_EXTRA)

            episode_record = BanditEpisode(
                action=action_payload,
                resolved_action=resolved_action,
                reward=cumulative_reward,
                metrics=metrics,
                observation=obs,
                weights=info.get("weights"),
                department_controls=department_controls,
            )
            self._history.episodes.append(episode_record)
            LOGGER.info(
                "Bandit episode=%s reward=%.4f action=%s",
                episode,
                cumulative_reward,
                action_payload,
                extra=LOG_EXTRA,
            )
        return self._history

    def _select_action(self) -> List[float]:
        if self._value_estimates and self._random.random() > self.config.epsilon:
            best = max(self._value_estimates.items(), key=lambda item: item[1])[0]
            return list(best)
        return [self._sample_value(spec) for spec in self._specs]

    def _raw_action_mapping(self, action: Sequence[float]) -> Dict[str, float]:
        return {
            spec.name: float(value)
            for spec, value in zip(self._specs, action, strict=True)
        }

    def _resolved_action_mapping(self, action: Sequence[float]) -> Dict[str, Any]:
        return {
            spec.name: spec.resolve(value)
            for spec, value in zip(self._specs, action, strict=True)
        }

    def _sample_value(self, spec: ParameterSpec) -> float:
        if spec.values:
            if len(spec.values) <= 1:
                return 0.0
            index = self._random.randrange(len(spec.values))
            return index / (len(spec.values) - 1)
        return self._random.random()


def _metrics_to_dict(metrics: EpisodeMetrics) -> Dict[str, float | Dict[str, int]]:
    payload: Dict[str, float | Dict[str, int]] = {
        "total_return": metrics.total_return,
        "max_drawdown": metrics.max_drawdown,
        "volatility": metrics.volatility,
        "sharpe_like": metrics.sharpe_like,
        "turnover": metrics.turnover,
        "turnover_value": metrics.turnover_value,
        "trade_count": float(metrics.trade_count),
        "risk_count": float(metrics.risk_count),
    }
    if metrics.risk_breakdown:
        payload["risk_breakdown"] = dict(metrics.risk_breakdown)
    return payload
