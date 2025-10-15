"""Optimization utilities for DecisionEnv-based parameter tuning."""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from app.backtest.decision_env import DecisionEnv, EpisodeMetrics
from app.backtest.decision_env import ParameterSpec
from app.utils.logging import get_logger
from app.utils.tuning import log_tuning_result

LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "decision_bandit"}


@dataclass
class BanditConfig:
    """Configuration shared by all global parameter search strategies."""

    experiment_id: str
    strategy: str = "epsilon_greedy"
    episodes: int = 20
    epsilon: float = 0.2
    seed: int | None = None
    exploration_weight: float = 0.01
    candidate_pool: int = 128
    initial_candidates: int = 27
    eta: int = 3
    max_rounds: int = 3


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


class _BaseOptimizer:
    """Shared helpers for global parameter search algorithms."""

    def __init__(self, env: DecisionEnv, config: BanditConfig) -> None:
        self.env = env
        self.config = config
        self._specs: List[ParameterSpec] = list(getattr(env, "_specs", []))
        if not self._specs:
            raise ValueError("DecisionEnv does not expose parameter specs")
        self._history = BanditSummary()
        self._random = random.Random(config.seed)

    def _evaluate_action(self, action: Sequence[float]) -> Tuple[float, EpisodeMetrics, Dict[str, float], Dict[str, Any]]:
        self.env.reset()
        cumulative_reward = 0.0
        obs: Dict[str, float] = {}
        info: Dict[str, Any] = {}
        done = False
        while not done:
            obs, reward, done, info = self.env.step(action)
            cumulative_reward += reward
        metrics = self.env.last_metrics
        if metrics is None:
            raise RuntimeError("DecisionEnv did not populate last_metrics")
        return cumulative_reward, metrics, obs, info

    def _record_episode(
        self,
        action: Sequence[float],
        reward: float,
        metrics: EpisodeMetrics,
        obs: Dict[str, float],
        info: Dict[str, Any],
    ) -> None:
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
                reward=reward,
                metrics=metrics_payload,
                weights=info.get("weights"),
            )
        except Exception:  # noqa: BLE001
            LOGGER.exception("failed to log tuning result", extra=LOG_EXTRA)

        episode_record = BanditEpisode(
            action=action_payload,
            resolved_action=resolved_action,
            reward=reward,
            metrics=metrics,
            observation=obs,
            weights=info.get("weights"),
            department_controls=department_controls,
        )
        self._history.episodes.append(episode_record)

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

    def _sample_random_action(self) -> List[float]:
        values: List[float] = []
        for spec in self._specs:
            if spec.values:
                if len(spec.values) <= 1:
                    values.append(0.0)
                else:
                    index = self._random.randrange(len(spec.values))
                    values.append(index / (len(spec.values) - 1))
            else:
                values.append(self._random.random())
        return values

    def _mutate_action(self, action: Sequence[float], scale: float = 0.1) -> List[float]:
        mutated = []
        for value in action:
            jitter = self._random.gauss(0.0, scale)
            mutated.append(min(1.0, max(0.0, float(value + jitter))))
        return mutated


class EpsilonGreedyBandit(_BaseOptimizer):
    """Epsilon-greedy tuner using DecisionEnv as the reward oracle."""

    def __init__(self, env: DecisionEnv, config: BanditConfig) -> None:
        super().__init__(env, config)
        self._value_estimates: Dict[Tuple[float, ...], float] = {}
        self._counts: Dict[Tuple[float, ...], int] = {}

    def run(self) -> BanditSummary:
        for episode in range(1, self.config.episodes + 1):
            action = self._select_action()
            reward, metrics, obs, info = self._evaluate_action(action)
            key = tuple(action)
            old_estimate = self._value_estimates.get(key, 0.0)
            count = self._counts.get(key, 0) + 1
            self._counts[key] = count
            self._value_estimates[key] = old_estimate + (reward - old_estimate) / count

            self._record_episode(action, reward, metrics, obs, info)
            LOGGER.info(
                "Bandit episode=%s reward=%.4f action=%s",
                episode,
                reward,
                self._raw_action_mapping(action),
                extra=LOG_EXTRA,
            )
        return self._history

    def _select_action(self) -> List[float]:
        if self._value_estimates and self._random.random() > self.config.epsilon:
            best = max(self._value_estimates.items(), key=lambda item: item[1])[0]
            return list(best)
        return self._sample_random_action()


class BayesianBandit(_BaseOptimizer):
    """Gaussian-process based Bayesian optimization."""

    def __init__(self, env: DecisionEnv, config: BanditConfig) -> None:
        super().__init__(env, config)
        self._X: List[np.ndarray] = []
        self._y: List[float] = []
        self._noise = 1e-6
        self._length_scale = 0.3

    def run(self) -> BanditSummary:
        for _ in range(self.config.episodes):
            action = self._propose_action()
            reward, metrics, obs, info = self._evaluate_action(action)
            self._record_episode(action, reward, metrics, obs, info)
            self._X.append(np.array(action, dtype=float))
            self._y.append(reward)
        return self._history

    def _propose_action(self) -> List[float]:
        if not self._X:
            return self._sample_random_action()

        X = np.vstack(self._X)
        y = np.asarray(self._y, dtype=float)
        K = self._kernel(X, X) + self._noise * np.eye(len(X))
        try:
            K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            K_inv = np.linalg.pinv(K)

        best_y = max(y)
        candidates = [self._sample_random_action() for _ in range(self.config.candidate_pool)]
        ei_values: List[Tuple[float, List[float]]] = []
        for candidate in candidates:
            x = np.asarray(candidate, dtype=float)
            k_star = self._kernel(X, x[None, :])[:, 0]
            mean = float(k_star @ K_inv @ y)
            k_ss = float(self._kernel(x[None, :], x[None, :])[0, 0])
            variance = max(k_ss - k_star @ K_inv @ k_star, 1e-9)
            std = math.sqrt(variance)
            improvement = mean - best_y - self.config.exploration_weight
            z = improvement / std if std > 0 else 0.0
            cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
            pdf = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * z * z)
            ei = improvement * cdf + std * pdf if std > 0 else max(improvement, 0.0)
            ei_values.append((ei, candidate))

        ei_values.sort(key=lambda item: item[0], reverse=True)
        best = ei_values[0][1] if ei_values else self._sample_random_action()
        return best

    def _kernel(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        sq_dist = np.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=2)
        return np.exp(-0.5 * sq_dist / (self._length_scale ** 2))


class SuccessiveHalvingOptimizer(_BaseOptimizer):
    """Simplified BOHB-style successive halving optimizer."""

    def run(self) -> BanditSummary:
        num_candidates = max(1, self.config.initial_candidates)
        eta = max(2, self.config.eta)
        actions = [self._sample_random_action() for _ in range(num_candidates)]

        for round_idx in range(self.config.max_rounds):
            if not actions:
                break
            evaluations: List[Tuple[float, List[float]]] = []
            for action in actions:
                reward, metrics, obs, info = self._evaluate_action(action)
                self._record_episode(action, reward, metrics, obs, info)
                evaluations.append((reward, action))
            evaluations.sort(key=lambda item: item[0], reverse=True)
            survivors = max(1, len(evaluations) // eta)
            actions = [action for _, action in evaluations[:survivors]]
            if len(actions) == 1:
                break
            actions = [self._mutate_action(action, scale=0.05 * (round_idx + 1)) for action in actions]
        return self._history


def _metrics_to_dict(metrics: EpisodeMetrics) -> Dict[str, float | Dict[str, int]]:
    payload: Dict[str, float | Dict[str, int]] = {
        "total_return": metrics.total_return,
        "max_drawdown": metrics.max_drawdown,
        "volatility": metrics.volatility,
        "sharpe_like": metrics.sharpe_like,
        "calmar_like": metrics.calmar_like,
        "turnover": metrics.turnover,
        "turnover_value": metrics.turnover_value,
        "trade_count": float(metrics.trade_count),
        "risk_count": float(metrics.risk_count),
    }
    if metrics.risk_breakdown:
        payload["risk_breakdown"] = dict(metrics.risk_breakdown)
    return payload
