"""Environment adapters bridging DecisionEnv to tensor-friendly interfaces."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from app.backtest.decision_env import DecisionEnv
from app.utils.logging import get_logger

LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "decision_env"}


@dataclass
class DecisionEnvAdapter:
    """Wraps :class:`DecisionEnv` to emit numpy arrays for RL algorithms."""

    env: DecisionEnv
    observation_keys: Sequence[str] | None = None

    def __post_init__(self) -> None:
        if self.observation_keys is None:
            reset_obs = self.env.reset()
            # Exclude bookkeeping fields not useful for learning policy values
            exclude = {"episode"}
            self._keys = [key for key in sorted(reset_obs.keys()) if key not in exclude]
            self._last_reset_obs = reset_obs
        else:
            self._keys = list(self.observation_keys)
            self._last_reset_obs = None
        LOGGER.debug(
            "初始化 DecisionEnvAdapter obs_dim=%s action_dim=%s keys=%s",
            len(self._keys),
            self.env.action_dim,
            self._keys,
            extra=LOG_EXTRA,
        )

    @property
    def action_dim(self) -> int:
        return self.env.action_dim

    @property
    def observation_dim(self) -> int:
        return len(self._keys)

    def reset(self) -> Tuple[np.ndarray, Dict[str, float]]:
        raw = self.env.reset()
        self._last_reset_obs = raw
        LOGGER.debug(
            "环境重置完成 episode=%s",
            raw.get("episode"),
            extra=LOG_EXTRA,
        )
        return self._to_array(raw), raw

    def step(
        self, action: Sequence[float]
    ) -> Tuple[np.ndarray, float, bool, Mapping[str, object], Mapping[str, float]]:
        obs_dict, reward, done, info = self.env.step(action)
        LOGGER.debug(
            "环境执行动作 action=%s reward=%.4f done=%s",
            [round(float(a), 4) for a in action],
            reward,
            done,
            extra=LOG_EXTRA,
        )
        return self._to_array(obs_dict), reward, done, info, obs_dict

    def _to_array(self, payload: Mapping[str, float]) -> np.ndarray:
        buffer = np.zeros(len(self._keys), dtype=np.float32)
        for idx, key in enumerate(self._keys):
            value = payload.get(key)
            buffer[idx] = float(value) if value is not None else 0.0
        return buffer

    def keys(self) -> List[str]:
        return list(self._keys)
