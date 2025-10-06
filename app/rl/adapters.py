"""Environment adapters bridging DecisionEnv to tensor-friendly interfaces."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from app.backtest.decision_env import DecisionEnv


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

    @property
    def action_dim(self) -> int:
        return self.env.action_dim

    @property
    def observation_dim(self) -> int:
        return len(self._keys)

    def reset(self) -> Tuple[np.ndarray, Dict[str, float]]:
        raw = self.env.reset()
        self._last_reset_obs = raw
        return self._to_array(raw), raw

    def step(
        self, action: Sequence[float]
    ) -> Tuple[np.ndarray, float, bool, Mapping[str, object], Mapping[str, float]]:
        obs_dict, reward, done, info = self.env.step(action)
        return self._to_array(obs_dict), reward, done, info, obs_dict

    def _to_array(self, payload: Mapping[str, float]) -> np.ndarray:
        buffer = np.zeros(len(self._keys), dtype=np.float32)
        for idx, key in enumerate(self._keys):
            value = payload.get(key)
            buffer[idx] = float(value) if value is not None else 0.0
        return buffer

    def keys(self) -> List[str]:
        return list(self._keys)
