"""Scope mappings for different game structures."""
from __future__ import annotations

from typing import Dict, Iterable, Set

from .protocols import GameStructure


_GAME_SCOPE_MAP: Dict[GameStructure, Set[str]] = {
    GameStructure.REPEATED: {
        "daily.close",
        "daily.open",
        "daily.high",
        "daily.low",
        "daily_basic.turnover_rate",
        "daily_basic.turnover_rate_f",
    },
    GameStructure.SIGNALING: {
        "daily.close",
        "daily.high",
        "daily_basic.turnover_rate",
        "daily_basic.volume_ratio",
    },
    GameStructure.BAYESIAN: {
        "daily.close",
        "daily_basic.turnover_rate",
        "factors.mom_20",
        "factors.mom_60",
        "factors.val_multiscore",
    },
    GameStructure.CUSTOM: {
        "factors.risk_penalty",
        "factors.turn_20",
        "factors.volat_20",
        "daily_basic.turnover_rate",
    },
}


def scope_for_structures(structures: Iterable[GameStructure]) -> Set[str]:
    scope: Set[str] = set()
    for structure in structures:
        scope.update(_GAME_SCOPE_MAP.get(structure, set()))
    return scope


def registered_structures() -> Dict[GameStructure, Set[str]]:
    return {key: set(values) for key, values in _GAME_SCOPE_MAP.items()}
