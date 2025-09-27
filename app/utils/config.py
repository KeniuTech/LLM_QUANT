"""Application configuration models and helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2] / "app" / "data"


@dataclass
class DataPaths:
    """Holds filesystem locations for persistent artifacts."""

    root: Path = field(default_factory=_default_root)
    database: Path = field(init=False)
    backups: Path = field(init=False)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "llm_quant.db"
        self.backups = self.root / "backups"
        self.backups.mkdir(parents=True, exist_ok=True)


@dataclass
class AgentWeights:
    """Default weighting for decision agents."""

    momentum: float = 0.30
    value: float = 0.20
    news: float = 0.20
    liquidity: float = 0.15
    macro: float = 0.15

    def as_dict(self) -> Dict[str, float]:
        return {
            "A_mom": self.momentum,
            "A_val": self.value,
            "A_news": self.news,
            "A_liq": self.liquidity,
            "A_macro": self.macro,
        }


@dataclass
class AppConfig:
    """User configurable settings persisted in a simple structure."""

    tushare_token: Optional[str] = None
    rss_sources: Dict[str, bool] = field(default_factory=dict)
    decision_method: str = "nash"
    data_paths: DataPaths = field(default_factory=DataPaths)
    agent_weights: AgentWeights = field(default_factory=AgentWeights)
    force_refresh: bool = False
    max_calls_per_minute: int = 180


CONFIG = AppConfig()


def get_config() -> AppConfig:
    """Return a mutable global configuration instance."""

    return CONFIG
