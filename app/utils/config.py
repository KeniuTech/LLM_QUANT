"""Application configuration models and helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2] / "app" / "data"


@dataclass
class DataPaths:
    """Holds filesystem locations for persistent artifacts."""

    root: Path = field(default_factory=_default_root)
    database: Path = field(init=False)
    backups: Path = field(init=False)
    config_file: Path = field(init=False)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "llm_quant.db"
        self.backups = self.root / "backups"
        self.backups.mkdir(parents=True, exist_ok=True)
        self.config_file = self.root / "config.json"


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

DEFAULT_LLM_MODEL_OPTIONS: Dict[str, Dict[str, object]] = {
    "ollama": {
        "models": ["llama3", "phi3", "qwen2"],
        "base_url": "http://localhost:11434",
        "temperature": 0.2,
        "timeout": 30.0,
    },
    "openai": {
        "models": ["gpt-4o-mini", "gpt-4.1-mini", "gpt-3.5-turbo"],
        "base_url": "https://api.openai.com",
        "temperature": 0.2,
        "timeout": 30.0,
    },
    "deepseek": {
        "models": ["deepseek-chat", "deepseek-coder"],
        "base_url": "https://api.deepseek.com",
        "temperature": 0.2,
        "timeout": 45.0,
    },
    "wenxin": {
        "models": ["ERNIE-Speed", "ERNIE-Bot"],
        "base_url": "https://aip.baidubce.com",
        "temperature": 0.2,
        "timeout": 60.0,
    },
}

DEFAULT_LLM_MODELS: Dict[str, str] = {
    provider: info["models"][0]
    for provider, info in DEFAULT_LLM_MODEL_OPTIONS.items()
}

DEFAULT_LLM_BASE_URLS: Dict[str, str] = {
    provider: info["base_url"]
    for provider, info in DEFAULT_LLM_MODEL_OPTIONS.items()
}

DEFAULT_LLM_TEMPERATURES: Dict[str, float] = {
    provider: float(info.get("temperature", 0.2))
    for provider, info in DEFAULT_LLM_MODEL_OPTIONS.items()
}

DEFAULT_LLM_TIMEOUTS: Dict[str, float] = {
    provider: float(info.get("timeout", 30.0))
    for provider, info in DEFAULT_LLM_MODEL_OPTIONS.items()
}

ALLOWED_LLM_STRATEGIES = {"single", "majority", "leader"}
LLM_STRATEGY_ALIASES = {"leader-follower": "leader"}


@dataclass
class LLMEndpoint:
    """Single LLM endpoint configuration."""

    provider: str = "ollama"
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.2
    timeout: float = 30.0

    def __post_init__(self) -> None:
        self.provider = (self.provider or "ollama").lower()
        if not self.model:
            self.model = DEFAULT_LLM_MODELS.get(self.provider, DEFAULT_LLM_MODELS["ollama"])
        if not self.base_url:
            self.base_url = DEFAULT_LLM_BASE_URLS.get(self.provider)
        if self.temperature == 0.2 or self.temperature is None:
            self.temperature = DEFAULT_LLM_TEMPERATURES.get(self.provider, 0.2)
        if self.timeout == 30.0 or self.timeout is None:
            self.timeout = DEFAULT_LLM_TIMEOUTS.get(self.provider, 30.0)


@dataclass
class LLMConfig:
    """LLM configuration allowing single or ensemble strategies."""

    primary: LLMEndpoint = field(default_factory=LLMEndpoint)
    ensemble: List[LLMEndpoint] = field(default_factory=list)
    strategy: str = "single"  # Options: single, majority, leader
    majority_threshold: int = 3


@dataclass
class DepartmentSettings:
    """Configuration for a single decision department."""

    code: str
    title: str
    description: str = ""
    weight: float = 1.0
    llm: LLMConfig = field(default_factory=LLMConfig)


def _default_departments() -> Dict[str, DepartmentSettings]:
    presets = [
        ("momentum", "动量策略部门"),
        ("value", "价值评估部门"),
        ("news", "新闻情绪部门"),
        ("liquidity", "流动性评估部门"),
        ("macro", "宏观研究部门"),
        ("risk", "风险控制部门"),
    ]
    return {
        code: DepartmentSettings(code=code, title=title)
        for code, title in presets
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
    llm: LLMConfig = field(default_factory=LLMConfig)
    departments: Dict[str, DepartmentSettings] = field(default_factory=_default_departments)


CONFIG = AppConfig()


def _endpoint_to_dict(endpoint: LLMEndpoint) -> Dict[str, object]:
    return {
        "provider": endpoint.provider,
        "model": endpoint.model,
        "base_url": endpoint.base_url,
        "api_key": endpoint.api_key,
        "temperature": endpoint.temperature,
        "timeout": endpoint.timeout,
    }


def _dict_to_endpoint(data: Dict[str, object]) -> LLMEndpoint:
    payload = {
        key: data.get(key)
        for key in ("provider", "model", "base_url", "api_key", "temperature", "timeout")
        if data.get(key) is not None
    }
    return LLMEndpoint(**payload)


def _load_from_file(cfg: AppConfig) -> None:
    path = cfg.data_paths.config_file
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return

    if isinstance(payload, dict):
        if "tushare_token" in payload:
            cfg.tushare_token = payload.get("tushare_token") or None
        if "force_refresh" in payload:
            cfg.force_refresh = bool(payload.get("force_refresh"))
        if "decision_method" in payload:
            cfg.decision_method = str(payload.get("decision_method") or cfg.decision_method)

        llm_payload = payload.get("llm")
        if isinstance(llm_payload, dict):
            primary_data = llm_payload.get("primary")
            if isinstance(primary_data, dict):
                cfg.llm.primary = _dict_to_endpoint(primary_data)

            ensemble_data = llm_payload.get("ensemble")
            if isinstance(ensemble_data, list):
                cfg.llm.ensemble = [
                    _dict_to_endpoint(item)
                    for item in ensemble_data
                    if isinstance(item, dict)
                ]

            strategy_raw = llm_payload.get("strategy")
            if isinstance(strategy_raw, str):
                normalized = LLM_STRATEGY_ALIASES.get(strategy_raw, strategy_raw)
                if normalized in ALLOWED_LLM_STRATEGIES:
                    cfg.llm.strategy = normalized

            majority = llm_payload.get("majority_threshold")
            if isinstance(majority, int) and majority > 0:
                cfg.llm.majority_threshold = majority

        departments_payload = payload.get("departments")
        if isinstance(departments_payload, dict):
            new_departments: Dict[str, DepartmentSettings] = {}
            for code, data in departments_payload.items():
                if not isinstance(data, dict):
                    continue
                title = data.get("title") or code
                description = data.get("description") or ""
                weight = float(data.get("weight", 1.0))
                llm_data = data.get("llm")
                llm_cfg = LLMConfig()
                if isinstance(llm_data, dict):
                    if isinstance(llm_data.get("primary"), dict):
                        llm_cfg.primary = _dict_to_endpoint(llm_data["primary"])
                    llm_cfg.ensemble = [
                        _dict_to_endpoint(item)
                        for item in llm_data.get("ensemble", [])
                        if isinstance(item, dict)
                    ]
                    strategy_raw = llm_data.get("strategy")
                    if isinstance(strategy_raw, str):
                        normalized = LLM_STRATEGY_ALIASES.get(strategy_raw, strategy_raw)
                        if normalized in ALLOWED_LLM_STRATEGIES:
                            llm_cfg.strategy = normalized
                    majority_raw = llm_data.get("majority_threshold")
                    if isinstance(majority_raw, int) and majority_raw > 0:
                        llm_cfg.majority_threshold = majority_raw
                new_departments[code] = DepartmentSettings(
                    code=code,
                    title=title,
                    description=description,
                    weight=weight,
                    llm=llm_cfg,
                )
            if new_departments:
                cfg.departments = new_departments


def save_config(cfg: AppConfig | None = None) -> None:
    cfg = cfg or CONFIG
    path = cfg.data_paths.config_file
    payload = {
        "tushare_token": cfg.tushare_token,
        "force_refresh": cfg.force_refresh,
        "decision_method": cfg.decision_method,
        "llm": {
            "strategy": cfg.llm.strategy if cfg.llm.strategy in ALLOWED_LLM_STRATEGIES else "single",
            "majority_threshold": cfg.llm.majority_threshold,
            "primary": _endpoint_to_dict(cfg.llm.primary),
            "ensemble": [_endpoint_to_dict(ep) for ep in cfg.llm.ensemble],
        },
        "departments": {
            code: {
                "title": dept.title,
                "description": dept.description,
                "weight": dept.weight,
                "llm": {
                    "strategy": dept.llm.strategy if dept.llm.strategy in ALLOWED_LLM_STRATEGIES else "single",
                    "majority_threshold": dept.llm.majority_threshold,
                    "primary": _endpoint_to_dict(dept.llm.primary),
                    "ensemble": [_endpoint_to_dict(ep) for ep in dept.llm.ensemble],
                },
            }
            for code, dept in cfg.departments.items()
        },
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
    except OSError:
        pass


def _load_env_defaults(cfg: AppConfig) -> None:
    """Populate sensitive fields from environment variables if present."""

    token = os.getenv("TUSHARE_TOKEN")
    if token:
        cfg.tushare_token = token.strip()

    api_key = os.getenv("LLM_API_KEY")
    if api_key:
        cfg.llm.primary.api_key = api_key.strip()


_load_from_file(CONFIG)
_load_env_defaults(CONFIG)


def get_config() -> AppConfig:
    """Return a mutable global configuration instance."""

    return CONFIG
