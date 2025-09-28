"""Application configuration models and helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Dict, List, Mapping, Optional


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
class LLMProfile:
    """Named LLM endpoint profile reusable across routes/departments."""

    key: str
    provider: str = "ollama"
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: float = 0.2
    timeout: float = 30.0
    title: str = ""
    enabled: bool = True

    def to_endpoint(self) -> LLMEndpoint:
        return LLMEndpoint(
            provider=self.provider,
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            temperature=self.temperature,
            timeout=self.timeout,
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "temperature": self.temperature,
            "timeout": self.timeout,
            "title": self.title,
            "enabled": self.enabled,
        }

    @classmethod
    def from_endpoint(
        cls,
        key: str,
        endpoint: LLMEndpoint,
        *,
        title: str = "",
        enabled: bool = True,
    ) -> "LLMProfile":
        return cls(
            key=key,
            provider=endpoint.provider,
            model=endpoint.model,
            base_url=endpoint.base_url,
            api_key=endpoint.api_key,
            temperature=endpoint.temperature,
            timeout=endpoint.timeout,
            title=title,
            enabled=enabled,
        )


@dataclass
class LLMRoute:
    """Declarative routing for selecting profiles and strategy."""

    name: str
    title: str = ""
    strategy: str = "single"
    majority_threshold: int = 3
    primary: str = "ollama"
    ensemble: List[str] = field(default_factory=list)

    def resolve(self, profiles: Mapping[str, LLMProfile]) -> LLMConfig:
        def _endpoint_from_key(key: str) -> LLMEndpoint:
            profile = profiles.get(key)
            if profile and profile.enabled:
                return profile.to_endpoint()
            fallback = profiles.get("ollama")
            if not fallback or not fallback.enabled:
                fallback = next(
                    (item for item in profiles.values() if item.enabled),
                    None,
                )
            endpoint = fallback.to_endpoint() if fallback else LLMEndpoint()
            endpoint.provider = key or endpoint.provider
            return endpoint

        primary_endpoint = _endpoint_from_key(self.primary)
        ensemble_endpoints = [
            _endpoint_from_key(key)
            for key in self.ensemble
            if key in profiles and profiles[key].enabled
        ]
        config = LLMConfig(
            primary=primary_endpoint,
            ensemble=ensemble_endpoints,
            strategy=self.strategy if self.strategy in ALLOWED_LLM_STRATEGIES else "single",
            majority_threshold=max(1, self.majority_threshold or 1),
        )
        return config

    def to_dict(self) -> Dict[str, object]:
        return {
            "title": self.title,
            "strategy": self.strategy,
            "majority_threshold": self.majority_threshold,
            "primary": self.primary,
            "ensemble": list(self.ensemble),
        }


def _default_llm_profiles() -> Dict[str, LLMProfile]:
    return {
        provider: LLMProfile(
            key=provider,
            provider=provider,
            model=DEFAULT_LLM_MODELS.get(provider),
            base_url=DEFAULT_LLM_BASE_URLS.get(provider),
            temperature=DEFAULT_LLM_TEMPERATURES.get(provider, 0.2),
            timeout=DEFAULT_LLM_TIMEOUTS.get(provider, 30.0),
            title=f"默认 {provider}",
        )
        for provider in DEFAULT_LLM_MODEL_OPTIONS
    }


def _default_llm_routes() -> Dict[str, LLMRoute]:
    return {
        "global": LLMRoute(name="global", title="全局默认路由"),
    }


@dataclass
class DepartmentSettings:
    """Configuration for a single decision department."""

    code: str
    title: str
    description: str = ""
    weight: float = 1.0
    llm: LLMConfig = field(default_factory=LLMConfig)
    llm_route: Optional[str] = None


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
        code: DepartmentSettings(code=code, title=title, llm_route="global")
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
    llm_route: str = "global"
    llm_profiles: Dict[str, LLMProfile] = field(default_factory=_default_llm_profiles)
    llm_routes: Dict[str, LLMRoute] = field(default_factory=_default_llm_routes)
    departments: Dict[str, DepartmentSettings] = field(default_factory=_default_departments)

    def resolve_llm(self, route: Optional[str] = None) -> LLMConfig:
        route_key = route or self.llm_route
        route_cfg = self.llm_routes.get(route_key)
        if route_cfg:
            return route_cfg.resolve(self.llm_profiles)
        return self.llm

    def sync_runtime_llm(self) -> None:
        self.llm = self.resolve_llm()


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

        routes_defined = False
        inline_primary_loaded = False

        profiles_payload = payload.get("llm_profiles")
        if isinstance(profiles_payload, dict):
            profiles: Dict[str, LLMProfile] = {}
            for key, data in profiles_payload.items():
                if not isinstance(data, dict):
                    continue
                provider = str(data.get("provider") or "ollama").lower()
                profile = LLMProfile(
                    key=key,
                    provider=provider,
                    model=data.get("model"),
                    base_url=data.get("base_url"),
                    api_key=data.get("api_key"),
                    temperature=float(data.get("temperature", DEFAULT_LLM_TEMPERATURES.get(provider, 0.2))),
                    timeout=float(data.get("timeout", DEFAULT_LLM_TIMEOUTS.get(provider, 30.0))),
                    title=str(data.get("title") or ""),
                    enabled=bool(data.get("enabled", True)),
                )
                profiles[key] = profile
            if profiles:
                cfg.llm_profiles = profiles

        routes_payload = payload.get("llm_routes")
        if isinstance(routes_payload, dict):
            routes: Dict[str, LLMRoute] = {}
            for name, data in routes_payload.items():
                if not isinstance(data, dict):
                    continue
                strategy_raw = str(data.get("strategy") or "single").lower()
                normalized = LLM_STRATEGY_ALIASES.get(strategy_raw, strategy_raw)
                route = LLMRoute(
                    name=name,
                    title=str(data.get("title") or ""),
                    strategy=normalized if normalized in ALLOWED_LLM_STRATEGIES else "single",
                    majority_threshold=max(1, int(data.get("majority_threshold", 3) or 3)),
                    primary=str(data.get("primary") or "global"),
                    ensemble=[
                        str(item)
                        for item in data.get("ensemble", [])
                        if isinstance(item, str)
                    ],
                )
                routes[name] = route
            if routes:
                cfg.llm_routes = routes
                routes_defined = True

        route_key = payload.get("llm_route")
        if isinstance(route_key, str) and route_key:
            cfg.llm_route = route_key

        llm_payload = payload.get("llm")
        if isinstance(llm_payload, dict):
            route_value = llm_payload.get("route")
            if isinstance(route_value, str) and route_value:
                cfg.llm_route = route_value
            primary_data = llm_payload.get("primary")
            if isinstance(primary_data, dict):
                cfg.llm.primary = _dict_to_endpoint(primary_data)
                inline_primary_loaded = True

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

        if inline_primary_loaded and not routes_defined:
            primary_key = "inline_global_primary"
            cfg.llm_profiles[primary_key] = LLMProfile.from_endpoint(
                primary_key,
                cfg.llm.primary,
                title="全局主模型",
            )
            ensemble_keys: List[str] = []
            for idx, endpoint in enumerate(cfg.llm.ensemble, start=1):
                inline_key = f"inline_global_ensemble_{idx}"
                cfg.llm_profiles[inline_key] = LLMProfile.from_endpoint(
                    inline_key,
                    endpoint,
                    title=f"全局协作#{idx}",
                )
                ensemble_keys.append(inline_key)
            auto_route = cfg.llm_routes.get("global") or LLMRoute(name="global", title="全局默认路由")
            auto_route.strategy = cfg.llm.strategy
            auto_route.majority_threshold = cfg.llm.majority_threshold
            auto_route.primary = primary_key
            auto_route.ensemble = ensemble_keys
            cfg.llm_routes["global"] = auto_route
            cfg.llm_route = cfg.llm_route or "global"

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
                route = data.get("llm_route")
                route_name = str(route).strip() if isinstance(route, str) and route else None
                resolved = llm_cfg
                if route_name and route_name in cfg.llm_routes:
                    resolved = cfg.llm_routes[route_name].resolve(cfg.llm_profiles)
                new_departments[code] = DepartmentSettings(
                    code=code,
                    title=title,
                    description=description,
                    weight=weight,
                    llm=resolved,
                    llm_route=route_name,
                )
            if new_departments:
                cfg.departments = new_departments

        cfg.sync_runtime_llm()


def save_config(cfg: AppConfig | None = None) -> None:
    cfg = cfg or CONFIG
    cfg.sync_runtime_llm()
    path = cfg.data_paths.config_file
    payload = {
        "tushare_token": cfg.tushare_token,
        "force_refresh": cfg.force_refresh,
        "decision_method": cfg.decision_method,
        "llm_route": cfg.llm_route,
        "llm": {
            "route": cfg.llm_route,
            "strategy": cfg.llm.strategy if cfg.llm.strategy in ALLOWED_LLM_STRATEGIES else "single",
            "majority_threshold": cfg.llm.majority_threshold,
            "primary": _endpoint_to_dict(cfg.llm.primary),
            "ensemble": [_endpoint_to_dict(ep) for ep in cfg.llm.ensemble],
        },
        "llm_profiles": {
            key: profile.to_dict()
            for key, profile in cfg.llm_profiles.items()
        },
        "llm_routes": {
            name: route.to_dict()
            for name, route in cfg.llm_routes.items()
        },
        "departments": {
            code: {
                "title": dept.title,
                "description": dept.description,
                "weight": dept.weight,
                "llm_route": dept.llm_route,
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
        sanitized = api_key.strip()
        cfg.llm.primary.api_key = sanitized
        route = cfg.llm_routes.get(cfg.llm_route)
        if route:
            profile = cfg.llm_profiles.get(route.primary)
            if profile:
                profile.api_key = sanitized

    cfg.sync_runtime_llm()


_load_from_file(CONFIG)
_load_env_defaults(CONFIG)


def get_config() -> AppConfig:
    """Return a mutable global configuration instance."""

    return CONFIG
