"""Application configuration models and helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional


LOGGER = logging.getLogger(__name__)


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2] / "app" / "data"


def _safe_float(value: object, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


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
        config_override = os.getenv("LLM_QUANT_CONFIG_PATH")
        if config_override:
            config_path = Path(config_override).expanduser()
            config_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            config_path = self.root / "config.json"
        self.config_file = config_path


@dataclass
class PortfolioSettings:
    """Portfolio configuration settings."""
    
    initial_capital: float = 1000000  # 默认100万
    currency: str = "CNY"  # 默认人民币
    max_position: float = 0.2  # 单个持仓上限 20%
    min_position: float = 0.02  # 单个持仓下限 2%
    max_total_positions: int = 20  # 最大持仓数
    max_sector_exposure: float = 0.35  # 行业敞口上限 35%


@dataclass
class AlertChannelSettings:
    """Configuration for external alert delivery channels."""

    key: str
    kind: str = "webhook"
    url: str = ""
    enabled: bool = True
    level: str = "warning"
    tags: List[str] = field(default_factory=list)
    headers: Dict[str, str] = field(default_factory=dict)
    timeout: float = 3.0
    method: str = "POST"
    template: str = ""
    signing_secret: Optional[str] = None
    cooldown_seconds: float = 0.0
    extra_params: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "kind": self.kind,
            "url": self.url,
            "enabled": self.enabled,
            "level": self.level,
            "tags": list(self.tags),
            "headers": dict(self.headers),
            "timeout": self.timeout,
            "method": self.method,
            "template": self.template,
            "cooldown_seconds": self.cooldown_seconds,
            "extra_params": dict(self.extra_params),
        }
        if self.signing_secret:
            payload["signing_secret"] = self.signing_secret
        return payload


@dataclass
class AgentWeights:
    """Default weighting for decision agents."""

    momentum: float = 0.30
    value: float = 0.20
    news: float = 0.20
    liquidity: float = 0.15
    macro: float = 0.15
    risk: float = 1.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "A_mom": self.momentum,
            "A_val": self.value,
            "A_news": self.news,
            "A_liq": self.liquidity,
            "A_macro": self.macro,
            "A_risk": self.risk,
        }

    def update_from_dict(self, data: Mapping[str, float]) -> None:
        mapping = {
            "A_mom": "momentum",
            "momentum": "momentum",
            "A_val": "value",
            "value": "value",
            "A_news": "news",
            "news": "news",
            "A_liq": "liquidity",
            "liquidity": "liquidity",
            "A_macro": "macro",
            "macro": "macro",
            "A_risk": "risk",
            "risk": "risk",
        }
        for key, attr in mapping.items():
            if key in data and data[key] is not None:
                try:
                    setattr(self, attr, float(data[key]))
                except (TypeError, ValueError):
                    continue

    @classmethod
    def from_dict(cls, data: Mapping[str, float]) -> "AgentWeights":
        inst = cls()
        inst.update_from_dict(data)
        return inst

DEFAULT_LLM_MODEL_OPTIONS: Dict[str, Dict[str, object]] = {
    "ollama": {
        "models": ["llama3", "phi3", "qwen2"],
        "base_url": "http://localhost:11434",
        "temperature": 0.2,
        "timeout": 30.0,
        "rate_limit_per_minute": 120,
        "rate_limit_burst": 40,
        "cache_enabled": True,
        "cache_ttl_seconds": 120,
    },
    "openai": {
        "models": ["gpt-4o-mini", "gpt-4.1-mini", "gpt-3.5-turbo"],
        "base_url": "https://api.openai.com",
        "temperature": 0.2,
        "timeout": 30.0,
        "rate_limit_per_minute": 60,
        "rate_limit_burst": 30,
        "cache_enabled": True,
        "cache_ttl_seconds": 180,
    },
    "deepseek": {
        "models": ["deepseek-chat", "deepseek-coder"],
        "base_url": "https://api.deepseek.com",
        "temperature": 0.2,
        "timeout": 45.0,
        "rate_limit_per_minute": 45,
        "rate_limit_burst": 20,
        "cache_enabled": True,
        "cache_ttl_seconds": 240,
    },
    "wenxin": {
        "models": ["ERNIE-Speed", "ERNIE-Bot"],
        "base_url": "https://aip.baidubce.com",
        "temperature": 0.2,
        "timeout": 60.0,
        "rate_limit_per_minute": 30,
        "rate_limit_burst": 15,
        "cache_enabled": True,
        "cache_ttl_seconds": 300,
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
class LLMProvider:
    """Provider level configuration shared across profiles and routes."""

    key: str
    title: str = ""
    base_url: str = ""
    api_key: Optional[str] = None
    models: List[str] = field(default_factory=list)
    default_model: Optional[str] = None
    default_temperature: float = 0.2
    default_timeout: float = 30.0
    prompt_template: str = ""
    enabled: bool = True
    mode: str = "openai"  # openai 或 ollama
    rate_limit_per_minute: int = 60
    rate_limit_burst: int = 30
    cache_enabled: bool = True
    cache_ttl_seconds: int = 180

    def to_dict(self) -> Dict[str, object]:
        return {
            "title": self.title,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "models": list(self.models),
            "default_model": self.default_model,
            "default_temperature": self.default_temperature,
            "default_timeout": self.default_timeout,
            "prompt_template": self.prompt_template,
            "enabled": self.enabled,
            "mode": self.mode,
            "rate_limit_per_minute": self.rate_limit_per_minute,
            "rate_limit_burst": self.rate_limit_burst,
            "cache_enabled": self.cache_enabled,
            "cache_ttl_seconds": self.cache_ttl_seconds,
        }


@dataclass
class LLMEndpoint:
    """Resolved endpoint payload used for actual LLM calls."""

    provider: str = "ollama"
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    temperature: Optional[float] = None
    timeout: Optional[float] = None
    prompt_template: Optional[str] = None

    def __post_init__(self) -> None:
        self.provider = (self.provider or "ollama").lower()
        if self.temperature is None:
            self.temperature = DEFAULT_LLM_TEMPERATURES.get(self.provider)
        else:
            try:
                self.temperature = float(self.temperature)
            except (TypeError, ValueError):
                self.temperature = DEFAULT_LLM_TEMPERATURES.get(self.provider)


@dataclass
class LLMCostSettings:
    """Configurable budgets and weights for LLM cost control."""

    enabled: bool = False
    hourly_budget: float = 5.0
    daily_budget: float = 50.0
    monthly_budget: float = 500.0
    model_weights: Dict[str, float] = field(default_factory=dict)

    def update_from_dict(self, data: Mapping[str, object]) -> None:
        if "enabled" in data:
            self.enabled = bool(data.get("enabled"))
        if "hourly_budget" in data:
            self.hourly_budget = _safe_float(data.get("hourly_budget"), self.hourly_budget)
        if "daily_budget" in data:
            self.daily_budget = _safe_float(data.get("daily_budget"), self.daily_budget)
        if "monthly_budget" in data:
            self.monthly_budget = _safe_float(data.get("monthly_budget"), self.monthly_budget)
        weights = data.get("model_weights") if isinstance(data, Mapping) else None
        if isinstance(weights, Mapping):
            normalized: Dict[str, float] = {}
            for key, value in weights.items():
                try:
                    normalized[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            if normalized:
                self.model_weights = normalized

    def to_cost_limits(self):
        """Convert into runtime `CostLimits` descriptor."""

        from app.llm.cost import CostLimits  # Imported lazily to avoid cycles

        weights: Dict[str, float] = {}
        for key, value in (self.model_weights or {}).items():
            try:
                weights[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return CostLimits(
            hourly_budget=float(self.hourly_budget),
            daily_budget=float(self.daily_budget),
            monthly_budget=float(self.monthly_budget),
            model_weights=weights,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "LLMCostSettings":
        inst = cls()
        inst.update_from_dict(data)
        return inst


@dataclass
class LLMConfig:
    """LLM configuration allowing single or ensemble strategies."""

    primary: LLMEndpoint = field(default_factory=LLMEndpoint)
    ensemble: List[LLMEndpoint] = field(default_factory=list)
    strategy: str = "single"  # Options: single, majority, leader
    majority_threshold: int = 3


def _default_llm_providers() -> Dict[str, LLMProvider]:
    providers: Dict[str, LLMProvider] = {}
    for provider, meta in DEFAULT_LLM_MODEL_OPTIONS.items():
        models = list(meta.get("models", []))
        mode = "ollama" if provider == "ollama" else "openai"
        providers[provider] = LLMProvider(
            key=provider,
            title=f"默认 {provider}",
            base_url=str(meta.get("base_url", DEFAULT_LLM_BASE_URLS.get(provider, "")) or ""),
            models=models,
            default_model=models[0] if models else DEFAULT_LLM_MODELS.get(provider),
            default_temperature=float(meta.get("temperature", DEFAULT_LLM_TEMPERATURES.get(provider, 0.2))),
            default_timeout=float(meta.get("timeout", DEFAULT_LLM_TIMEOUTS.get(provider, 30.0))),
            mode=mode,
            rate_limit_per_minute=int(meta.get("rate_limit_per_minute", 60) or 0),
            rate_limit_burst=int(meta.get("rate_limit_burst", meta.get("rate_limit_per_minute", 60)) or 0),
            cache_enabled=bool(meta.get("cache_enabled", True)),
            cache_ttl_seconds=int(meta.get("cache_ttl_seconds", 180) or 0),
        )
    return providers


@dataclass
class DepartmentSettings:
    """Configuration for a single decision department."""

    code: str
    title: str
    description: str = ""
    weight: float = 1.0
    data_scope: List[str] = field(default_factory=list)
    prompt: str = ""
    llm: LLMConfig = field(default_factory=LLMConfig)
    prompt_template_id: Optional[str] = None

    @property
    def prompt_template_version(self) -> Optional[str]:
        value = getattr(self.llm.primary, "prompt_template", None)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @prompt_template_version.setter
    def prompt_template_version(self, value: Optional[str]) -> None:
        if value is None:
            self.llm.primary.prompt_template = None
            return
        text = str(value).strip()
        self.llm.primary.prompt_template = text or None


def _default_departments() -> Dict[str, DepartmentSettings]:
    presets = [
        {
            "code": "momentum",
            "title": "动量策略部门",
            "description": "跟踪价格动量与量价共振，评估短线趋势延续的概率。",
            "data_scope": [
                "daily.close",
                "daily.open",
                "daily_basic.turnover_rate",
                "factors.mom_20",
                "factors.mom_60",
                "factors.volat_20",
            ],
            "prompt": "你主导动量风格研究，关注价格与成交量的加速变化，需在保持纪律的前提下判定短期多空倾向。",
        },
        {
            "code": "value",
            "title": "价值评估部门",
            "description": "衡量估值水平与盈利质量，为中期配置提供性价比判断。",
            "data_scope": [
                "daily_basic.pe",
                "daily_basic.pb",
                "daily_basic.ps",
                "daily_basic.dv_ratio",
                "factors.turn_20",
            ],
            "prompt": "你负责价值与质量评估，应结合估值分位、盈利持续性及安全边际给出配置建议。",
        },
        {
            "code": "news",
            "title": "新闻情绪部门",
            "description": "监控舆情热度与事件影响，识别情绪驱动的短期风险与机会。",
            "data_scope": [
                "news.sentiment_index",
                "news.heat_score",
            ],
            "prompt": "你专注新闻和事件驱动，应评估正负面舆情对标的短线波动的可能影响。",
        },
        {
            "code": "liquidity",
            "title": "流动性评估部门",
            "description": "衡量成交活跃度与交易成本，控制进出场的实现可能性。",
            "data_scope": [
                "daily_basic.volume_ratio",
                "daily_basic.turnover_rate",
                "daily_basic.turnover_rate_f",
                "factors.turn_20",
                "stk_limit.up_limit",
                "stk_limit.down_limit",
            ],
            "prompt": "你负责评估该标的的流动性与滑点风险，需要提出可执行的仓位调整建议。",
        },
        {
            "code": "macro",
            "title": "宏观研究部门",
            "description": "追踪宏观与行业景气度，为行业配置和风险偏好提供参考。",
            "data_scope": [
                "macro.industry_heat",
                "index.performance_peers",
                "macro.relative_strength",
            ],
            "prompt": "你负责宏观与行业研判，应结合宏观周期、行业景气与相对强弱给出方向性意见。",
        },
        {
            "code": "risk",
            "title": "风险控制部门",
            "description": "监控极端风险、合规与交易限制，必要时行使否决。",
            "data_scope": [
                "daily.pct_chg",
                "suspend.suspend_type",
                "stk_limit.up_limit",
                "stk_limit.down_limit",
            ],
            "prompt": "你负责风险控制，应识别停牌、涨跌停、持仓约束等因素，必要时提出减仓或观望建议。",
        },
    ]
    return {
        item["code"]: DepartmentSettings(
            code=item["code"],
            title=item["title"],
            description=item.get("description", ""),
            data_scope=list(item.get("data_scope", [])),
            prompt=item.get("prompt", ""),
            prompt_template_id=f"{item['code']}_dept",
        )
        for item in presets
    }


def _normalize_data_scope(raw: object) -> List[str]:
    if isinstance(raw, str):
        tokens = raw.replace(";", "\n").replace(",", "\n").splitlines()
        return [token.strip() for token in tokens if token.strip()]
    if isinstance(raw, Iterable) and not isinstance(raw, (bytes, bytearray, str)):
        return [str(item).strip() for item in raw if str(item).strip()]
    return []


def _default_rss_sources() -> Dict[str, object]:
    return {
        "cls_depth_headline": {
            "url": "https://rsshub.app/cls/depth/1000",
            "source": "财联社·深度头条",
            "enabled": True,
            "hours_back": 48,
            "max_items": 80,
            "ts_codes": [],
            "keywords": [],
        },
        "cls_depth_stock": {
            "url": "https://rsshub.app/cls/depth/1003",
            "source": "财联社·深度股市",
            "enabled": True,
            "hours_back": 48,
            "max_items": 80,
            "ts_codes": [],
            "keywords": [],
        },
        "cls_depth_hk": {
            "url": "https://rsshub.app/cls/depth/1135",
            "source": "财联社·深度港股",
            "enabled": True,
            "hours_back": 48,
            "max_items": 80,
            "ts_codes": [],
            "keywords": [],
        },
        "cls_depth_global": {
            "url": "https://rsshub.app/cls/depth/1007",
            "source": "财联社·深度环球",
            "enabled": True,
            "hours_back": 48,
            "max_items": 80,
            "ts_codes": [],
            "keywords": [],
        },
        "cls_telegraph_all": {
            "url": "https://rsshub.app/cls/telegraph",
            "source": "财联社·电报",
            "enabled": True,
            "hours_back": 48,
            "max_items": 80,
            "ts_codes": [],
            "keywords": [],
        },
        "cls_telegraph_red": {
            "url": "https://rsshub.app/cls/telegraph/red",
            "source": "财联社·电报加红",
            "enabled": True,
            "hours_back": 48,
            "max_items": 80,
            "ts_codes": [],
            "keywords": [],
        },
    }


@dataclass
class AppConfig:
    """User configurable settings persisted in a simple structure."""

    tushare_token: Optional[str] = None
    log_level: str = "DEBUG"
    rss_sources: Dict[str, object] = field(default_factory=_default_rss_sources)
    decision_method: str = "nash"
    data_paths: DataPaths = field(default_factory=DataPaths)
    agent_weights: AgentWeights = field(default_factory=AgentWeights)
    force_refresh: bool = False
    auto_update_data: bool = False
    data_update_interval: int = 7  # 数据更新间隔（天）
    llm_providers: Dict[str, LLMProvider] = field(default_factory=_default_llm_providers)
    llm: LLMConfig = field(default_factory=LLMConfig)
    llm_cost: LLMCostSettings = field(default_factory=LLMCostSettings)
    departments: Dict[str, DepartmentSettings] = field(default_factory=_default_departments)
    portfolio: PortfolioSettings = field(default_factory=PortfolioSettings)
    alert_channels: Dict[str, AlertChannelSettings] = field(default_factory=dict)

    def resolve_llm(self, route: Optional[str] = None) -> LLMConfig:
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
        "prompt_template": endpoint.prompt_template,
    }


def _dict_to_endpoint(data: Dict[str, object]) -> LLMEndpoint:
    payload = {
        key: data.get(key)
        for key in (
            "provider",
            "model",
            "base_url",
            "api_key",
            "temperature",
            "timeout",
            "prompt_template",
        )
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

    if not isinstance(payload, dict):
        return

    if "tushare_token" in payload:
        cfg.tushare_token = payload.get("tushare_token") or None
    if "force_refresh" in payload:
        cfg.force_refresh = bool(payload.get("force_refresh"))
    if "auto_update_data" in payload:
        cfg.auto_update_data = bool(payload.get("auto_update_data"))
    log_level_raw = payload.get("log_level")
    if isinstance(log_level_raw, str) and log_level_raw.strip():
        cfg.log_level = log_level_raw.strip()
    if "decision_method" in payload:
        cfg.decision_method = str(payload.get("decision_method") or cfg.decision_method)

    rss_payload = payload.get("rss_sources")
    default_rss = _default_rss_sources()
    if isinstance(rss_payload, dict):
        for key, value in rss_payload.items():
            if isinstance(value, dict):
                default_rss[str(key)] = value
    cfg.rss_sources = default_rss

    weights_payload = payload.get("agent_weights")
    if isinstance(weights_payload, dict):
        cfg.agent_weights.update_from_dict(weights_payload)

    portfolio_payload = payload.get("portfolio")
    if isinstance(portfolio_payload, dict):
        limits_payload = portfolio_payload.get("position_limits")
        if not isinstance(limits_payload, dict):
            limits_payload = portfolio_payload

        current = cfg.portfolio

        def _float_value(container: Dict[str, object], key: str, fallback: float) -> float:
            value = container.get(key) if isinstance(container, dict) else None
            try:
                return float(value)
            except (TypeError, ValueError):
                return fallback

        def _int_value(container: Dict[str, object], key: str, fallback: int) -> int:
            value = container.get(key) if isinstance(container, dict) else None
            try:
                return int(value)
            except (TypeError, ValueError):
                return fallback

        updated_portfolio = PortfolioSettings(
            initial_capital=_float_value(portfolio_payload, "initial_capital", current.initial_capital),
            currency=str(portfolio_payload.get("currency") or current.currency),
            max_position=_float_value(limits_payload, "max_position", current.max_position),
            min_position=_float_value(limits_payload, "min_position", current.min_position),
            max_total_positions=_int_value(limits_payload, "max_total_positions", current.max_total_positions),
            max_sector_exposure=_float_value(limits_payload, "max_sector_exposure", current.max_sector_exposure),
        )
        cfg.portfolio = updated_portfolio

    alert_channels_payload = payload.get("alert_channels")
    if isinstance(alert_channels_payload, dict):
        channels: Dict[str, AlertChannelSettings] = {}
        for key, data in alert_channels_payload.items():
            if not isinstance(data, dict):
                continue
            normalized_key = str(key)
            raw_tags = data.get("tags")
            tags: List[str] = []
            if isinstance(raw_tags, list):
                tags = [
                    str(tag).strip()
                    for tag in raw_tags
                    if isinstance(tag, str) and tag.strip()
                ]
            headers: Dict[str, str] = {}
            raw_headers = data.get("headers")
            if isinstance(raw_headers, Mapping):
                headers = {
                    str(h_key): str(h_val)
                    for h_key, h_val in raw_headers.items()
                    if h_key is not None
                }
            extra_params: Dict[str, object] = {}
            raw_extra = data.get("extra_params")
            if isinstance(raw_extra, Mapping):
                extra_params = dict(raw_extra)
            channel = AlertChannelSettings(
                key=normalized_key,
                kind=str(data.get("kind") or "webhook"),
                url=str(data.get("url") or ""),
                enabled=bool(data.get("enabled", True)),
                level=str(data.get("level") or "warning"),
                tags=tags,
                headers=headers,
                timeout=float(data.get("timeout", 3.0) or 3.0),
                method=str(data.get("method") or "POST").upper(),
                template=str(data.get("template") or ""),
                signing_secret=str(data.get("signing_secret")) if data.get("signing_secret") else None,
                cooldown_seconds=float(data.get("cooldown_seconds", 0.0) or 0.0),
                extra_params=extra_params,
            )
            if channel.url:
                channels[channel.key] = channel
        cfg.alert_channels = channels

    cost_payload = payload.get("llm_cost")
    if isinstance(cost_payload, dict):
        cfg.llm_cost.update_from_dict(cost_payload)

    legacy_profiles: Dict[str, Dict[str, object]] = {}
    legacy_routes: Dict[str, Dict[str, object]] = {}

    providers_payload = payload.get("llm_providers")
    if isinstance(providers_payload, dict):
        providers: Dict[str, LLMProvider] = {}
        for key, data in providers_payload.items():
            if not isinstance(data, dict):
                continue
            provider_key = str(key).lower()
            models_raw = data.get("models")
            if isinstance(models_raw, str):
                models = [item.strip() for item in models_raw.split(',') if item.strip()]
            elif isinstance(models_raw, list):
                models = [str(item).strip() for item in models_raw if str(item).strip()]
            else:
                models = []
            defaults = DEFAULT_LLM_MODEL_OPTIONS.get(provider_key, {})
            def _safe_int(value: object, fallback: int) -> int:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return fallback
            rate_limit_per_minute = _safe_int(data.get("rate_limit_per_minute"), int(defaults.get("rate_limit_per_minute", 60) or 0))
            rate_limit_burst = _safe_int(
                data.get("rate_limit_burst"),
                int(defaults.get("rate_limit_burst", defaults.get("rate_limit_per_minute", rate_limit_per_minute)) or rate_limit_per_minute or 0),
            )
            cache_ttl_seconds = _safe_int(
                data.get("cache_ttl_seconds"),
                int(defaults.get("cache_ttl_seconds", 180) or 0),
            )
            provider = LLMProvider(
                key=provider_key,
                title=str(data.get("title") or ""),
                base_url=str(data.get("base_url") or ""),
                api_key=data.get("api_key"),
                models=models,
                default_model=data.get("default_model") or (models[0] if models else None),
                default_temperature=float(data.get("default_temperature", 0.2)),
                default_timeout=float(data.get("default_timeout", 30.0)),
                prompt_template=str(data.get("prompt_template") or ""),
                enabled=bool(data.get("enabled", True)),
                mode=str(data.get("mode") or ("ollama" if provider_key == "ollama" else "openai")),
                rate_limit_per_minute=max(0, rate_limit_per_minute),
                rate_limit_burst=max(1, rate_limit_burst) if rate_limit_per_minute > 0 else max(0, rate_limit_burst),
                cache_enabled=bool(data.get("cache_enabled", defaults.get("cache_enabled", True))),
                cache_ttl_seconds=max(0, cache_ttl_seconds),
            )
            providers[provider.key] = provider
        if providers:
            cfg.llm_providers = providers

    profiles_payload = payload.get("llm_profiles")
    if isinstance(profiles_payload, dict):
        for key, data in profiles_payload.items():
            if isinstance(data, dict):
                legacy_profiles[str(key)] = data

    routes_payload = payload.get("llm_routes")
    if isinstance(routes_payload, dict):
        for name, data in routes_payload.items():
            if isinstance(data, dict):
                legacy_routes[str(name)] = data

    def _endpoint_from_payload(item: object) -> LLMEndpoint:
        if isinstance(item, dict):
            return _dict_to_endpoint(item)
        if isinstance(item, str):
            profile_data = legacy_profiles.get(item)
            if isinstance(profile_data, dict):
                return _dict_to_endpoint(profile_data)
            return LLMEndpoint(provider=item)
        return LLMEndpoint()

    def _resolve_route(route_name: str) -> Optional[LLMConfig]:
        route_data = legacy_routes.get(route_name)
        if not route_data:
            return None
        strategy_raw = str(route_data.get("strategy") or "single").lower()
        strategy = LLM_STRATEGY_ALIASES.get(strategy_raw, strategy_raw)
        primary_ref = route_data.get("primary")
        primary_ep = _endpoint_from_payload(primary_ref)
        ensemble_refs = route_data.get("ensemble", [])
        ensemble_eps = [
            _endpoint_from_payload(ref)
            for ref in ensemble_refs
            if isinstance(ref, (dict, str))
        ]
        cfg_obj = LLMConfig(
            primary=primary_ep,
            ensemble=ensemble_eps,
            strategy=strategy if strategy in ALLOWED_LLM_STRATEGIES else "single",
            majority_threshold=max(1, int(route_data.get("majority_threshold", 3) or 3)),
        )
        return cfg_obj

    llm_payload = payload.get("llm")
    if isinstance(llm_payload, dict):
        route_value = llm_payload.get("route")
        resolved_cfg = None
        if isinstance(route_value, str) and route_value:
            resolved_cfg = _resolve_route(route_value)
        if resolved_cfg is None:
            resolved_cfg = LLMConfig()
            primary_data = llm_payload.get("primary")
            if isinstance(primary_data, dict):
                resolved_cfg.primary = _dict_to_endpoint(primary_data)
            ensemble_data = llm_payload.get("ensemble")
            if isinstance(ensemble_data, list):
                resolved_cfg.ensemble = [
                    _dict_to_endpoint(item)
                    for item in ensemble_data
                    if isinstance(item, dict)
                ]
            strategy_raw = llm_payload.get("strategy")
            if isinstance(strategy_raw, str):
                normalized = LLM_STRATEGY_ALIASES.get(strategy_raw, strategy_raw)
                if normalized in ALLOWED_LLM_STRATEGIES:
                    resolved_cfg.strategy = normalized
            majority = llm_payload.get("majority_threshold")
            if isinstance(majority, int) and majority > 0:
                resolved_cfg.majority_threshold = majority
        cfg.llm = resolved_cfg

    departments_payload = payload.get("departments")
    if isinstance(departments_payload, dict):
        new_departments: Dict[str, DepartmentSettings] = {}
        for code, data in departments_payload.items():
            if not isinstance(data, dict):
                continue
            current_setting = cfg.departments.get(code)
            title = data.get("title") or code
            description = data.get("description") or ""
            weight = float(data.get("weight", 1.0))
            prompt_text = str(data.get("prompt") or "")
            data_scope = _normalize_data_scope(data.get("data_scope"))
            llm_cfg = LLMConfig()
            route_name = data.get("llm_route")
            resolved_cfg = None
            if isinstance(route_name, str) and route_name:
                resolved_cfg = _resolve_route(route_name)
            if resolved_cfg is None:
                llm_data = data.get("llm")
                if isinstance(llm_data, dict):
                    primary_data = llm_data.get("primary")
                    if isinstance(primary_data, dict):
                        llm_cfg.primary = _dict_to_endpoint(primary_data)
                    ensemble_data = llm_data.get("ensemble")
                    if isinstance(ensemble_data, list):
                        llm_cfg.ensemble = [
                            _dict_to_endpoint(item)
                            for item in ensemble_data
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
                resolved_cfg = llm_cfg
            template_id_raw = data.get("prompt_template_id")
            if isinstance(template_id_raw, str):
                template_id_candidate = template_id_raw.strip()
            elif template_id_raw is not None:
                template_id_candidate = str(template_id_raw).strip()
            else:
                template_id_candidate = ""
            if template_id_candidate:
                template_id = template_id_candidate
            elif current_setting and current_setting.prompt_template_id:
                template_id = current_setting.prompt_template_id
            else:
                template_id = f"{code}_dept"

            new_departments[code] = DepartmentSettings(
                code=code,
                title=title,
                description=description,
                weight=weight,
                data_scope=data_scope,
                prompt=prompt_text,
                llm=resolved_cfg,
                prompt_template_id=template_id,
            )
            template_version_raw = data.get("prompt_template_version")
            template_version = None
            if isinstance(template_version_raw, str):
                template_version = template_version_raw.strip() or None
            elif template_version_raw is not None:
                template_version = str(template_version_raw).strip() or None
            elif isinstance(resolved_cfg.primary.prompt_template, str):
                template_version = resolved_cfg.primary.prompt_template.strip() or None
            elif current_setting:
                template_version = current_setting.prompt_template_version
            if template_version:
                new_departments[code].prompt_template_version = template_version
        if new_departments:
            cfg.departments = new_departments

    cfg.sync_runtime_llm()


def save_config(cfg: AppConfig | None = None) -> None:
    cfg = cfg or CONFIG
    cfg.sync_runtime_llm()
    path = cfg.data_paths.config_file
    payload = {
        "tushare_token": cfg.tushare_token,
        "log_level": cfg.log_level,
        "force_refresh": cfg.force_refresh,
        "auto_update_data": cfg.auto_update_data,
        "decision_method": cfg.decision_method,
        "rss_sources": cfg.rss_sources,
        "agent_weights": cfg.agent_weights.as_dict(),
        "portfolio": {
            "initial_capital": cfg.portfolio.initial_capital,
            "currency": cfg.portfolio.currency,
            "position_limits": {
                "max_position": cfg.portfolio.max_position,
                "min_position": cfg.portfolio.min_position,
                "max_total_positions": cfg.portfolio.max_total_positions,
                "max_sector_exposure": cfg.portfolio.max_sector_exposure,
            },
        },
        "alert_channels": {
            name: channel.to_dict()
            for name, channel in cfg.alert_channels.items()
        },
        "llm": {
            "strategy": cfg.llm.strategy if cfg.llm.strategy in ALLOWED_LLM_STRATEGIES else "single",
            "majority_threshold": cfg.llm.majority_threshold,
            "primary": _endpoint_to_dict(cfg.llm.primary),
            "ensemble": [_endpoint_to_dict(ep) for ep in cfg.llm.ensemble],
        },
        "llm_cost": {
            "enabled": cfg.llm_cost.enabled,
            "hourly_budget": cfg.llm_cost.hourly_budget,
            "daily_budget": cfg.llm_cost.daily_budget,
            "monthly_budget": cfg.llm_cost.monthly_budget,
            "model_weights": cfg.llm_cost.model_weights,
        },
        "llm_providers": {
            key: provider.to_dict()
            for key, provider in cfg.llm_providers.items()
        },
        "departments": {
            code: {
                "title": dept.title,
                "description": dept.description,
                "weight": dept.weight,
                "data_scope": list(dept.data_scope),
                "prompt": dept.prompt,
                "prompt_template_id": dept.prompt_template_id,
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
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)

    try:
        existing = path.read_text(encoding="utf-8")
    except OSError:
        existing = None

    if existing == serialized:
        LOGGER.info("配置未变更，跳过写入：%s", path)
        return

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            fh.write(serialized)
        LOGGER.info("配置已写入：%s", path)
    except OSError:
        LOGGER.exception("配置写入失败：%s", path)
        return

    try:
        from app.utils import alerts as _alerts  # 延迟导入以避免循环
        _alerts.configure_channels(cfg.alert_channels)
    except Exception:  # noqa: BLE001
        LOGGER.debug("更新告警通道失败", exc_info=True)


def _load_env_defaults(cfg: AppConfig) -> None:
    """Populate sensitive fields from environment variables if present."""

    token = os.getenv("TUSHARE_TOKEN")
    if token:
        cfg.tushare_token = token.strip()

    log_level_env = os.getenv("LLM_QUANT_LOG_LEVEL")
    if log_level_env:
        cfg.log_level = log_level_env.strip()

    api_key = os.getenv("LLM_API_KEY")
    if api_key:
        sanitized = api_key.strip()
        cfg.llm.primary.api_key = sanitized
        provider_cfg = cfg.llm_providers.get(cfg.llm.primary.provider)
        if provider_cfg:
            provider_cfg.api_key = sanitized

    webhook = os.getenv("LLM_QUANT_ALERT_WEBHOOK")
    if webhook:
        key = "env_webhook"
        channel = AlertChannelSettings(
            key=key,
            kind="webhook",
            url=webhook.strip(),
            headers={"Content-Type": "application/json"},
            enabled=True,
            level=str(os.getenv("LLM_QUANT_ALERT_LEVEL", "warning") or "warning"),
        )
        tags_raw = os.getenv("LLM_QUANT_ALERT_TAGS")
        if tags_raw:
            channel.tags = [tag.strip() for tag in tags_raw.split(",") if tag.strip()]
        cfg.alert_channels[key] = channel

    cfg.sync_runtime_llm()


_load_from_file(CONFIG)
_load_env_defaults(CONFIG)

try:
    from app.utils import alerts as _alerts_module  # 延迟导入避免循环依赖
    _alerts_module.configure_channels(CONFIG.alert_channels)
except Exception:  # noqa: BLE001
    LOGGER.debug("初始化告警通道失败", exc_info=True)


def get_config() -> AppConfig:
    """Return a mutable global configuration instance."""

    return CONFIG
