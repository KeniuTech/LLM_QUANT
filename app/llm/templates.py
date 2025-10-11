"""LLM prompt templates management with configuration driven design."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .version import TemplateVersionManager


@dataclass
class PromptTemplate:
    """Configuration driven prompt template."""

    id: str
    name: str
    description: str
    template: str
    variables: List[str]
    max_length: int = 4000
    required_context: List[str] = None
    validation_rules: List[str] = None

    def validate(self) -> List[str]:
        """Validate template configuration."""
        errors = []
        
        # Check template contains all variables
        for var in self.variables:
            if f"{{{var}}}" not in self.template:
                errors.append(f"Template missing variable: {var}")

        # Check required context fields
        if self.required_context:
            for field in self.required_context:
                if not field:
                    errors.append("Empty required context field")

        # Check validation rules format
        if self.validation_rules:
            for rule in self.validation_rules:
                if not rule:
                    errors.append("Empty validation rule")

        return errors

    def format(self, context: Dict[str, Any]) -> str:
        """Format template with provided context."""
        # Validate required context
        if self.required_context:
            missing = [f for f in self.required_context if f not in context]
            if missing:
                raise ValueError(f"Missing required context: {', '.join(missing)}")

        pattern = re.compile(r"\{([^{}]+)\}")

        def _replace(match: re.Match[str]) -> str:
            token = match.group(1)
            if token in context:
                return str(context[token])
            return match.group(0)

        result = pattern.sub(_replace, self.template)

        # Truncate if needed, preserving exact number of characters
        if self.max_length > 0 and len(result) > self.max_length:
            if self.max_length >= 3:
                result = result[: self.max_length - 3] + "..."
            else:
                result = result[: self.max_length]

        return result


class TemplateRegistry:
    """Global registry for prompt templates with version awareness."""

    _templates: Dict[str, PromptTemplate] = {}
    _version_manager: Optional["TemplateVersionManager"] = None
    _default_version_label: str = "1.0.0"

    @classmethod
    def _manager(cls) -> "TemplateVersionManager":
        if cls._version_manager is None:
            from .version import TemplateVersionManager  # Local import to avoid circular dependency

            cls._version_manager = TemplateVersionManager()
        return cls._version_manager

    @classmethod
    def register(
        cls,
        template: PromptTemplate,
        *,
        version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        activate: bool = False,
    ) -> None:
        """Register a new template and optionally version it."""

        errors = template.validate()
        if errors:
            raise ValueError(f"Invalid template {template.id}: {'; '.join(errors)}")

        cls._templates[template.id] = template

        manager = cls._manager()
        existing_versions = manager.list_versions(template.id)
        resolved_metadata: Dict[str, Any] = dict(metadata or {})
        if version:
            manager.add_version(
                template,
                version,
                metadata=resolved_metadata or None,
                activate=activate,
            )
        elif not existing_versions:
            if "source" not in resolved_metadata:
                resolved_metadata["source"] = "default"
            manager.add_version(
                template,
                cls._default_version_label,
                metadata=resolved_metadata,
                activate=True,
            )

    @classmethod
    def register_version(
        cls,
        template_id: str,
        *,
        version: str,
        template: Optional[PromptTemplate] = None,
        metadata: Optional[Dict[str, Any]] = None,
        activate: bool = False,
    ) -> None:
        """Register an additional version for an existing template."""

        base_template = template or cls._templates.get(template_id)
        if not base_template:
            raise ValueError(f"Template {template_id} not found for version registration")

        manager = cls._manager()
        manager.add_version(
            base_template,
            version,
            metadata=metadata,
            activate=activate,
        )

    @classmethod
    def activate_version(cls, template_id: str, version: str) -> None:
        """Activate a specific template version."""

        manager = cls._manager()
        manager.activate_version(template_id, version)

    @classmethod
    def get(
        cls,
        template_id: str,
        *,
        version: Optional[str] = None,
    ) -> Optional[PromptTemplate]:
        """Get template by ID and optional version."""

        manager = cls._manager()
        if version:
            stored = manager.get_version(template_id, version)
            if stored:
                return stored.template

        active = manager.get_active_version(template_id)
        if active:
            return active.template

        return cls._templates.get(template_id)

    @classmethod
    def get_active_version(cls, template_id: str) -> Optional[str]:
        """Return the currently active version label for a template."""

        manager = cls._manager()
        active = manager.get_active_version(template_id)
        return active.version if active else None

    @classmethod
    def list(cls) -> List[PromptTemplate]:
        """List all registered templates (active versions preferred)."""

        collected: Dict[str, PromptTemplate] = {}
        manager = cls._manager()
        for template_id, template in cls._templates.items():
            active = manager.get_active_version(template_id)
            collected[template_id] = active.template if active else template
        return list(collected.values())

    @classmethod
    def list_versions(cls, template_id: str) -> List[str]:
        """List available version labels for a template."""

        manager = cls._manager()
        return [ver.version for ver in manager.list_versions(template_id)]

    @classmethod
    def load_from_json(cls, json_str: str) -> None:
        """Load templates from JSON string."""

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")

        if not isinstance(data, dict):
            raise ValueError("JSON root must be an object")

        for template_id, cfg in data.items():
            if not isinstance(cfg, dict):
                raise ValueError(f"Template {template_id} configuration must be an object")
            version = cfg.get("version")
            metadata = cfg.get("metadata")
            if metadata is not None and not isinstance(metadata, dict):
                raise ValueError(f"Template {template_id} metadata must be an object")
            activate = bool(cfg.get("activate", False))
            template = PromptTemplate(
                id=template_id,
                name=cfg.get("name", template_id),
                description=cfg.get("description", ""),
                template=cfg.get("template", ""),
                variables=cfg.get("variables", []),
                max_length=cfg.get("max_length", 4000),
                required_context=cfg.get("required_context", []),
                validation_rules=cfg.get("validation_rules", [])
            )
            cls.register(
                template,
                version=version,
                metadata=metadata,
                activate=activate,
            )

    @classmethod
    def clear(cls, *, reload_defaults: bool = False) -> None:
        """Clear all registered templates and optionally reload defaults."""

        cls._templates.clear()
        cls._version_manager = None
        if reload_defaults:
            register_default_templates()


DEFAULT_TEMPLATES: Dict[str, Dict[str, Any]] = {}

_FALLBACK_TEMPLATE = (
    "部门：{title}\n"
    "股票代码：{ts_code}\n"
    "交易日：{trade_date}\n\n"
    "【职责与说明】\n"
    "- 描述：{description}\n"
    "- 指令：{instruction}\n\n"
    "【数据概览】\n"
    "- 数据范围：{data_scope}\n"
    "- 核心特征：{features}\n"
    "- 市场背景：{market_snapshot}\n"
    "- 追加数据：{supplements}\n"
)

_INLINE_FALLBACK_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "department_base": {
        "name": "部门基础模板（Fallback）",
        "description": "提示模板目录缺失时加载的简易版本。",
        "template": _FALLBACK_TEMPLATE,
        "variables": [
            "title",
            "ts_code",
            "trade_date",
            "description",
            "instruction",
            "data_scope",
            "features",
            "market_snapshot",
            "supplements",
        ],
        "required_context": [
            "ts_code",
            "trade_date",
            "features",
            "market_snapshot",
        ],
        "metadata": {"source": "inline_fallback"},
    },
    "momentum_dept": {
        "name": "动量研究部门模板（Fallback）",
        "description": "用于缺省情况下的动量提示。",
        "template": _FALLBACK_TEMPLATE,
        "variables": [
            "ts_code",
            "trade_date",
            "data_scope",
            "features",
            "market_snapshot",
            "supplements",
        ],
        "required_context": [
            "ts_code",
            "trade_date",
            "features",
            "market_snapshot",
        ],
        "metadata": {"source": "inline_fallback"},
    },
    "value_dept": {
        "name": "价值评估部门模板（Fallback）",
        "description": "用于缺省情况下的价值提示。",
        "template": _FALLBACK_TEMPLATE,
        "variables": [
            "ts_code",
            "trade_date",
            "data_scope",
            "features",
            "market_snapshot",
            "supplements",
        ],
        "required_context": [
            "ts_code",
            "trade_date",
            "features",
            "market_snapshot",
        ],
        "metadata": {"source": "inline_fallback"},
    },
    "news_dept": {
        "name": "新闻情绪部门模板（Fallback）",
        "description": "用于缺省情况下的新闻提示。",
        "template": _FALLBACK_TEMPLATE,
        "variables": [
            "ts_code",
            "trade_date",
            "data_scope",
            "features",
            "market_snapshot",
            "supplements",
        ],
        "required_context": [
            "ts_code",
            "trade_date",
            "features",
            "market_snapshot",
        ],
        "metadata": {"source": "inline_fallback"},
    },
    "liquidity_dept": {
        "name": "流动性评估部门模板（Fallback）",
        "description": "用于缺省情况下的流动性提示。",
        "template": _FALLBACK_TEMPLATE,
        "variables": [
            "ts_code",
            "trade_date",
            "data_scope",
            "features",
            "market_snapshot",
            "supplements",
        ],
        "required_context": [
            "ts_code",
            "trade_date",
            "features",
            "market_snapshot",
        ],
        "metadata": {"source": "inline_fallback"},
    },
    "macro_dept": {
        "name": "宏观研究部门模板（Fallback）",
        "description": "用于缺省情况下的宏观提示。",
        "template": _FALLBACK_TEMPLATE,
        "variables": [
            "ts_code",
            "trade_date",
            "data_scope",
            "features",
            "market_snapshot",
            "supplements",
        ],
        "required_context": [
            "ts_code",
            "trade_date",
            "features",
            "market_snapshot",
        ],
        "metadata": {"source": "inline_fallback"},
    },
    "risk_dept": {
        "name": "风险控制部门模板（Fallback）",
        "description": "用于缺省情况下的风险提示。",
        "template": _FALLBACK_TEMPLATE,
        "variables": [
            "ts_code",
            "trade_date",
            "data_scope",
            "features",
            "market_snapshot",
            "supplements",
        ],
        "required_context": [
            "ts_code",
            "trade_date",
            "features",
            "market_snapshot",
        ],
        "metadata": {"source": "inline_fallback"},
    },
}

EXTERNAL_TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "data" / "prompt_templates"


def load_external_template_configs(directory: Path | str = EXTERNAL_TEMPLATE_DIR) -> int:
    """Load additional template versions from JSON files in the given directory."""

    directory_path = Path(directory)
    if not directory_path.exists() or not directory_path.is_dir():
        return 0

    loaded = 0
    for file_path in sorted(directory_path.glob("*.json")):
        try:
            raw_data = file_path.read_text(encoding="utf-8")
        except OSError:
            logging.warning("无法读取提示模板配置文件 %s", file_path)
            continue

        try:
            payload = json.loads(raw_data)
        except json.JSONDecodeError as exc:
            logging.warning("提示模板配置文件 %s 解析失败：%s", file_path, exc)
            continue

        enriched_payload = {}
        for template_id, cfg in payload.items():
            if not isinstance(cfg, dict):
                logging.warning(
                    "提示模板配置文件 %s 中的 %s 配置无效（应为对象）",
                    file_path,
                    template_id,
                )
                continue
            metadata = cfg.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            metadata.setdefault("source", file_path.name)
            enriched_payload[template_id] = {
                **cfg,
                "metadata": metadata,
            }

        if not enriched_payload:
            continue

        try:
            TemplateRegistry.load_from_json(json.dumps(enriched_payload, ensure_ascii=False))
            loaded += len(enriched_payload)
        except Exception as exc:  # noqa: BLE001
            logging.warning(
                "注册提示模板配置 %s 失败：%s",
                file_path,
                exc,
            )
    return loaded


def _register_inline_fallbacks() -> None:
    """Register minimal inline templates when file-based templates are unavailable."""

    for template_id, cfg in _INLINE_FALLBACK_TEMPLATES.items():
        try:
            template = PromptTemplate(
                id=template_id,
                name=cfg.get("name", template_id),
                description=cfg.get("description", ""),
                template=cfg.get("template", ""),
                variables=cfg.get("variables", []),
                max_length=cfg.get("max_length", 4000),
                required_context=cfg.get("required_context", []),
                validation_rules=cfg.get("validation_rules", []),
            )
            TemplateRegistry.register(
                template,
                version="0.0.1",
                metadata=cfg.get("metadata"),
                activate=True,
            )
        except ValueError as exc:  # noqa: BLE001
            logging.warning("Fallback template %s 注册失败：%s", template_id, exc)


def register_default_templates() -> None:
    """Load templates from configuration files, falling back to inline defaults if needed."""

    loaded = load_external_template_configs()
    if loaded == 0:
        logging.warning(
            "未在 %s 中找到提示模板配置，使用内置 fallback。",
            EXTERNAL_TEMPLATE_DIR,
        )
        _register_inline_fallbacks()


register_default_templates()
