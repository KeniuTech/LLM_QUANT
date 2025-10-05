"""LLM prompt templates management with configuration driven design."""
from __future__ import annotations

import json
import logging
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

        # Format template
        try:
            result = self.template.format(**context)
        except KeyError as e:
            raise ValueError(f"Missing template variable: {e}")

        # Truncate if needed, preserving exact number of characters
        if len(result) > self.max_length:
            target = self.max_length - 3  # Reserve space for "..."
            if target > 0:  # Only truncate if we have space for content
                result = result[:target] + "..."
            else:
                result = "..."  # If max_length <= 3, just return "..."
        
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


# Default template definitions
DEFAULT_TEMPLATES = {
    "department_base": {
        "name": "部门基础模板",
        "description": "通用的部门分析提示模板",
        "template": """
部门名称：{title}
股票代码：{ts_code}
交易日：{trade_date}

角色说明：{description}
职责指令：{instruction}

【可用数据范围】
{data_scope}

【核心特征】
{features}

【市场背景】
{market_snapshot}

【追加数据】
{supplements}

请基于以上数据给出该部门对当前股票的操作建议。输出必须是 JSON，字段如下：
{{
  "action": "BUY|BUY_S|BUY_M|BUY_L|SELL|HOLD",
  "confidence": 0-1 之间的小数，表示信心,
  "summary": "一句话概括理由",
  "signals": ["详细要点", "..."],
  "risks": ["风险点", "..."]
}}

如需额外数据，请调用工具 `fetch_data`，仅支持请求 `daily` 或 `daily_basic` 表。
请严格返回单个 JSON 对象，不要添加额外文本。
""",
        "variables": [
            "title", "ts_code", "trade_date", "description", "instruction",
            "data_scope", "features", "market_snapshot", "supplements"
        ],
        "required_context": [
            "ts_code", "trade_date", "features", "market_snapshot"
        ],
        "validation_rules": [
            "len(features) > 0",
            "len(market_snapshot) > 0"
        ]
    },
    "momentum_dept": {
        "name": "动量研究部门",
        "description": "专注于动量因子分析的部门模板",
        "template": """
部门名称：动量研究部门
股票代码：{ts_code}
交易日：{trade_date}

角色说明：专注于分析股票价格动量、成交量动量和技术指标动量
职责指令：重点关注以下方面:
1. 价格趋势强度和持续性
2. 成交量配合度
3. 技术指标背离

【可用数据范围】
{data_scope}

【动量特征】
{features}

【市场背景】
{market_snapshot}

【追加数据】
{supplements}

请基于以上数据进行动量分析并给出操作建议。输出必须是 JSON，字段如下：
{{
  "action": "BUY|BUY_S|BUY_M|BUY_L|SELL|HOLD",
  "confidence": 0-1 之间的小数，表示信心,
  "summary": "一句话概括动量分析结论",
  "signals": ["动量信号要点", "..."],
  "risks": ["动量风险点", "..."]
}}
""",
        "variables": [
            "ts_code", "trade_date", "data_scope", 
            "features", "market_snapshot", "supplements"
        ],
        "required_context": [
            "ts_code", "trade_date", "features", "market_snapshot"
        ],
        "validation_rules": [
            "len(features) > 0",
            "'momentum' in ' '.join(features.keys()).lower()"
        ]
    }
}


def register_default_templates() -> None:
    """Register all default templates from DEFAULT_TEMPLATES."""
    for template_id, cfg in DEFAULT_TEMPLATES.items():
        template_config = {
            "id": template_id,
            "name": cfg.get("name", template_id),
            "description": cfg.get("description", ""),
            "template": cfg.get("template", ""),
            "variables": cfg.get("variables", []),
            "max_length": cfg.get("max_length", 4000),
            "required_context": cfg.get("required_context", []),
            "validation_rules": cfg.get("validation_rules", [])
        }
        try:
            template = PromptTemplate(**template_config)
            version_label = str(
                cfg.get("version") or TemplateRegistry._default_version_label
            )
            metadata_raw = cfg.get("metadata")
            metadata = dict(metadata_raw) if isinstance(metadata_raw, dict) else {}
            metadata.setdefault("source", "defaults")
            TemplateRegistry.register(
                template,
                version=version_label,
                metadata=metadata,
                activate=cfg.get("activate", True),
            )
        except ValueError as e:
            logging.warning(f"Failed to register template {template_id}: {e}")


# Auto-register default templates on module import
register_default_templates()
