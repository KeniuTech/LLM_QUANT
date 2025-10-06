"""LLM prompt templates management with configuration driven design."""
from __future__ import annotations

import json
import logging
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
        "description": "所有部门通用的审慎分析提示词骨架",
        "template": """
部门：{title}
股票代码：{ts_code}
交易日：{trade_date}

【角色定位】
- 角色说明：{description}
- 行动守则：{instruction}

【数据边界】
- 可用字段：
{data_scope}
- 核心特征：
{features}
- 市场背景：
{market_snapshot}
- 追加数据：
{supplements}

【分析步骤】
1. 判断信息是否充分，如不充分，请说明缺口并优先调用工具 `fetch_data`（仅限 `daily`、`daily_basic`）。
2. 梳理 2-3 个关键支撑信号与潜在风险，确保基于提供的数据。
3. 结合量化证据与限制条件，给出操作建议和信心来源，避免主观臆测。

【输出要求】
仅返回一个 JSON 对象，不要添加额外文本：
{{
  "action": "BUY|BUY_S|BUY_M|BUY_L|SELL|HOLD",
  "confidence": 0-1 之间的小数，
  "summary": "一句话结论",
  "signals": ["关键支撑要点", "..."],
  "risks": ["关键风险要点", "..."]
}}
如需说明未完成的数据请求，请在 `risks` 或 `signals` 中明确。
""",
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
        "metadata": {
            "category": "department",
            "preset": "base",
        },
    },
    "momentum_dept": {
        "name": "动量研究部门模板",
        "description": "围绕价格与量能动量的决策提示",
        "template": """
部门：动量研究部门
股票代码：{ts_code}
交易日：{trade_date}

【角色定位】
- 专注价格动量、成交量共振与技术指标背离。
- 保持纪律，识别趋势延续与反转风险。

【研究重点】
1. 多时间窗口动量是否同向？
2. 成交量是否验证价格走势？
3. 是否出现过热或背离信号？

【数据边界】
- 可用字段：
{data_scope}
- 动量特征：
{features}
- 市场背景：
{market_snapshot}
- 追加数据：
{supplements}

请沿用【部门基础模板】的分析步骤与输出要求，重点量化趋势动能和量价配合度。
""",
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
        "metadata": {
            "category": "department",
            "preset": "momentum",
        },
    },
    "value_dept": {
        "name": "价值评估部门模板",
        "description": "衡量估值与盈利质量的提示词",
        "template": """
部门：价值评估部门
股票代码：{ts_code}
交易日：{trade_date}

【角色定位】
- 关注估值分位、盈利质量与安全边际。
- 从中期配置角度评价当前价格的性价比。

【研究重点】
1. 历史及同业视角的估值位置。
2. 盈利与分红的可持续性。
3. 潜在的估值修复催化或压制因素。

【数据边界】
- 可用字段：
{data_scope}
- 估值与质量特征：
{features}
- 市场背景：
{market_snapshot}
- 追加数据：
{supplements}

请按照【部门基础模板】的分析步骤输出结论，并明确估值安全边际来源。
""",
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
        "metadata": {
            "category": "department",
            "preset": "value",
        },
    },
    "news_dept": {
        "name": "新闻情绪部门模板",
        "description": "针对舆情热度与事件影响的提示词",
        "template": """
部门：新闻情绪部门
股票代码：{ts_code}
交易日：{trade_date}

【角色定位】
- 监控舆情热度、事件驱动与短期情绪。
- 评估新闻对价格波动的正负面影响。

【研究重点】
1. 新闻情绪是否集中且持续？
2. 主题与行情是否匹配？
3. 情绪驱动的风险敞口。

【数据边界】
- 可用字段：
{data_scope}
- 舆情特征：
{features}
- 市场背景：
{market_snapshot}
- 追加数据：
{supplements}

请遵循【部门基础模板】的分析步骤，突出情绪驱动的力度与时效性。
""",
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
        "metadata": {
            "category": "department",
            "preset": "news",
        },
    },
    "liquidity_dept": {
        "name": "流动性评估部门模板",
        "description": "衡量成交活跃度与执行成本的提示词",
        "template": """
部门：流动性评估部门
股票代码：{ts_code}
交易日：{trade_date}

【角色定位】
- 评估成交活跃度、交易成本与可执行性。
- 提醒潜在的流动性风险与仓位限制。

【研究重点】
1. 当前成交量与历史均值的对比。
2. 价量限制（涨跌停、停牌等）对执行的影响。
3. 预估滑点与转手难度。

【数据边界】
- 可用字段：
{data_scope}
- 流动性特征：
{features}
- 市场背景：
{market_snapshot}
- 追加数据：
{supplements}

请遵循【部门基础模板】的分析步骤，重点描述执行可行性与仓位建议。
""",
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
        "metadata": {
            "category": "department",
            "preset": "liquidity",
        },
    },
    "macro_dept": {
        "name": "宏观研究部门模板",
        "description": "宏观与行业景气度分析提示词",
        "template": """
部门：宏观研究部门
股票代码：{ts_code}
交易日：{trade_date}

【角色定位】
- 追踪宏观周期、行业景气与相对强弱。
- 评估宏观事件对该标的的方向性影响。

【研究重点】
1. 行业相对大盘的表现与热点程度。
2. 宏观/政策事件对行业或标的的指引。
3. 需警惕的宏观风险与流动性环境。

【数据边界】
- 可用字段：
{data_scope}
- 宏观特征：
{features}
- 市场背景：
{market_snapshot}
- 追加数据：
{supplements}

请执行【部门基础模板】的分析步骤，并输出宏观驱动的信号与风险。
""",
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
        "metadata": {
            "category": "department",
            "preset": "macro",
        },
    },
    "risk_dept": {
        "name": "风险控制部门模板",
        "description": "识别极端风险与限制条件的提示词",
        "template": """
部门：风险控制部门
股票代码：{ts_code}
交易日：{trade_date}

【角色定位】
- 防范停牌、涨跌停、仓位与合规限制。
- 必要时对高风险决策行使否决权。

【研究重点】
1. 交易限制或异常波动情况。
2. 仓位、集中度或风险指标是否触顶。
3. 潜在的黑天鹅或执行障碍。

【数据边界】
- 可用字段：
{data_scope}
- 风险特征：
{features}
- 市场背景：
{market_snapshot}
- 追加数据：
{supplements}

请按照【部门基础模板】的分析步骤，必要时明确阻止交易的理由。
""",
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
        "metadata": {
            "category": "department",
            "preset": "risk",
        },
    },
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
EXTERNAL_TEMPLATE_DIR = Path(__file__).resolve().parents[1] / "data" / "prompt_templates"


def load_external_template_configs(directory: Path | str = EXTERNAL_TEMPLATE_DIR) -> None:
    """Load additional template versions from JSON files in the given directory."""

    directory_path = Path(directory)
    if not directory_path.exists() or not directory_path.is_dir():
        return

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
        except Exception as exc:  # noqa: BLE001
            logging.warning(
                "注册提示模板配置 %s 失败：%s",
                file_path,
                exc,
            )


register_default_templates()
load_external_template_configs()
