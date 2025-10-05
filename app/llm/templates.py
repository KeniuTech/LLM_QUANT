"""LLM prompt templates management with configuration driven design."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


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

        # Truncate if needed
        if len(result) > self.max_length:
            result = result[:self.max_length-3] + "..."

        return result


class TemplateRegistry:
    """Global registry for prompt templates."""

    _templates: Dict[str, PromptTemplate] = {}

    @classmethod
    def register(cls, template: PromptTemplate) -> None:
        """Register a new template."""
        errors = template.validate()
        if errors:
            raise ValueError(f"Invalid template {template.id}: {'; '.join(errors)}")
        cls._templates[template.id] = template

    @classmethod
    def get(cls, template_id: str) -> Optional[PromptTemplate]:
        """Get template by ID."""
        return cls._templates.get(template_id)

    @classmethod
    def list(cls) -> List[PromptTemplate]:
        """List all registered templates."""
        return list(cls._templates.values())

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
            cls.register(template)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered templates."""
        cls._templates.clear()


# Register default templates
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

# Register default templates
for template_id, cfg in DEFAULT_TEMPLATES.items():
    TemplateRegistry.register(PromptTemplate(**{"id": template_id, **cfg}))
