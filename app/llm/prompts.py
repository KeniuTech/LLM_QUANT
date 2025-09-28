"""Prompt templates for natural language outputs."""
from __future__ import annotations

from typing import Dict, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from app.utils.config import DepartmentSettings
    from app.agents.departments import DepartmentContext


def plan_prompt(data: Dict) -> str:
    """Build a concise instruction prompt for the LLM."""

    _ = data
    return "你是一个投资助理，请根据提供的数据给出三条要点和两条风险提示。"


def department_prompt(settings: "DepartmentSettings", context: "DepartmentContext") -> str:
    """Compose a structured prompt for department-level LLM ensemble."""

    feature_lines = "\n".join(
        f"- {key}: {value}" for key, value in sorted(context.features.items())
    )
    market_lines = "\n".join(
        f"- {key}: {value}" for key, value in sorted(context.market_snapshot.items())
    )
    scope_lines = "\n".join(f"- {item}" for item in settings.data_scope)
    role_description = settings.description.strip()
    role_instruction = settings.prompt.strip()

    instructions = f"""
部门名称：{settings.title}
股票代码：{context.ts_code}
交易日：{context.trade_date}

角色说明：{role_description or '未配置，默认沿用部门名称所代表的研究职责。'}
职责指令：{role_instruction or '在保持部门风格的前提下，结合可用数据做出审慎判断。'}

【可用数据范围】
{scope_lines or '- 使用系统提供的全部上下文，必要时指出仍需的额外数据。'}

【核心特征】
{feature_lines or '- (无)'}

【市场背景】
{market_lines or '- (无)'}

请基于以上数据给出该部门对当前股票的操作建议。输出必须是 JSON，字段如下：
{{
  "action": "BUY|BUY_S|BUY_M|BUY_L|SELL|HOLD",
  "confidence": 0-1 之间的小数，表示信心，
  "summary": "一句话概括理由",
  "signals": ["详细要点", "..."],
  "risks": ["风险点", "..."]
}}

请严格返回单个 JSON 对象，不要添加额外文本。
"""
    return instructions.strip()
