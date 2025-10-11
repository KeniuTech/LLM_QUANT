"""Prompt templates for natural language outputs."""
from __future__ import annotations

import logging
from typing import Dict, TYPE_CHECKING

from .templates import TemplateRegistry

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from app.utils.config import DepartmentSettings
    from app.agents.departments import DepartmentContext


def plan_prompt(data: Dict) -> str:
    """Build a concise instruction prompt for the LLM."""

    _ = data
    return "你是一个投资助理，请根据提供的数据给出三条要点和两条风险提示。"


def department_prompt(
    settings: "DepartmentSettings",
    context: "DepartmentContext",
    supplements: str = "",
) -> str:
    """Compose a structured prompt for department-level LLM ensemble."""
    
    # Format data for template
    feature_lines = "\n".join(
        f"- {key}: {value}" for key, value in sorted(context.features.items())
    )
    market_lines = "\n".join(
        f"- {key}: {value}" for key, value in sorted(context.market_snapshot.items())
    )
    scope_lines = "\n".join(f"- {item}" for item in settings.data_scope)
    role_description = settings.description.strip()
    role_instruction = settings.prompt.strip()
    
    # Determine template ID and version
    template_id = (getattr(settings, "prompt_template_id", None) or f"{settings.code.lower()}_dept").strip()
    requested_version = getattr(settings, "prompt_template_version", None)
    original_requested_version = requested_version
    template = TemplateRegistry.get(template_id, version=requested_version)
    applied_version = requested_version if template and requested_version else None

    if not template:
        if requested_version:
            LOGGER.warning(
                "Template %s version %s not found, falling back to active version",
                template_id,
                requested_version,
            )
        template = TemplateRegistry.get(template_id)
        applied_version = TemplateRegistry.get_active_version(template_id)

    if not template:
        LOGGER.warning(
            "Template %s unavailable, using department_base fallback",
            template_id,
        )
        template_id = "department_base"
        template = TemplateRegistry.get(template_id)
        requested_version = None
        applied_version = TemplateRegistry.get_active_version(template_id)

    if not template:
        raise ValueError("No prompt template available for department prompts")

    if applied_version is None:
        applied_version = TemplateRegistry.get_active_version(template_id)
    template_meta = {
        "template_id": template_id,
        "requested_version": original_requested_version,
        "applied_version": applied_version,
    }

    raw_container = getattr(context, "raw", None)
    if isinstance(raw_container, dict):
        meta_store = raw_container.setdefault("template_meta", {})
        meta_store[settings.code] = template_meta

    # Prepare template variables
    template_vars = {
        "title": settings.title,
        "ts_code": context.ts_code,
        "trade_date": context.trade_date,
        "description": role_description or "未配置，默认沿用部门名称所代表的研究职责。",
        "instruction": role_instruction or "在保持部门风格的前提下，结合可用数据做出审慎判断。",
        "data_scope": scope_lines or "- 使用系统提供的全部上下文，必要时指出仍需的额外数据。",
        "features": feature_lines or "- (无)",
        "market_snapshot": market_lines or "- (无)",
        "supplements": supplements.strip() or "- 当前无追加数据",
        "action": ""  # 添加 action 变量以避免模板格式化错误
    }
    template_vars.setdefault("scratchpad", "")

    # Ensure all declared template variables exist to avoid KeyError
    try:
        declared_vars = list(getattr(template, "variables", []) or [])
    except Exception:  # noqa: BLE001
        declared_vars = []
    for var in declared_vars:
        template_vars.setdefault(var, "")
    
    # Get template and format prompt
    return template.format(template_vars)
