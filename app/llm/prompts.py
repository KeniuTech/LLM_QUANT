"""Prompt templates for natural language outputs."""
from __future__ import annotations

from typing import Dict


def plan_prompt(data: Dict) -> str:
    """Build a concise instruction prompt for the LLM."""

    _ = data
    return "你是一个投资助理，请根据提供的数据给出三条要点和两条风险提示。"
