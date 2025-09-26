"""LLM assisted explanations and summaries."""
from __future__ import annotations

from typing import Dict

from .prompts import plan_prompt


def make_human_card(ts_code: str, trade_date: str, context: Dict) -> Dict:
    """Compose payload for UI cards and LLM requests."""

    prompt = plan_prompt(context)
    return {
        "ts_code": ts_code,
        "trade_date": trade_date,
        "prompt": prompt,
        "context": context,
    }
