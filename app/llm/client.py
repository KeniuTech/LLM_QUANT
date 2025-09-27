"""Unified LLM client supporting Ollama and OpenAI compatible APIs."""
from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional

import requests

from app.utils.config import get_config
from app.utils.logging import get_logger

LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "llm"}


class LLMError(RuntimeError):
    """Raised when LLM provider returns an error response."""


def _default_base_url(provider: str) -> str:
    if provider == "ollama":
        return "http://localhost:11434"
    return "https://api.openai.com"


def _build_messages(prompt: str, system: Optional[str] = None) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _request_ollama(model: str, prompt: str, *, base_url: str, temperature: float, timeout: float, system: Optional[str]) -> str:
    url = f"{base_url.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "messages": _build_messages(prompt, system),
        "stream": False,
        "options": {"temperature": temperature},
    }
    LOGGER.debug("调用 Ollama: %s %s", model, url, extra=LOG_EXTRA)
    response = requests.post(url, json=payload, timeout=timeout)
    if response.status_code != 200:
        raise LLMError(f"Ollama 调用失败: {response.status_code} {response.text}")
    data = response.json()
    message = data.get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        return "".join(chunk.get("text", "") or chunk.get("content", "") for chunk in content)
    return str(content)


def _request_openai(model: str, prompt: str, *, base_url: str, api_key: str, temperature: float, timeout: float, system: Optional[str]) -> str:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": _build_messages(prompt, system),
        "temperature": temperature,
    }
    LOGGER.debug("调用 OpenAI 兼容接口: %s %s", model, url, extra=LOG_EXTRA)
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if response.status_code != 200:
        raise LLMError(f"OpenAI API 调用失败: {response.status_code} {response.text}")
    data = response.json()
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        raise LLMError(f"OpenAI 响应解析失败: {json.dumps(data, ensure_ascii=False)}") from exc


def run_llm(prompt: str, *, system: Optional[str] = None) -> str:
    """Execute the configured LLM provider with the given prompt."""

    cfg = get_config().llm
    provider = (cfg.provider or "ollama").lower()
    base_url = cfg.base_url or _default_base_url(provider)
    model = cfg.model
    temperature = max(0.0, min(cfg.temperature, 2.0))
    timeout = max(5.0, cfg.timeout or 30.0)

    LOGGER.info(
        "触发 LLM 请求：provider=%s model=%s base=%s", provider, model, base_url, extra=LOG_EXTRA
    )

    if provider == "openai":
        if not cfg.api_key:
            raise LLMError("缺少 OpenAI 兼容 API Key")
        return _request_openai(
            model,
            prompt,
            base_url=base_url,
            api_key=cfg.api_key,
            temperature=temperature,
            timeout=timeout,
            system=system,
        )
    if provider == "ollama":
        return _request_ollama(
            model,
            prompt,
            base_url=base_url,
            temperature=temperature,
            timeout=timeout,
            system=system,
        )
    raise LLMError(f"不支持的 LLM provider: {cfg.provider}")


def llm_config_snapshot() -> Dict[str, object]:
    """Return a sanitized snapshot of current LLM configuration for debugging."""

    cfg = get_config().llm
    data = asdict(cfg)
    if data.get("api_key"):
        data["api_key"] = "***"
    return data
