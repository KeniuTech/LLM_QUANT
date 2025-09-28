"""Unified LLM client supporting Ollama and OpenAI compatible APIs."""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional

import requests

from app.utils.config import (
    DEFAULT_LLM_BASE_URLS,
    DEFAULT_LLM_MODELS,
    LLMConfig,
    LLMEndpoint,
    get_config,
)
from app.utils.logging import get_logger

LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "llm"}

class LLMError(RuntimeError):
    """Raised when LLM provider returns an error response."""


def _default_base_url(provider: str) -> str:
    provider = (provider or "openai").lower()
    return DEFAULT_LLM_BASE_URLS.get(provider, DEFAULT_LLM_BASE_URLS["openai"])


def _default_model(provider: str) -> str:
    provider = (provider or "").lower()
    return DEFAULT_LLM_MODELS.get(provider, DEFAULT_LLM_MODELS["ollama"])


def _build_messages(prompt: str, system: Optional[str] = None) -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def _request_ollama(
    model: str,
    prompt: str,
    *,
    base_url: str,
    temperature: float,
    timeout: float,
    system: Optional[str],
) -> str:
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


def _request_openai(
    model: str,
    prompt: str,
    *,
    base_url: str,
    api_key: str,
    temperature: float,
    timeout: float,
    system: Optional[str],
) -> str:
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


def _call_endpoint(endpoint: LLMEndpoint, prompt: str, system: Optional[str]) -> str:
    provider = (endpoint.provider or "ollama").lower()
    base_url = endpoint.base_url or _default_base_url(provider)
    model = endpoint.model or _default_model(provider)
    temperature = max(0.0, min(endpoint.temperature, 2.0))
    timeout = max(5.0, endpoint.timeout or 30.0)

    LOGGER.info(
        "触发 LLM 请求：provider=%s model=%s base=%s",
        provider,
        model,
        base_url,
        extra=LOG_EXTRA,
    )

    if provider in {"openai", "deepseek", "wenxin"}:
        api_key = endpoint.api_key
        if not api_key:
            raise LLMError(f"缺少 {provider} API Key (model={model})")
        return _request_openai(
            model,
            prompt,
            base_url=base_url,
            api_key=api_key,
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
    raise LLMError(f"不支持的 LLM provider: {endpoint.provider}")


def _normalize_response(text: str) -> str:
    return " ".join(text.strip().split())


def run_llm(prompt: str, *, system: Optional[str] = None) -> str:
    """Execute the globally configured LLM strategy with the given prompt."""

    settings = get_config().llm
    return run_llm_with_config(settings, prompt, system=system)


def _run_majority_vote(config: LLMConfig, prompt: str, system: Optional[str]) -> str:
    endpoints: List[LLMEndpoint] = [config.primary] + list(config.ensemble)
    responses: List[Dict[str, str]] = []
    failures: List[str] = []

    for idx, endpoint in enumerate(endpoints, start=1):
        try:
            result = _call_endpoint(endpoint, prompt, system)
            responses.append({
                "provider": endpoint.provider,
                "model": endpoint.model,
                "raw": result,
                "normalized": _normalize_response(result),
            })
        except Exception as exc:  # noqa: BLE001
            summary = f"{endpoint.provider}:{endpoint.model} -> {exc}"
            failures.append(summary)
            LOGGER.warning("LLM 调用失败：%s", summary, extra=LOG_EXTRA)

    if not responses:
        raise LLMError("所有 LLM 调用均失败，无法返回结果。")

    threshold = max(1, config.majority_threshold)
    threshold = min(threshold, len(responses))

    counter = Counter(item["normalized"] for item in responses)
    top_value, top_count = counter.most_common(1)[0]
    if top_count >= threshold:
        chosen_raw = next(item["raw"] for item in responses if item["normalized"] == top_value)
        LOGGER.info(
            "LLM 多模型投票通过：value=%s votes=%s/%s threshold=%s",
            top_value[:80],
            top_count,
            len(responses),
            threshold,
            extra=LOG_EXTRA,
        )
        return chosen_raw

    LOGGER.info(
        "LLM 多模型投票未达门槛：votes=%s/%s threshold=%s，返回首个结果",
        top_count,
        len(responses),
        threshold,
        extra=LOG_EXTRA,
    )
    if failures:
        LOGGER.warning("LLM 调用失败列表：%s", failures, extra=LOG_EXTRA)
    return responses[0]["raw"]


def _run_leader_follow(config: LLMConfig, prompt: str, system: Optional[str]) -> str:
    advisors: List[Dict[str, str]] = []
    for endpoint in config.ensemble:
        try:
            raw = _call_endpoint(endpoint, prompt, system)
            advisors.append(
                {
                    "provider": endpoint.provider,
                    "model": endpoint.model or "",
                    "raw": raw,
                }
            )
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "顾问模型调用失败：%s:%s -> %s",
                endpoint.provider,
                endpoint.model,
                exc,
                extra=LOG_EXTRA,
            )

    if not advisors:
        LOGGER.info("领导者策略顾问为空，回退至主模型", extra=LOG_EXTRA)
        return _call_endpoint(config.primary, prompt, system)

    advisor_chunks = []
    for idx, record in enumerate(advisors, start=1):
        snippet = record["raw"].strip()
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "..."
        advisor_chunks.append(
            f"顾问#{idx} ({record['provider']}:{record['model']}):\n{snippet}"
        )
    advisor_section = "\n\n".join(advisor_chunks)
    leader_prompt = (
        "【顾问模型意见】\n"
        f"{advisor_section}\n\n"
        "请在充分参考顾问模型观点的基础上，保持原始指令的输出格式进行最终回答。\n\n"
        f"{prompt}"
    )
    LOGGER.info(
        "领导者策略触发：顾问数量=%s",
        len(advisors),
        extra=LOG_EXTRA,
    )
    return _call_endpoint(config.primary, leader_prompt, system)


def run_llm_with_config(
    config: LLMConfig,
    prompt: str,
    *,
    system: Optional[str] = None,
) -> str:
    """Execute an LLM request using the provided configuration block."""

    strategy = (config.strategy or "single").lower()
    if strategy == "leader-follower":
        strategy = "leader"
    if strategy == "majority":
        return _run_majority_vote(config, prompt, system)
    if strategy == "leader":
        return _run_leader_follow(config, prompt, system)
    return _call_endpoint(config.primary, prompt, system)


def llm_config_snapshot() -> Dict[str, object]:
    """Return a sanitized snapshot of current LLM configuration for debugging."""

    settings = get_config().llm
    primary = asdict(settings.primary)
    if primary.get("api_key"):
        primary["api_key"] = "***"
    ensemble = []
    for endpoint in settings.ensemble:
        record = asdict(endpoint)
        if record.get("api_key"):
            record["api_key"] = "***"
        ensemble.append(record)
    return {
        "strategy": settings.strategy,
        "majority_threshold": settings.majority_threshold,
        "primary": primary,
        "ensemble": ensemble,
    }
