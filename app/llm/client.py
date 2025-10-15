"""Unified LLM client supporting Ollama and OpenAI compatible APIs."""
from __future__ import annotations

import json
from collections import Counter
import time
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional

import requests

from .context import ContextManager, Message
from .templates import TemplateRegistry
from .cost import configure_cost_limits, get_cost_controller, budget_available

from app.utils.config import (
    DEFAULT_LLM_BASE_URLS,
    DEFAULT_LLM_MODELS,
    DEFAULT_LLM_TEMPERATURES,
    DEFAULT_LLM_TIMEOUTS,
    LLMConfig,
    LLMEndpoint,
    get_config,
)
from app.llm.metrics import record_call, record_template_usage
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


def _build_messages(prompt: str, system: Optional[str] = None) -> List[Dict[str, object]]:
    messages: List[Dict[str, object]] = []
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


def _request_openai_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, object]],
    temperature: float,
    timeout: float,
    tools: Optional[List[Dict[str, object]]] = None,
    tool_choice: Optional[object] = None,
) -> Dict[str, object]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, object] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if tools:
        payload["tools"] = tools
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice
    LOGGER.debug("调用 OpenAI 兼容接口: %s %s", model, url, extra=LOG_EXTRA)
    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if response.status_code != 200:
        raise LLMError(f"OpenAI API 调用失败: {response.status_code} {response.text}")
    return response.json()


def resolve_endpoint(endpoint: LLMEndpoint) -> Dict[str, object]:
    cfg = get_config()
    provider_key = (endpoint.provider or "ollama").lower()
    provider_cfg = cfg.llm_providers.get(provider_key)

    base_url = endpoint.base_url
    api_key = endpoint.api_key
    model = endpoint.model
    temperature = endpoint.temperature
    timeout = endpoint.timeout
    prompt_template = endpoint.prompt_template

    if provider_cfg:
        if not provider_cfg.enabled:
            raise LLMError(f"Provider {provider_key} 已被禁用")
        base_url = base_url or provider_cfg.base_url or _default_base_url(provider_key)
        api_key = api_key or provider_cfg.api_key
        model = model or provider_cfg.default_model or (provider_cfg.models[0] if provider_cfg.models else _default_model(provider_key))
        if temperature is None:
            temperature = provider_cfg.default_temperature
        if timeout is None:
            timeout = provider_cfg.default_timeout
        prompt_template = prompt_template or (provider_cfg.prompt_template or None)
        mode = provider_cfg.mode or ("ollama" if provider_key == "ollama" else "openai")
    else:
        base_url = base_url or _default_base_url(provider_key)
        model = model or _default_model(provider_key)

        if temperature is None:
            temperature = DEFAULT_LLM_TEMPERATURES.get(provider_key, 0.2)
        if timeout is None:
            timeout = DEFAULT_LLM_TIMEOUTS.get(provider_key, 30.0)
        mode = "ollama" if provider_key == "ollama" else "openai"

    return {
        "provider_key": provider_key,
        "mode": mode,
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
        "temperature": max(0.0, min(float(temperature), 2.0)),
        "timeout": max(5.0, float(timeout)),
        "prompt_template": prompt_template,
    }


def _call_endpoint(endpoint: LLMEndpoint, prompt: str, system: Optional[str]) -> str:
    resolved = resolve_endpoint(endpoint)
    provider_key = resolved["provider_key"]
    mode = resolved["mode"]
    prompt_template = resolved["prompt_template"]

    if prompt_template:
        try:
            prompt = prompt_template.format(prompt=prompt)
        except Exception:  # noqa: BLE001
            LOGGER.warning("Prompt 模板格式化失败，使用原始 prompt", extra=LOG_EXTRA)

    messages = _build_messages(prompt, system)
    response = call_endpoint_with_messages(
        endpoint,
        messages,
        tools=None,
    )
    if mode == "ollama":
        message = response.get("message") or {}
        content = message.get("content", "")
        if isinstance(content, list):
            return "".join(chunk.get("text", "") or chunk.get("content", "") for chunk in content)
        return str(content)
    try:
        return response["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as exc:
        raise LLMError(f"OpenAI 响应解析失败: {json.dumps(response, ensure_ascii=False)}") from exc


def call_endpoint_with_messages(
    endpoint: LLMEndpoint,
    messages: List[Dict[str, object]],
    *,
    tools: Optional[List[Dict[str, object]]] = None,
    tool_choice: Optional[object] = None,
) -> Dict[str, object]:
    resolved = resolve_endpoint(endpoint)
    provider_key = resolved["provider_key"]
    mode = resolved["mode"]
    base_url = resolved["base_url"]
    model = resolved["model"]
    temperature = resolved["temperature"]
    timeout = resolved["timeout"]
    api_key = resolved["api_key"]

    cfg = get_config()
    cost_cfg = getattr(cfg, "llm_cost", None)
    enforce_cost = False
    cost_controller = None
    if cost_cfg and getattr(cost_cfg, "enabled", False):
        try:
            limits = cost_cfg.to_cost_limits()
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning(
                "成本控制配置解析失败，将忽略限制: %s",
                exc,
                extra=LOG_EXTRA,
            )
        else:
            configure_cost_limits(limits)
            enforce_cost = True
            if not budget_available():
                raise LLMError("LLM 调用预算已耗尽，请稍后重试。")
            cost_controller = get_cost_controller()

    LOGGER.info(
        "触发 LLM 请求：provider=%s model=%s base=%s",
        provider_key,
        model,
        base_url,
        extra=LOG_EXTRA,
    )

    if mode == "ollama":
        # Ollama supports function/tool calling via the /api/chat endpoint.
        # Include `tools` and optional `tool_choice` in the payload when provided.
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        if tools:
            # Ollama expects `tools` at the top level similar to OpenAI-compatible API
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

        start_time = time.perf_counter()
        response = requests.post(
            f"{base_url.rstrip('/')}/api/chat",
            json=payload,
            timeout=timeout,
        )
        duration = time.perf_counter() - start_time
        if response.status_code != 200:
            raise LLMError(f"Ollama 调用失败: {response.status_code} {response.text}")
        data = response.json()
        record_call(provider_key, model, duration=duration)
        if enforce_cost and cost_controller:
            cost_controller.record_usage(model or provider_key, 0, 0)
        # Ollama may return `tool_calls` under message.tool_calls when tools are used.
        # Return the raw response so callers can handle either OpenAI-like responses or
        # Ollama's message structure with `tool_calls`.
        return data

    if not api_key:
        raise LLMError(f"缺少 {provider_key} API Key (model={model})")
    start_time = time.perf_counter()
    data = _request_openai_chat(
        base_url=base_url,
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=temperature,
        timeout=timeout,
        tools=tools,
        tool_choice=tool_choice,
    )
    duration = time.perf_counter() - start_time
    usage = data.get("usage", {}) if isinstance(data, dict) else {}
    prompt_tokens = usage.get("prompt_tokens") or usage.get("prompt_tokens_total")
    completion_tokens = usage.get("completion_tokens") or usage.get("completion_tokens_total")
    record_call(
        provider_key,
        model,
        prompt_tokens,
        completion_tokens,
        duration=duration,
    )
    if enforce_cost and cost_controller:
        prompt_count = int(prompt_tokens or 0)
        completion_count = int(completion_tokens or 0)
        within_limits = cost_controller.record_usage(model or provider_key, prompt_count, completion_count)
        if not within_limits:
            LOGGER.warning(
                "LLM 成本预算已超限：provider=%s model=%s",
                provider_key,
                model,
                extra=LOG_EXTRA,
            )
    return data


def _normalize_response(text: str) -> str:
    return " ".join(text.strip().split())


def run_llm(
    prompt: str, 
    *,
    system: Optional[str] = None,
    context_id: Optional[str] = None,
    template_id: Optional[str] = None,
    template_vars: Optional[Dict[str, Any]] = None
) -> str:
    """Execute the globally configured LLM strategy with the given prompt.
    
    Args:
        prompt: Raw prompt string or template variable if template_id is provided
        system: Optional system message
        context_id: Optional context ID for conversation tracking
        template_id: Optional template ID to use
        template_vars: Variables to use with the template
    """
    # Get config and prepare context
    cfg = get_config()
    if context_id:
        context = ContextManager.get_context(context_id)
        if not context:
            context = ContextManager.create_context(context_id)
    else:
        context = None

    # Apply template if specified
    applied_template_version: Optional[str] = None
    if template_id:
        template = TemplateRegistry.get(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        applied_template_version = TemplateRegistry.get_active_version(template_id)
        vars_dict = template_vars or {}
        if isinstance(prompt, str):
            vars_dict["prompt"] = prompt
        elif isinstance(prompt, dict):
            vars_dict.update(prompt)
        prompt = template.format(vars_dict)

    # Add to context if tracking
    if context:
        if system:
            context.add_message(Message(role="system", content=system))
        context.add_message(Message(role="user", content=prompt))

    # Execute LLM call
    response = run_llm_with_config(cfg.llm, prompt, system=system)

    # Update context with response
    if context:
        context.add_message(Message(role="assistant", content=response))

    if template_id:
        record_template_usage(
            template_id,
            version=applied_template_version,
        )
    return response


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

    cfg = get_config()
    settings = cfg.llm
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
        "providers": {
            key: {
                "base_url": provider.base_url,
                "default_model": provider.default_model,
                "enabled": provider.enabled,
            }
            for key, provider in cfg.llm_providers.items()
        },
    }
