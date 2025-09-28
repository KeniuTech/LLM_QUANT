"""Department-level LLM agents coordinating multi-model decisions."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from app.agents.base import AgentAction
from app.llm.client import call_endpoint_with_messages, run_llm_with_config, LLMError
from app.llm.prompts import department_prompt
from app.utils.config import AppConfig, DepartmentSettings, LLMConfig
from app.utils.logging import get_logger
from app.utils.data_access import DataBroker

LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "department"}


@dataclass
class DataRequest:
    field: str
    window: int = 1


@dataclass
class DepartmentContext:
    """Structured data fed into a department for decision making."""

    ts_code: str
    trade_date: str
    features: Mapping[str, Any] = field(default_factory=dict)
    market_snapshot: Mapping[str, Any] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class DepartmentDecision:
    """Result produced by a department agent."""

    department: str
    action: AgentAction
    confidence: float
    summary: str
    raw_response: str
    signals: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    supplements: List[Dict[str, Any]] = field(default_factory=list)
    dialogue: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "department": self.department,
            "action": self.action.value,
            "confidence": self.confidence,
            "summary": self.summary,
            "signals": self.signals,
            "risks": self.risks,
            "raw_response": self.raw_response,
            "supplements": self.supplements,
            "dialogue": self.dialogue,
        }


class DepartmentAgent:
    """Wraps LLM ensemble logic for a single analytical department."""

    def __init__(
        self,
        settings: DepartmentSettings,
        resolver: Optional[Callable[[DepartmentSettings], LLMConfig]] = None,
    ) -> None:
        self.settings = settings
        self._resolver = resolver
        self._broker = DataBroker()
        self._max_rounds = 3

    def _get_llm_config(self) -> LLMConfig:
        if self._resolver:
            return self._resolver(self.settings)
        return self.settings.llm

    def analyze(self, context: DepartmentContext) -> DepartmentDecision:
        mutable_context = _ensure_mutable_context(context)
        system_prompt = (
            "你是一个多智能体量化投研系统中的分部决策者，需要根据提供的结构化信息给出买卖意见。"
        )
        llm_cfg = self._get_llm_config()

        if llm_cfg.strategy not in (None, "", "single") or llm_cfg.ensemble:
            LOGGER.warning(
                "部门 %s 当前配置不支持函数调用模式，回退至传统提示",
                self.settings.code,
                extra=LOG_EXTRA,
            )
            return self._analyze_legacy(mutable_context, system_prompt)

        tools = self._build_tools()
        messages: List[Dict[str, object]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append(
            {
                "role": "user",
                "content": department_prompt(self.settings, mutable_context),
            }
        )

        transcript: List[str] = []
        delivered_requests: set[Tuple[str, int]] = {
            (field, 1)
            for field in (mutable_context.raw.get("scope_values") or {}).keys()
        }

        primary_endpoint = llm_cfg.primary
        final_message: Optional[Dict[str, Any]] = None

        for round_idx in range(self._max_rounds):
            try:
                response = call_endpoint_with_messages(
                    primary_endpoint,
                    messages,
                    tools=tools,
                    tool_choice="auto",
                )
            except LLMError as exc:
                LOGGER.warning(
                    "部门 %s 函数调用失败，回退传统提示：%s",
                    self.settings.code,
                    exc,
                    extra=LOG_EXTRA,
                )
                return self._analyze_legacy(mutable_context, system_prompt)

            choice = (response.get("choices") or [{}])[0]
            message = choice.get("message", {})
            transcript.append(_message_to_text(message))

            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                for call in tool_calls:
                    tool_response, delivered = self._handle_tool_call(
                        mutable_context,
                        call,
                        delivered_requests,
                        round_idx,
                    )
                    transcript.append(
                        json.dumps({"tool_response": tool_response}, ensure_ascii=False)
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id"),
                            "name": call.get("function", {}).get("name"),
                            "content": json.dumps(tool_response, ensure_ascii=False),
                        }
                    )
                    delivered_requests.update(delivered)
                continue

            final_message = message
            break

        if final_message is None:
            LOGGER.warning(
                "部门 %s 函数调用达到轮次上限仍未返回文本，使用最后一次消息",
                self.settings.code,
                extra=LOG_EXTRA,
            )
            final_message = message

        mutable_context.raw["supplement_transcript"] = list(transcript)

        content_text = _extract_message_content(final_message)
        decision_data = _parse_department_response(content_text)

        action = _normalize_action(decision_data.get("action"))
        confidence = _clamp_float(decision_data.get("confidence"), default=0.5)
        summary = decision_data.get("summary") or decision_data.get("reason") or ""
        signals = decision_data.get("signals") or decision_data.get("rationale") or []
        if isinstance(signals, str):
            signals = [signals]
        risks = decision_data.get("risks") or decision_data.get("warnings") or []
        if isinstance(risks, str):
            risks = [risks]

        decision = DepartmentDecision(
            department=self.settings.code,
            action=action,
            confidence=confidence,
            summary=summary or "未提供摘要",
            signals=[str(sig) for sig in signals if sig],
            risks=[str(risk) for risk in risks if risk],
            raw_response=content_text,
            supplements=list(mutable_context.raw.get("supplement_data", [])),
            dialogue=list(transcript),
        )
        LOGGER.debug(
            "部门 %s 决策：action=%s confidence=%.2f",
            self.settings.code,
            decision.action.value,
            decision.confidence,
            extra=LOG_EXTRA,
        )
        return decision

    @staticmethod
    def _normalize_trade_date(value: str) -> str:
        if not isinstance(value, str):
            return str(value)
        return value.replace("-", "")

    def _fulfill_data_requests(
        self,
        context: DepartmentContext,
        requests: Sequence[DataRequest],
    ) -> Tuple[List[str], List[Dict[str, Any]], set[Tuple[str, int]]]:
        lines: List[str] = []
        payload: List[Dict[str, Any]] = []
        delivered: set[Tuple[str, int]] = set()

        ts_code = context.ts_code
        trade_date = self._normalize_trade_date(context.trade_date)

        latest_groups: Dict[str, List[str]] = {}
        series_requests: List[Tuple[DataRequest, Tuple[str, str]]] = []
        values_map, db_alias_map, series_map = _build_context_lookup(context)

        for req in requests:
            field = req.field.strip()
            if not field:
                continue
            window = req.window
            resolved: Optional[Tuple[str, str]] = None
            if "." in field:
                resolved = self._broker.resolve_field(field)
            elif field in db_alias_map:
                resolved = db_alias_map[field]

            if resolved:
                table, column = resolved
                canonical = f"{table}.{column}"
                if window <= 1:
                    latest_groups.setdefault(canonical, []).append(field)
                    delivered.add((field, 1))
                    delivered.add((canonical, 1))
                else:
                    series_requests.append((req, resolved))
                    delivered.add((field, window))
                    delivered.add((canonical, window))
                continue

            if field in values_map:
                value = values_map[field]
                if window <= 1:
                    payload.append(
                        {
                            "field": field,
                            "window": 1,
                            "source": "context",
                            "values": [
                                {
                                    "trade_date": context.trade_date,
                                    "value": value,
                                }
                            ],
                        }
                    )
                    lines.append(f"- {field}: {value} (来自上下文)")
                else:
                    series = series_map.get(field)
                    if series:
                        trimmed = series[: window]
                        payload.append(
                            {
                                "field": field,
                                "window": window,
                                "source": "context_series",
                                "values": [
                                    {"trade_date": dt, "value": val}
                                    for dt, val in trimmed
                                ],
                            }
                        )
                        preview = ", ".join(
                            f"{dt}:{val:.4f}" for dt, val in trimmed[: min(len(trimmed), 5)]
                        )
                        lines.append(
                            f"- {field} (window={window} 来自上下文序列): {preview}"
                        )
                    else:
                        payload.append(
                            {
                                "field": field,
                                "window": window,
                                "source": "context",
                                "values": [
                                    {
                                        "trade_date": context.trade_date,
                                        "value": value,
                                    }
                                ],
                                "warning": "仅提供当前值，缺少历史序列",
                            }
                        )
                        lines.append(
                            f"- {field} (window={window}): 仅有当前值 {value}, 无历史序列"
                        )
                delivered.add((field, window))
                if field in db_alias_map:
                    resolved = db_alias_map[field]
                    canonical = f"{resolved[0]}.{resolved[1]}"
                    delivered.add((canonical, window))
                continue

            lines.append(f"- {field}: 字段不存在或不可用")

        if latest_groups:
            latest_values = self._broker.fetch_latest(
                ts_code, trade_date, list(latest_groups.keys())
            )
            for canonical, aliases in latest_groups.items():
                value = latest_values.get(canonical)
                if value is None:
                    lines.append(f"- {canonical}: (数据缺失)")
                else:
                    lines.append(f"- {canonical}: {value}")
                for alias in aliases:
                    payload.append(
                        {
                            "field": alias,
                            "window": 1,
                            "source": "database",
                            "values": [
                                {
                                    "trade_date": trade_date,
                                    "value": value,
                                }
                            ],
                        }
                    )

        for req, resolved in series_requests:
            table, column = resolved
            series = self._broker.fetch_series(
                table,
                column,
                ts_code,
                trade_date,
                window=req.window,
            )
            if series:
                preview = ", ".join(
                    f"{dt}:{val:.4f}"
                    for dt, val in series[: min(len(series), 5)]
                )
                lines.append(
                    f"- {req.field} (window={req.window}): {preview}"
                )
            else:
                lines.append(
                    f"- {req.field} (window={req.window}): (数据缺失)"
                )
            payload.append(
                {
                    "field": req.field,
                    "window": req.window,
                    "source": "database",
                    "values": [
                        {"trade_date": dt, "value": val}
                        for dt, val in series
                    ],
                }
            )

        return lines, payload, delivered


    def _handle_tool_call(
        self,
        context: DepartmentContext,
        call: Mapping[str, Any],
        delivered_requests: set[Tuple[str, int]],
        round_idx: int,
    ) -> Tuple[Dict[str, Any], set[Tuple[str, int]]]:
        function_block = call.get("function") or {}
        name = function_block.get("name") or ""
        if name != "fetch_data":
            LOGGER.warning(
                "部门 %s 收到未知工具调用：%s",
                self.settings.code,
                name,
                extra=LOG_EXTRA,
            )
            return {
                "status": "error",
                "message": f"未知工具 {name}",
            }, set()

        args = _parse_tool_arguments(function_block.get("arguments"))
        raw_requests = args.get("requests") or []
        requests: List[DataRequest] = []
        skipped: List[str] = []
        for item in raw_requests:
            field = str(item.get("field", "")).strip()
            if not field:
                continue
            try:
                window = int(item.get("window", 1))
            except (TypeError, ValueError):
                window = 1
            window = max(1, min(window, getattr(self._broker, "MAX_WINDOW", 120)))
            key = (field, window)
            if key in delivered_requests:
                skipped.append(field)
                continue
            requests.append(DataRequest(field=field, window=window))

        if not requests:
            return {
                "status": "ok",
                "round": round_idx + 1,
                "results": [],
                "skipped": skipped,
            }, set()

        lines, payload, delivered = self._fulfill_data_requests(context, requests)
        if payload:
            context.raw.setdefault("supplement_data", []).extend(payload)
            context.raw.setdefault("supplement_rounds", []).append(
                {
                    "round": round_idx + 1,
                    "requests": [req.__dict__ for req in requests],
                    "data": payload,
                    "notes": lines,
                }
            )
        if lines:
            context.raw.setdefault("supplement_notes", []).append(
                {
                    "round": round_idx + 1,
                    "lines": lines,
                }
            )

        response_payload = {
            "status": "ok",
            "round": round_idx + 1,
            "results": payload,
            "notes": lines,
            "skipped": skipped,
        }
        return response_payload, delivered


    def _build_tools(self) -> List[Dict[str, Any]]:
        max_window = getattr(self._broker, "MAX_WINDOW", 120)
        return [
            {
                "type": "function",
                "function": {
                    "name": "fetch_data",
                    "description": (
                        "根据字段请求数据库中的最新值或时间序列。支持 table.column 格式的字段，"
                        "window 表示希望返回的最近数据点数量。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "requests": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "field": {
                                            "type": "string",
                                            "description": "数据字段，格式为 table.column",
                                        },
                                        "window": {
                                            "type": "integer",
                                            "minimum": 1,
                                            "maximum": max_window,
                                            "description": "返回最近多少个数据点，默认为 1",
                                        },
                                    },
                                    "required": ["field"],
                                },
                                "minItems": 1,
                            }
                        },
                        "required": ["requests"],
                    },
                },
            }
        ]

    def _analyze_legacy(
        self,
        context: DepartmentContext,
        system_prompt: str,
    ) -> DepartmentDecision:
        prompt = department_prompt(self.settings, context)
        llm_cfg = self._get_llm_config()
        try:
            response = run_llm_with_config(llm_cfg, prompt, system=system_prompt)
        except Exception as exc:  # noqa: BLE001
            LOGGER.exception(
                "部门 %s 调用 LLM 失败：%s",
                self.settings.code,
                exc,
                extra=LOG_EXTRA,
            )
            return DepartmentDecision(
                department=self.settings.code,
                action=AgentAction.HOLD,
                confidence=0.0,
                summary=f"LLM 调用失败：{exc}",
                raw_response=str(exc),
            )

        context.raw["supplement_transcript"] = [response]
        decision_data = _parse_department_response(response)
        action = _normalize_action(decision_data.get("action"))
        confidence = _clamp_float(decision_data.get("confidence"), default=0.5)
        summary = decision_data.get("summary") or decision_data.get("reason") or ""
        signals = decision_data.get("signals") or decision_data.get("rationale") or []
        if isinstance(signals, str):
            signals = [signals]
        risks = decision_data.get("risks") or decision_data.get("warnings") or []
        if isinstance(risks, str):
            risks = [risks]

        decision = DepartmentDecision(
            department=self.settings.code,
            action=action,
            confidence=confidence,
            summary=summary or "未提供摘要",
            signals=[str(sig) for sig in signals if sig],
            risks=[str(risk) for risk in risks if risk],
            raw_response=response,
            supplements=list(context.raw.get("supplement_data", [])),
            dialogue=[response],
        )
        return decision
def _ensure_mutable_context(context: DepartmentContext) -> DepartmentContext:
    if not isinstance(context.features, dict):
        context.features = dict(context.features or {})
    if not isinstance(context.market_snapshot, dict):
        context.market_snapshot = dict(context.market_snapshot or {})
    raw = dict(context.raw or {})
    scope_values = raw.get("scope_values")
    if scope_values is not None and not isinstance(scope_values, dict):
        raw["scope_values"] = dict(scope_values)
    context.raw = raw
    return context


def _parse_data_requests(payload: Mapping[str, Any]) -> List[DataRequest]:
    raw_requests = payload.get("data_requests")
    requests: List[DataRequest] = []
    if not isinstance(raw_requests, list):
        return requests
    seen: set[Tuple[str, int]] = set()
    for item in raw_requests:
        field = ""
        window = 1
        if isinstance(item, str):
            field = item.strip()
        elif isinstance(item, Mapping):
            candidate = item.get("field")
            if candidate is None:
                continue
            field = str(candidate).strip()
            try:
                window = int(item.get("window", 1))
            except (TypeError, ValueError):
                window = 1
        else:
            continue
        if not field:
            continue
        window = max(1, window)
        key = (field, window)
        if key in seen:
            continue
        seen.add(key)
        requests.append(DataRequest(field=field, window=window))
    return requests


def _parse_tool_arguments(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        return dict(payload)
    if isinstance(payload, str):
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            LOGGER.debug("工具参数解析失败：%s", payload, extra=LOG_EXTRA)
            return {}
        if isinstance(data, dict):
            return data
    return {}


def _message_to_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, Mapping) and "text" in item:
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        if parts:
            return "".join(parts)
    elif isinstance(content, str) and content.strip():
        return content
    tool_calls = message.get("tool_calls")
    if tool_calls:
        return json.dumps({"tool_calls": tool_calls}, ensure_ascii=False)
    return ""


def _extract_message_content(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, list):
        texts = [
            str(item.get("text", ""))
            for item in content
            if isinstance(item, Mapping) and "text" in item
        ]
        if texts:
            return "".join(texts)
    if isinstance(content, str):
        return content
    return json.dumps(message, ensure_ascii=False)


def _build_context_lookup(
    context: DepartmentContext,
) -> Tuple[Dict[str, Any], Dict[str, Tuple[str, str]], Dict[str, List[Tuple[str, float]]]]:
    values: Dict[str, Any] = {}
    db_alias: Dict[str, Tuple[str, str]] = {}
    series_map: Dict[str, List[Tuple[str, float]]] = {}

    for source in (context.features or {}, context.market_snapshot or {}):
        for key, value in source.items():
            values[str(key)] = value

    scope_values = context.raw.get("scope_values") or {}
    for key, value in scope_values.items():
        key_str = str(key)
        values[key_str] = value
        if "." in key_str:
            table, column = key_str.split(".", 1)
            db_alias.setdefault(column, (table, column))
            db_alias.setdefault(key_str, (table, column))
            values.setdefault(column, value)

    close_series = context.raw.get("close_series") or []
    if isinstance(close_series, list) and close_series:
        series_map["close"] = close_series
        series_map["daily.close"] = close_series

    turnover_series = context.raw.get("turnover_series") or []
    if isinstance(turnover_series, list) and turnover_series:
        series_map["turnover_rate"] = turnover_series
        series_map["daily_basic.turnover_rate"] = turnover_series

    return values, db_alias, series_map


class DepartmentManager:
    """Orchestrates all departments defined in configuration."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.agents: Dict[str, DepartmentAgent] = {
            code: DepartmentAgent(settings, self._resolve_llm)
            for code, settings in config.departments.items()
        }

    def evaluate(self, context: DepartmentContext) -> Dict[str, DepartmentDecision]:
        results: Dict[str, DepartmentDecision] = {}
        for code, agent in self.agents.items():
            raw_base = dict(context.raw or {})
            if "scope_values" in raw_base:
                raw_base["scope_values"] = dict(raw_base.get("scope_values") or {})
            dept_context = DepartmentContext(
                ts_code=context.ts_code,
                trade_date=context.trade_date,
                features=dict(context.features or {}),
                market_snapshot=dict(context.market_snapshot or {}),
                raw=raw_base,
            )
            results[code] = agent.analyze(dept_context)
        return results

    def _resolve_llm(self, settings: DepartmentSettings) -> LLMConfig:
        return settings.llm


def _parse_department_response(text: str) -> Dict[str, Any]:
    """Extract a JSON object from the LLM response if possible."""

    cleaned = text.strip()
    candidate = None
    if cleaned.startswith("{") and cleaned.endswith("}"):
        candidate = cleaned
    else:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
    if candidate:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            LOGGER.debug("部门响应 JSON 解析失败，返回原始文本", extra=LOG_EXTRA)
    return {"summary": cleaned}


def _normalize_action(value: Any) -> AgentAction:
    if isinstance(value, str):
        upper = value.strip().upper()
        mapping = {
            "BUY": AgentAction.BUY_M,
            "BUY_S": AgentAction.BUY_S,
            "BUY_M": AgentAction.BUY_M,
            "BUY_L": AgentAction.BUY_L,
            "SELL": AgentAction.SELL,
            "HOLD": AgentAction.HOLD,
        }
        if upper in mapping:
            return mapping[upper]
        if "SELL" in upper:
            return AgentAction.SELL
        if "BUY" in upper:
            return AgentAction.BUY_M
    return AgentAction.HOLD


def _clamp_float(value: Any, default: float = 0.5) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, num))
