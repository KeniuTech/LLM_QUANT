"""Department-level LLM agents coordinating multi-model decisions."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

from app.agents.base import AgentAction
from app.llm.client import (
    call_endpoint_with_messages,
    resolve_endpoint,
    run_llm_with_config,
    LLMError,
)
from app.llm.prompts import department_prompt
from app.utils.config import AppConfig, DepartmentSettings, LLMConfig
from app.utils.logging import get_logger, get_conversation_logger
from app.utils.data_access import DataBroker

LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "department"}
CONV_LOGGER = get_conversation_logger()


@dataclass
class TableRequest:
    name: str
    window: int = 1
    trade_date: Optional[str] = None


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
    telemetry: Dict[str, Any] = field(default_factory=dict)

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
            "telemetry": self.telemetry,
        }


class DepartmentAgent:
    """Wraps LLM ensemble logic for a single analytical department."""

    ALLOWED_TABLES: ClassVar[List[str]] = [
        "daily",
        "daily_basic",
        "stk_limit",
        "suspend",
        "heat_daily",
        "news",
        "index_daily",
    ]

    def __init__(
        self,
        settings: DepartmentSettings,
        resolver: Optional[Callable[[DepartmentSettings], LLMConfig]] = None,
    ) -> None:
        self.settings = settings
        self._resolver = resolver
        self._broker = DataBroker()
        self._max_rounds = 3
        self._tool_choice = "auto"

    @property
    def max_rounds(self) -> int:
        return self._max_rounds

    @max_rounds.setter
    def max_rounds(self, value: Any) -> None:
        try:
            numeric = int(round(float(value)))
        except (TypeError, ValueError):
            raise ValueError("max_rounds must be numeric") from None
        if numeric < 1:
            numeric = 1
        if numeric > 6:
            numeric = 6
        self._max_rounds = numeric

    @property
    def tool_choice(self) -> str:
        return self._tool_choice

    @tool_choice.setter
    def tool_choice(self, value: Any) -> None:
        if value is None:
            self._tool_choice = "auto"
            return
        normalized = str(value).strip().lower()
        allowed = {"auto", "none", "required"}
        if normalized not in allowed:
            raise ValueError(f"Unsupported tool choice: {value}")
        self._tool_choice = normalized

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
        prompt_body = department_prompt(self.settings, mutable_context)
        template_meta = {}
        raw_templates = mutable_context.raw.get("template_meta") if isinstance(mutable_context.raw, dict) else None
        if isinstance(raw_templates, dict):
            template_meta = dict(raw_templates.get(self.settings.code, {}))
        prompt_checksum = hashlib.sha1(prompt_body.encode("utf-8")).hexdigest()
        prompt_preview = prompt_body[:240]
        messages.append({"role": "user", "content": prompt_body})

        transcript: List[str] = []
        delivered_requests: set[Tuple[str, int, str]] = set()

        primary_endpoint = llm_cfg.primary
        try:
            resolved_primary = resolve_endpoint(primary_endpoint)
        except LLMError as exc:
            LOGGER.warning(
                "部门 %s 无法解析 LLM 端点，回退传统提示：%s",
                self.settings.code,
                exc,
                extra=LOG_EXTRA,
            )
            return self._analyze_legacy(mutable_context, system_prompt)

        final_message: Optional[Dict[str, Any]] = None
        usage_records: List[Dict[str, Any]] = []
        tool_call_records: List[Dict[str, Any]] = []
        rounds_executed = 0
        CONV_LOGGER.info(
            "dept=%s ts_code=%s trade_date=%s start",
            self.settings.code,
            context.ts_code,
            context.trade_date,
        )

        for round_idx in range(self._max_rounds):
            try:
                response = call_endpoint_with_messages(
                    primary_endpoint,
                    messages,
                    tools=tools,
                    tool_choice=self._tool_choice,
                )
            except LLMError as exc:
                LOGGER.warning(
                    "部门 %s 函数调用失败，回退传统提示：%s",
                    self.settings.code,
                    exc,
                    extra=LOG_EXTRA,
                )
                return self._analyze_legacy(mutable_context, system_prompt)

            rounds_executed = round_idx + 1

            usage = response.get("usage") if isinstance(response, Mapping) else None
            if isinstance(usage, Mapping):
                usage_payload = {"round": round_idx + 1}
                usage_payload.update(dict(usage))
                usage_records.append(usage_payload)

            choice = (response.get("choices") or [{}])[0]
            message = choice.get("message", {})
            transcript.append(_message_to_text(message))

            assistant_record: Dict[str, Any] = {
                "role": "assistant",
                "content": _extract_message_content(message),
            }
            if message.get("tool_calls"):
                assistant_record["tool_calls"] = message.get("tool_calls")
            messages.append(assistant_record)
            CONV_LOGGER.info(
                "dept=%s round=%s assistant=%s",
                self.settings.code,
                round_idx + 1,
                assistant_record,
            )

            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                for call in tool_calls:
                    function_block = call.get("function") or {}
                    tool_response, delivered = self._handle_tool_call(
                        mutable_context,
                        call,
                        delivered_requests,
                        round_idx,
                    )
                    tables_summary: List[Dict[str, Any]] = []
                    for item in tool_response.get("results") or []:
                        if isinstance(item, Mapping):
                            tables_summary.append(
                                {
                                    "table": item.get("table"),
                                    "window": item.get("window"),
                                    "trade_date": item.get("trade_date"),
                                    "row_count": len(item.get("rows") or []),
                                }
                            )
                    tool_call_records.append(
                        {
                            "round": round_idx + 1,
                            "id": call.get("id"),
                            "name": function_block.get("name"),
                            "arguments": function_block.get("arguments"),
                            "status": tool_response.get("status"),
                            "results": len(tool_response.get("results") or []),
                            "tables": tables_summary,
                            "skipped": list(tool_response.get("skipped") or []),
                        }
                    )
                    transcript.append(
                        json.dumps({"tool_response": tool_response}, ensure_ascii=False)
                    )
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id"),
                            "content": json.dumps(tool_response, ensure_ascii=False),
                        }
                    )
                    delivered_requests.update(delivered)
                    CONV_LOGGER.info(
                        "dept=%s round=%s tool_call=%s response=%s",
                        self.settings.code,
                        round_idx + 1,
                        call,
                        tool_response,
                    )
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
            CONV_LOGGER.warning(
                "dept=%s rounds_exhausted last_message=%s",
                self.settings.code,
                final_message,
            )

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

        def _safe_int(value: Any) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):  # noqa: PERF203 - clarity
                return 0

        prompt_tokens_total = 0
        completion_tokens_total = 0
        total_tokens_reported = 0
        for usage_payload in usage_records:
            prompt_tokens_total += _safe_int(
                usage_payload.get("prompt_tokens")
                or usage_payload.get("prompt_tokens_total")
            )
            completion_tokens_total += _safe_int(
                usage_payload.get("completion_tokens")
                or usage_payload.get("completion_tokens_total")
            )
            reported_total = _safe_int(
                usage_payload.get("total_tokens")
                or usage_payload.get("total_tokens_total")
            )
            if reported_total:
                total_tokens_reported += reported_total

        total_tokens = (
            total_tokens_reported
            if total_tokens_reported
            else prompt_tokens_total + completion_tokens_total
        )

        telemetry: Dict[str, Any] = {
            "provider": resolved_primary.get("provider_key"),
            "model": resolved_primary.get("model"),
            "temperature": resolved_primary.get("temperature"),
            "timeout": resolved_primary.get("timeout"),
            "endpoint_prompt_template": resolved_primary.get("prompt_template"),
            "rounds": rounds_executed,
            "tool_call_count": len(tool_call_records),
            "tool_trace": tool_call_records,
            "usage_by_round": usage_records,
            "tokens": {
                "prompt": prompt_tokens_total,
                "completion": completion_tokens_total,
                "total": total_tokens,
            },
            "prompt": {
                "checksum": prompt_checksum,
                "length": len(prompt_body),
                "preview": prompt_preview,
                "role_description": self.settings.description,
                "instruction": self.settings.prompt,
                "system": system_prompt,
            },
            "template": template_meta,
            "messages_exchanged": len(messages),
            "supplement_rounds": len(tool_call_records),
        }

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
            telemetry=telemetry,
        )
        LOGGER.debug(
            "部门 %s 决策：action=%s confidence=%.2f",
            self.settings.code,
            decision.action.value,
            decision.confidence,
            extra=LOG_EXTRA,
        )
        CONV_LOGGER.info(
            "dept=%s decision action=%s confidence=%.2f summary=%s",
            self.settings.code,
            decision.action.value,
            decision.confidence,
            summary or "",
        )
        CONV_LOGGER.info(
            "dept=%s telemetry=%s",
            self.settings.code,
            json.dumps(telemetry, ensure_ascii=False),
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
        requests: Sequence[TableRequest],
    ) -> Tuple[List[str], List[Dict[str, Any]], set[Tuple[str, int, str]]]:
        lines: List[str] = []
        payload: List[Dict[str, Any]] = []
        delivered: set[Tuple[str, int, str]] = set()

        ts_code = context.ts_code
        default_trade_date = self._normalize_trade_date(context.trade_date)

        for req in requests:
            table = (req.name or "").strip().lower()
            if not table:
                continue
            if table not in self.ALLOWED_TABLES:
                lines.append(f"- {table}: 不在允许的表列表中")
                continue
            trade_date = self._normalize_trade_date(req.trade_date or default_trade_date)
            window = max(1, min(req.window or 1, getattr(self._broker, "MAX_WINDOW", 120)))
            key = (table, window, trade_date)
            if key in delivered:
                lines.append(f"- {table}: 已返回窗口 {window} 的数据，跳过重复请求")
                continue

            rows = self._broker.fetch_table_rows(
                table,
                ts_code,
                trade_date,
                window,
                auto_refresh=False  # 避免在回测过程中触发自动补数
            )
            if rows:
                preview = ", ".join(
                    f"{row.get('trade_date', 'NA')}" for row in rows[: min(len(rows), 5)]
                )
                lines.append(
                    f"- {table} (window={window} trade_date<= {trade_date}): 返回 {len(rows)} 行 {preview}"
                )
            else:
                lines.append(
                    f"- {table} (window={window} trade_date<= {trade_date}): (数据缺失)"
                )
            payload.append(
                {
                    "table": table,
                    "window": window,
                    "trade_date": trade_date,
                    "rows": rows,
                }
            )
            delivered.add(key)

        return lines, payload, delivered


    def _handle_tool_call(
        self,
        context: DepartmentContext,
        call: Mapping[str, Any],
        delivered_requests: set[Tuple[str, int, str]],
        round_idx: int,
    ) -> Tuple[Dict[str, Any], set[Tuple[str, int, str]]]:
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
        base_trade_date = self._normalize_trade_date(
            args.get("trade_date") or context.trade_date
        )
        raw_requests = args.get("tables") or []
        requests: List[TableRequest] = []
        skipped: List[str] = []
        for item in raw_requests:
            name = str(item.get("name", "")).strip().lower()
            if not name:
                continue
            window_raw = item.get("window")
            try:
                window = int(window_raw) if window_raw is not None else 1
            except (TypeError, ValueError):
                window = 1
            window = max(1, min(window, getattr(self._broker, "MAX_WINDOW", 120)))
            override_date = item.get("trade_date")
            req_date = self._normalize_trade_date(override_date or base_trade_date)
            key = (name, window, req_date)
            if key in delivered_requests:
                skipped.append(name)
                continue
            requests.append(TableRequest(name=name, window=window, trade_date=req_date))

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
                        "根据表名请求指定交易日及窗口的历史数据。当前仅支持 'daily' 与 'daily_basic' 表。"
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tables": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "enum": self.ALLOWED_TABLES,
                                            "description": "表名，例如 daily 或 daily_basic",
                                        },
                                        "window": {
                                            "type": "integer",
                                            "minimum": 1,
                                            "maximum": max_window,
                                            "description": "向前回溯的记录条数，默认为 1",
                                        },
                                        "trade_date": {
                                            "type": "string",
                                            "pattern": r"^\\d{8}$",
                                            "description": "覆盖默认交易日（格式 YYYYMMDD）",
                                        },
                                    },
                                    "required": ["name"],
                                },
                                "minItems": 1,
                            },
                            "trade_date": {
                                "type": "string",
                                "pattern": r"^\\d{8}$",
                                "description": "默认交易日（格式 YYYYMMDD）",
                            },
                        },
                        "required": ["tables"],
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
            CONV_LOGGER.error(
                "dept=%s legacy_call_failed err=%s",
                self.settings.code,
                exc,
            )
            return DepartmentDecision(
                department=self.settings.code,
                action=AgentAction.HOLD,
                confidence=0.0,
                summary=f"LLM 调用失败：{exc}",
                raw_response=str(exc),
            )

        context.raw["supplement_transcript"] = [response]
        CONV_LOGGER.info("dept=%s legacy_response=%s", self.settings.code, response)
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
