"""Department-level LLM agents coordinating multi-model decisions."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from app.agents.base import AgentAction
from app.llm.client import run_llm_with_config
from app.llm.prompts import department_prompt
from app.utils.config import AppConfig, DepartmentSettings, LLMConfig
from app.utils.logging import get_logger
from app.utils.data_access import DataBroker, parse_field_path

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
        supplement_chunks: List[str] = []
        transcript: List[str] = []
        delivered_requests = {
            (field, 1)
            for field in (mutable_context.raw.get("scope_values") or {}).keys()
        }

        response = ""
        decision_data: Dict[str, Any] = {}
        for round_idx in range(self._max_rounds):
            supplement_text = "\n\n".join(chunk for chunk in supplement_chunks if chunk)
            prompt = department_prompt(self.settings, mutable_context, supplements=supplement_text)
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

            transcript.append(response)
            decision_data = _parse_department_response(response)
            data_requests = _parse_data_requests(decision_data)
            filtered_requests = [
                req
                for req in data_requests
                if (req.field, req.window) not in delivered_requests
            ]

            if filtered_requests and round_idx < self._max_rounds - 1:
                lines, payload, delivered = self._fulfill_data_requests(
                    mutable_context, filtered_requests
                )
                if payload:
                    supplement_chunks.append(
                        f"回合 {round_idx + 1} 追加数据:\n" + "\n".join(lines)
                    )
                    mutable_context.raw.setdefault("supplement_data", []).extend(payload)
                    mutable_context.raw.setdefault("supplement_rounds", []).append(
                        {
                            "round": round_idx + 1,
                            "requests": [req.__dict__ for req in filtered_requests],
                            "data": payload,
                        }
                    )
                    delivered_requests.update(delivered)
                    decision_data.pop("data_requests", None)
                    continue
                LOGGER.debug(
                    "部门 %s 数据请求无结果：%s",
                    self.settings.code,
                    filtered_requests,
                    extra=LOG_EXTRA,
                )
            decision_data.pop("data_requests", None)
            break

        mutable_context.raw["supplement_transcript"] = list(transcript)

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
            raw_response="\n\n".join(transcript) if transcript else response,
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

        latest_fields: List[str] = []
        series_requests: List[Tuple[DataRequest, Tuple[str, str]]] = []

        for req in requests:
            field = req.field.strip()
            if not field:
                continue
            if req.window <= 1:
                if field not in latest_fields:
                    latest_fields.append(field)
                delivered.add((field, 1))
                continue
            parsed = parse_field_path(field)
            if not parsed:
                lines.append(f"- {field}: 字段不合法，已忽略")
                continue
            series_requests.append((req, parsed))
            delivered.add((field, req.window))

        if latest_fields:
            latest_values = self._broker.fetch_latest(ts_code, trade_date, latest_fields)
            for field in latest_fields:
                value = latest_values.get(field)
                if value is None:
                    lines.append(f"- {field}: (数据缺失)")
                else:
                    lines.append(f"- {field}: {value}")
                payload.append({"field": field, "window": 1, "values": value})

        for req, parsed in series_requests:
            table, column = parsed
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
            payload.append({"field": req.field, "window": req.window, "values": series})

        return lines, payload, delivered


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
