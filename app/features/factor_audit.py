"""Utilities for auditing persisted factor values against live formulas."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Mapping, Optional, Sequence

from app.features.factors import (
    DEFAULT_FACTORS,
    FactorResult,
    FactorSpec,
    compute_factors,
    lookup_factor_spec,
)
from app.utils.db import db_session
from app.utils.logging import get_logger

LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "factor_audit"}


@dataclass
class FactorAuditIssue:
    """Details for a single factor mismatch discovered during auditing."""

    ts_code: str
    factor: str
    stored: Optional[float]
    recomputed: Optional[float]
    difference: Optional[float]


@dataclass
class FactorAuditSummary:
    """Aggregated results for a factor audit run."""

    trade_date: date
    tolerance: float
    factor_names: List[str]
    total_persisted: int
    total_recomputed: int
    evaluated: int
    mismatched: int
    missing_persisted: int
    missing_recomputed: int
    missing_columns: List[str] = field(default_factory=list)
    issues: List[FactorAuditIssue] = field(default_factory=list)


def audit_factors(
    trade_date: date,
    *,
    factors: Optional[Sequence[str | FactorSpec]] = None,
    tolerance: float = 1e-6,
    max_issues: int = 50,
) -> FactorAuditSummary:
    """Recompute factor values and compare them with persisted records.

    Args:
        trade_date: 需要审计的交易日
        factors: 因子名称或 ``FactorSpec`` 序列，缺省为默认因子集合
        tolerance: 比较阈值，超出视为不一致
        max_issues: 限制返回的详细问题数量

    Returns:
        FactorAuditSummary: 审计结果摘要
    """

    specs = _resolve_factor_specs(factors)
    factor_names = [spec.name for spec in specs]
    trade_date_str = trade_date.strftime("%Y%m%d")

    persisted_map, missing_columns = _load_persisted_factors(trade_date_str, factor_names)
    recomputed_results = compute_factors(
        trade_date,
        specs,
        persist=False,
    )
    recomputed_map = {result.ts_code: result.values for result in recomputed_results}

    mismatched = 0
    evaluated = 0
    missing_persisted = 0
    issues: List[FactorAuditIssue] = []

    for ts_code, values in recomputed_map.items():
        stored = persisted_map.get(ts_code)
        if not stored:
            missing_persisted += 1
            LOGGER.debug(
                "审计未找到持久化记录 ts_code=%s trade_date=%s",
                ts_code,
                trade_date_str,
                extra=LOG_EXTRA,
            )
            continue
        evaluated += 1
        for factor in factor_names:
            recomputed_value = values.get(factor)
            stored_value = stored.get(factor)
            numeric_recomputed = _coerce_float(recomputed_value)
            numeric_stored = _coerce_float(stored_value)
            if numeric_recomputed is None and numeric_stored is None:
                continue
            if numeric_recomputed is None or numeric_stored is None:
                mismatched += 1
                if len(issues) < max_issues:
                    issues.append(
                        FactorAuditIssue(
                            ts_code=ts_code,
                            factor=factor,
                            stored=stored_value,
                            recomputed=recomputed_value,
                            difference=None,
                        )
                    )
                continue
            diff = abs(numeric_recomputed - numeric_stored)
            if math.isnan(diff) or diff > tolerance:
                mismatched += 1
                if len(issues) < max_issues:
                    issues.append(
                        FactorAuditIssue(
                            ts_code=ts_code,
                            factor=factor,
                            stored=numeric_stored,
                            recomputed=numeric_recomputed,
                            difference=diff if not math.isnan(diff) else None,
                        )
                    )

    missing_recomputed = len(
        {code for code in persisted_map.keys() if code not in recomputed_map}
    )

    summary = FactorAuditSummary(
        trade_date=trade_date,
        tolerance=tolerance,
        factor_names=factor_names,
        total_persisted=len(persisted_map),
        total_recomputed=len(recomputed_map),
        evaluated=evaluated,
        mismatched=mismatched,
        missing_persisted=missing_persisted,
        missing_recomputed=missing_recomputed,
        missing_columns=missing_columns,
        issues=issues,
    )

    LOGGER.info(
        "因子审计完成 trade_date=%s evaluated=%s mismatched=%s missing=%s/%s",
        trade_date_str,
        evaluated,
        mismatched,
        missing_persisted,
        missing_recomputed,
        extra=LOG_EXTRA,
    )
    if missing_columns:
        LOGGER.warning(
            "因子审计缺少字段 columns=%s trade_date=%s",
            missing_columns,
            trade_date_str,
            extra=LOG_EXTRA,
        )
    return summary


def _resolve_factor_specs(
    factors: Optional[Sequence[str | FactorSpec]],
) -> List[FactorSpec]:
    if not factors:
        return list(DEFAULT_FACTORS)
    resolved: Dict[str, FactorSpec] = {}
    for item in factors:
        if isinstance(item, FactorSpec):
            resolved[item.name] = FactorSpec(name=item.name, window=item.window)
            continue
        spec = lookup_factor_spec(str(item))
        if spec is None:
            LOGGER.debug("忽略未知因子，无法审计 factor=%s", item, extra=LOG_EXTRA)
            continue
        resolved[spec.name] = spec
    return list(resolved.values()) or list(DEFAULT_FACTORS)


def _load_persisted_factors(
    trade_date: str,
    factor_names: Sequence[str],
) -> tuple[Dict[str, Dict[str, Optional[float]]], List[str]]:
    if not factor_names:
        return {}, []
    with db_session(read_only=True) as conn:
        table_info = conn.execute("PRAGMA table_info(factors)").fetchall()
        available_columns: set[str] = set()
        for row in table_info:
            if isinstance(row, Mapping):
                available_columns.add(str(row.get("name")))
            else:
                available_columns.add(str(row[1]))
        selected = [name for name in factor_names if name in available_columns]
        missing_columns = [name for name in factor_names if name not in available_columns]
        if not selected:
            return {}, missing_columns
        column_clause = ", ".join(["ts_code", *selected])
        query = f"SELECT {column_clause} FROM factors WHERE trade_date = ?"
        rows = conn.execute(query, (trade_date,)).fetchall()
    persisted: Dict[str, Dict[str, Optional[float]]] = {}
    for row in rows:
        ts_code = row["ts_code"]
        persisted[ts_code] = {name: row[name] for name in selected}
    return persisted, missing_columns


def _coerce_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or not math.isfinite(numeric):
        return None
    return numeric


__all__ = [
    "FactorAuditIssue",
    "FactorAuditSummary",
    "audit_factors",
]
