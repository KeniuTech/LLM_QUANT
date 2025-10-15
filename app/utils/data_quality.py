"""Utility helpers for performing lightweight data quality checks."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, Iterable, List, Optional

from app.utils.db import db_session
from app.utils.logging import get_logger

LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "data_quality"}

Severity = str  # Literal["ERROR", "WARN", "INFO"] (avoid importing Literal for py<3.8)


@dataclass
class DataQualityResult:
    check: str
    severity: Severity
    detail: str
    extras: Optional[Dict[str, object]] = None


def _parse_date(value: object) -> Optional[date]:
    """Best-effort parse for trade_date columns stored as str/int."""
    if value is None:
        return None
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        LOGGER.debug("无法解析日期字段 value=%s", value, extra=LOG_EXTRA)
        return None


def run_data_quality_checks(*, window_days: int = 7) -> List[DataQualityResult]:
    """Execute a suite of lightweight data quality checks."""
    results: List[DataQualityResult] = []
    today = date.today()
    window_start = today - timedelta(days=window_days)

    try:
        with db_session(read_only=True) as conn:
            # 1. 候选池最新数据
            try:
                row = conn.execute(
                    """
                    SELECT trade_date, COUNT(DISTINCT ts_code) AS cnt
                    FROM investment_pool
                    ORDER BY trade_date DESC
                    LIMIT 1
                    """
                ).fetchone()
            except Exception:  # noqa: BLE001
                LOGGER.exception("查询 investment_pool 失败", extra=LOG_EXTRA)
                results.append(
                    DataQualityResult(
                        "候选池可用性",
                        "ERROR",
                        "读取候选池数据失败，请检查 investment_pool 表结构与权限。",
                    )
                )
                row = None

            latest_candidate_date = _parse_date(row["trade_date"]) if row else None
            candidate_count = int(row["cnt"]) if row and row["cnt"] is not None else 0
            if row is None:
                pass
            elif latest_candidate_date is None:
                results.append(
                    DataQualityResult(
                        "候选池可用性",
                        "ERROR",
                        "未解析到候选池日期，请确认 trade_date 字段格式。",
                        {"raw_value": row["trade_date"]},
                    )
                )
            else:
                age = (today - latest_candidate_date).days
                extras = {
                    "trade_date": latest_candidate_date.isoformat(),
                    "candidate_count": candidate_count,
                    "age_days": age,
                }
                if candidate_count == 0:
                    results.append(
                        DataQualityResult(
                            "候选池可用性",
                            "ERROR",
                            f"{latest_candidate_date} 的候选池为空。",
                            extras,
                        )
                    )
                elif age > window_days:
                    results.append(
                        DataQualityResult(
                            "候选池可用性",
                            "WARN",
                            f"候选池停留在 {latest_candidate_date}，已超过 {window_days} 天未更新。",
                            extras,
                        )
                    )
                else:
                    results.append(
                        DataQualityResult(
                            "候选池可用性",
                            "INFO",
                            f"最新候选池（{latest_candidate_date}）包含 {candidate_count} 个标的。",
                            extras,
                        )
                    )

            # 2. agent_utils 覆盖率
            try:
                row_agent = conn.execute(
                    """
                    SELECT trade_date, COUNT(DISTINCT ts_code) AS cnt
                    FROM agent_utils
                    ORDER BY trade_date DESC
                    LIMIT 1
                    """
                ).fetchone()
            except Exception:  # noqa: BLE001
                LOGGER.exception("查询 agent_utils 失败", extra=LOG_EXTRA)
                results.append(
                    DataQualityResult(
                        "策略评估数据",
                        "ERROR",
                        "读取 agent_utils 失败，无法评估部门/代理数据是否可用。",
                    )
                )
                row_agent = None

            latest_agent_date = _parse_date(row_agent["trade_date"]) if row_agent else None
            agent_count = int(row_agent["cnt"]) if row_agent and row_agent["cnt"] is not None else 0
            if row_agent is None:
                pass
            elif latest_agent_date is None:
                results.append(
                    DataQualityResult(
                        "策略评估数据",
                        "ERROR",
                        "未解析到 agent_utils 日期，请确认 trade_date 字段格式。",
                        {"raw_value": row_agent["trade_date"]},
                    )
                )
            else:
                extras = {
                    "trade_date": latest_agent_date.isoformat(),
                    "decision_count": agent_count,
                }
                if agent_count == 0:
                    results.append(
                        DataQualityResult(
                            "策略评估数据",
                            "WARN",
                            f"{latest_agent_date} 的 agent_utils 中未找到标的记录。",
                            extras,
                        )
                    )
                else:
                    results.append(
                        DataQualityResult(
                            "策略评估数据",
                            "INFO",
                            f"{latest_agent_date} 共有 {agent_count} 个标的完成策略评估。",
                            extras,
                        )
                    )

                if latest_candidate_date and latest_candidate_date != latest_agent_date:
                    results.append(
                        DataQualityResult(
                            "候选与策略同步",
                            "WARN",
                            "候选池与策略评估日期不一致，建议重新触发评估或数据同步。",
                            {
                                "candidate_date": latest_candidate_date.isoformat(),
                                "agent_date": latest_agent_date.isoformat(),
                            },
                        )
                    )

            # 3. 开仓记录 vs 快照
            try:
                open_positions = conn.execute(
                    """
                    SELECT COUNT(*) AS cnt
                    FROM portfolio_positions
                    WHERE status = 'open'
                    """
                ).fetchone()
                open_position_count = int(open_positions["cnt"]) if open_positions else 0
            except Exception:  # noqa: BLE001
                LOGGER.exception("查询 portfolio_positions 失败", extra=LOG_EXTRA)
                open_position_count = 0
                results.append(
                    DataQualityResult(
                        "持仓数据",
                        "WARN",
                        "无法读取当前持仓，检查 portfolio_positions 表是否存在。",
                    )
                )

            latest_snapshot_date = None
            snapshot_date_column = None
            try:
                snapshot_info = conn.execute("PRAGMA table_info(portfolio_snapshots)").fetchall()
            except Exception:  # noqa: BLE001
                LOGGER.exception("读取 portfolio_snapshots 结构失败", extra=LOG_EXTRA)
                snapshot_info = []

            preferred_snapshot_columns: Iterable[str] = (
                "as_of",
                "snapshot_date",
                "trade_date",
                "date",
                "created_at",
                "timestamp",
            )
            available_snapshot_columns: List[str] = []
            for row in snapshot_info:
                name = row["name"] if "name" in row.keys() else row[1]
                available_snapshot_columns.append(name)
                if name in preferred_snapshot_columns and snapshot_date_column is None:
                    snapshot_date_column = name

            if snapshot_date_column:
                try:
                    snap_row = conn.execute(
                        f"""
                        SELECT MAX({snapshot_date_column}) AS latest_snapshot
                        FROM portfolio_snapshots
                        """
                    ).fetchone()
                    if snap_row and snap_row["latest_snapshot"]:
                        latest_snapshot_date = _parse_date(snap_row["latest_snapshot"])
                except Exception:  # noqa: BLE001
                    LOGGER.exception("查询 portfolio_snapshots 失败", extra=LOG_EXTRA)
            elif available_snapshot_columns:
                results.append(
                    DataQualityResult(
                        "持仓快照",
                        "WARN",
                        "未找到标准快照日期字段（如 as_of/snapshot_date），请确认表结构。",
                        {"columns": available_snapshot_columns},
                    )
                )
            else:
                results.append(
                    DataQualityResult(
                        "持仓快照",
                        "WARN",
                        "未检测到 portfolio_snapshots 表，无法校验持仓快照。",
                    )
                )

            if open_position_count > 0:
                if latest_snapshot_date is None:
                    results.append(
                        DataQualityResult(
                            "持仓快照",
                            "WARN",
                            "存在未平仓头寸，但未找到任何持仓快照记录。",
                            {"open_positions": open_position_count},
                        )
                    )
                elif latest_snapshot_date < window_start:
                    results.append(
                        DataQualityResult(
                            "持仓快照",
                            "WARN",
                            f"最新持仓快照停留在 {latest_snapshot_date}，已超过窗口 {window_days} 天。",
                            {
                                "latest_snapshot": latest_snapshot_date.isoformat(),
                                "open_positions": open_position_count,
                            },
                        )
                    )
                else:
                    results.append(
                        DataQualityResult(
                            "持仓快照",
                            "INFO",
                            f"最新持仓快照日期：{latest_snapshot_date}。",
                            {
                                "latest_snapshot": latest_snapshot_date.isoformat(),
                                "open_positions": open_position_count,
                            },
                        )
                    )

            # 4. 新闻数据时效
            latest_news_date = None
            try:
                news_row = conn.execute(
                    """
                    SELECT MAX(pub_time) AS latest_pub
                    FROM news
                    """
                ).fetchone()
                if news_row and news_row["latest_pub"]:
                    try:
                        latest_news_date = datetime.fromisoformat(
                            str(news_row["latest_pub"])
                        )
                    except ValueError:
                        LOGGER.debug(
                            "无法解析新闻时间字段 value=%s",
                            news_row["latest_pub"],
                            extra=LOG_EXTRA,
                        )
            except Exception:  # noqa: BLE001
                LOGGER.exception("查询 news 失败", extra=LOG_EXTRA)

            if latest_news_date:
                if latest_news_date.tzinfo is not None:
                    now_ts = datetime.now(tz=latest_news_date.tzinfo)
                else:
                    now_ts = datetime.now()
                delta_days = (now_ts - latest_news_date).days
                if delta_days > window_days:
                    results.append(
                        DataQualityResult(
                            "新闻数据时效",
                            "WARN",
                            f"最新新闻时间为 {latest_news_date}, 已超过 {window_days} 天未更新。",
                            {"latest_pub_time": str(latest_news_date)},
                        )
                    )
                else:
                    results.append(
                        DataQualityResult(
                            "新闻数据时效",
                            "INFO",
                            f"新闻数据最新时间：{latest_news_date}",
                            {"latest_pub_time": str(latest_news_date)},
                        )
                    )
            else:
                results.append(
                    DataQualityResult(
                        "新闻数据时效",
                        "WARN",
                        "未找到最新新闻记录，请检查 RSS 或新闻数据接入。",
                    )
                )
    except Exception:  # noqa: BLE001
        LOGGER.exception("执行数据质量检查失败", extra=LOG_EXTRA)
        results.append(
            DataQualityResult(
                "运行状态",
                "ERROR",
                "数据质量检查过程中发生异常，请查看日志。",
            )
        )

    return results
