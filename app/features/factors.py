"""Feature engineering for signals and indicator computation."""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, date, timezone, timedelta
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from app.core.indicators import momentum, rolling_mean, volatility
from app.data.schema import initialize_database
from app.utils.data_access import DataBroker
from app.utils.feature_snapshots import FeatureSnapshotService
from app.utils.db import db_session
from app.utils.logging import get_logger
# 导入扩展因子模块
from app.features.extended_factors import ExtendedFactors
from app.features.sentiment_factors import SentimentFactors
from app.features.value_risk_factors import ValueRiskFactors
# 导入因子验证功能
from app.features.validation import check_data_sufficiency, check_data_sufficiency_for_zero_window, detect_outliers
# 导入UI进度状态管理
try:
    from app.features.progress import get_progress_handler
except ImportError:  # pragma: no cover - optional dependency
    def get_progress_handler():
        return None


LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "factor_compute"}
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_LATEST_BASE_FIELDS: List[str] = [
    "daily_basic.pe",
    "daily_basic.pb",
    "daily_basic.ps",
    "daily_basic.turnover_rate",
    "daily_basic.volume_ratio",
    "daily.close",
    "daily.amount",
    "daily.vol",
    "daily_basic.dv_ratio",
]


@dataclass
class FactorSpec:
    name: str
    window: int


@dataclass
class FactorResult:
    ts_code: str
    trade_date: date
    values: Dict[str, float | None]


# 基础因子和扩展因子的完整列表
DEFAULT_FACTORS: List[FactorSpec] = [
    # 基础动量因子
    FactorSpec("mom_5", 5),
    FactorSpec("mom_20", 20),
    FactorSpec("mom_60", 60),
    # 波动率因子
    FactorSpec("volat_20", 20),
    # 换手率因子
    FactorSpec("turn_20", 20),
    FactorSpec("turn_5", 5),
    # 估值因子
    FactorSpec("val_pe_score", 0),
    FactorSpec("val_pb_score", 0),
    # 量比因子
    FactorSpec("volume_ratio_score", 0),
    # 扩展因子
    # 增强动量因子
    FactorSpec("mom_10_30", 0),  # 10日与30日动量差
    FactorSpec("mom_5_20_rank", 0),  # 相对排名动量因子
    FactorSpec("mom_dynamic", 0),  # 动态窗口动量因子
    # 波动率相关因子
    FactorSpec("volat_5", 5),  # 短期波动率
    FactorSpec("volat_ratio", 0),  # 长短期波动率比率
    # 换手率扩展因子
    FactorSpec("turn_60", 60),  # 长期换手率
    FactorSpec("turn_rank", 0),  # 换手率相对排名
    # 价格均线比率因子
    FactorSpec("price_ma_10_ratio", 0),  # 当前价格与10日均线比率
    FactorSpec("price_ma_20_ratio", 0),  # 当前价格与20日均线比率
    FactorSpec("price_ma_60_ratio", 0),  # 当前价格与60日均线比率
    # 成交量均线比率因子
    FactorSpec("volume_ma_5_ratio", 0),  # 当前成交量与5日均线比率
    FactorSpec("volume_ma_20_ratio", 0),  # 当前成交量与20日均线比率
    # 高级估值因子
    FactorSpec("val_ps_score", 0),  # PS估值评分
    FactorSpec("val_multiscore", 0),  # 综合估值评分
    FactorSpec("val_dividend_score", 0),  # 股息率估值评分
    # 市场状态因子
    FactorSpec("market_regime", 0),  # 市场状态因子
    FactorSpec("trend_strength", 0),  # 趋势强度因子
    # 情绪因子
    FactorSpec("sent_momentum", 20),  # 新闻情感动量
    FactorSpec("sent_impact", 0),    # 新闻影响力
    FactorSpec("sent_market", 20),   # 市场情绪指数
    FactorSpec("sent_divergence", 0),  # 行业情绪背离度
    # 风险和估值因子
    FactorSpec("risk_penalty", 0),  # 风险惩罚因子
]

_FACTOR_SPEC_MAP: Dict[str, FactorSpec] = {spec.name: spec for spec in DEFAULT_FACTORS}


def lookup_factor_spec(name: str) -> Optional[FactorSpec]:
    """Return a copy of the registered ``FactorSpec`` for ``name`` if available."""

    base = _FACTOR_SPEC_MAP.get(name)
    if base is None:
        return None
    return FactorSpec(name=base.name, window=base.window)


def compute_factors(
    trade_date: date,
    factors: Iterable[FactorSpec] = DEFAULT_FACTORS,
    *, 
    ts_codes: Optional[Sequence[str]] = None,
    skip_existing: bool = False,
    batch_size: int = 100,
) -> List[FactorResult]:
    """Calculate and persist factor values for the requested date.

    ``ts_codes`` can be supplied to restrict computation to a subset of the
    universe. When ``skip_existing`` is True, securities that already have an
    entry for ``trade_date`` will be ignored.
    
    Args:
        trade_date: 交易日日期
        factors: 要计算的因子列表
        ts_codes: 可选，限制计算的证券代码列表
        skip_existing: 是否跳过已存在的因子值
        batch_size: 批处理大小，用于优化性能
    
    Returns:
        因子计算结果列表
    """

    specs = [spec for spec in factors if spec.window >= 0]
    if not specs:
        return []

    initialize_database()
    trade_date_str = trade_date.strftime("%Y%m%d")

    _ensure_factor_columns(specs)

    allowed = {code.strip().upper() for code in ts_codes or () if code.strip()}
    universe = _load_universe(trade_date_str, allowed if allowed else None)
    if not universe:
        LOGGER.info("无可用标的生成因子 trade_date=%s", trade_date_str, extra=LOG_EXTRA)
        return []

    if skip_existing:
        # 检查所有因子名称
        factor_names = [spec.name for spec in specs]
        existing = _existing_factor_codes_with_factors(trade_date_str, factor_names)
        universe = [code for code in universe if code not in existing]
        if not universe:
            LOGGER.debug(
                "目标交易日所有因子已存在 trade_date=%s universe_size=%s",
                trade_date_str,
                len(existing),
                extra=LOG_EXTRA,
            )
            return []

    LOGGER.info(
        "开始计算因子 universe_size=%s factors=%s trade_date=%s",
        len(universe),
        [spec.name for spec in specs],
        trade_date_str,
        extra=LOG_EXTRA,
    )
    
    # 数据有效性校验初始化
    validation_stats = {
        "total": len(universe),
        "skipped": 0,
        "success": 0,
        "data_missing": 0,
        "outliers": 0
    }

    broker = DataBroker()
    results: List[FactorResult] = []
    rows_to_persist: List[tuple[str, Dict[str, float | None]]] = []
    total_batches = (len(universe) + batch_size - 1) // batch_size if universe else 0
    progress = get_progress_handler()
    if progress and universe:
        try:
            progress.start_calculation(len(universe), total_batches)
        except Exception:  # noqa: BLE001
            LOGGER.debug("Progress handler start_calculation 失败", extra=LOG_EXTRA)
            progress = None
    
    try:
        # 分批处理以优化性能
        for i in range(0, len(universe), batch_size):
            batch = universe[i:i+batch_size]
            batch_results = _compute_batch_factors(
                broker, 
                batch, 
                trade_date_str, 
                specs, 
                validation_stats,
                batch_index=i // batch_size,
                total_batches=total_batches or 1,
                processed_securities=i,
                total_securities=len(universe),
                progress=progress,
            )
            
            for ts_code, values in batch_results:
                if values:
                    results.append(FactorResult(ts_code=ts_code, trade_date=trade_date, values=values))
                    rows_to_persist.append((ts_code, values))
            
            # 显示进度
            processed = min(i + batch_size, len(universe))
            if processed % (batch_size * 5) == 0 or processed == len(universe):
                LOGGER.info(
                    "因子计算进度: %s/%s (%.1f%%) 成功:%s 跳过:%s 数据缺失:%s 异常值:%s",
                    processed, len(universe), 
                    (processed / len(universe)) * 100,
                    validation_stats["success"],
                    validation_stats["skipped"],
                    validation_stats["data_missing"],
                    validation_stats["outliers"],
                    extra=LOG_EXTRA,
                )

        if rows_to_persist:
            _persist_factor_rows(trade_date_str, rows_to_persist, specs)
        
        # 更新UI进度状态为完成
        if progress:
            try:
                progress.complete_calculation(
                    message=f"因子计算完成: 总数量={len(universe)}, 成功={validation_stats['success']}, 失败={len(universe) - validation_stats['success']}"
                )
            except Exception:  # noqa: BLE001
                LOGGER.debug("Progress handler complete_calculation 失败", extra=LOG_EXTRA)
            
        LOGGER.info(
            "因子计算完成 总数量:%s 成功:%s 失败:%s",
            len(universe), 
            validation_stats["success"],
            validation_stats["total"] - validation_stats["success"],
            extra=LOG_EXTRA,
        )
        
        return results
        
    except Exception as exc:
        # 发生错误时更新UI状态
        error_message = f"因子计算过程中发生错误: {exc}"
        if progress:
            try:
                progress.error_occurred(error_message)
            except Exception:  # noqa: BLE001
                LOGGER.debug("Progress handler error_occurred 失败", extra=LOG_EXTRA)
        LOGGER.error(error_message, extra=LOG_EXTRA)
        raise


def compute_factor_range(
    start: date,
    end: date,
    *,
    factors: Iterable[FactorSpec] = DEFAULT_FACTORS,
    ts_codes: Optional[Sequence[str]] = None,
    skip_existing: bool = True,
) -> List[FactorResult]:
    """Compute factors for all trading days within ``[start, end]`` inclusive."""

    if end < start:
        raise ValueError("end date must not precede start date")

    initialize_database()
    allowed = None
    if ts_codes:
        allowed = tuple(dict.fromkeys(code.strip().upper() for code in ts_codes if code.strip()))
        if not allowed:
            allowed = None

    start_str = start.strftime("%Y%m%d")
    end_str = end.strftime("%Y%m%d")
    trade_dates = _list_trade_dates(start_str, end_str, allowed)

    aggregated: List[FactorResult] = []
    for trade_date_str in trade_dates:
        trade_day = datetime.strptime(trade_date_str, "%Y%m%d").date()
        aggregated.extend(
            compute_factors(
                trade_day,
                factors,
                ts_codes=allowed,
                skip_existing=skip_existing,
            )
        )
    return aggregated


def compute_factors_incremental(
    *,
    factors: Iterable[FactorSpec] = DEFAULT_FACTORS,
    ts_codes: Optional[Sequence[str]] = None,
    skip_existing: bool = True,
    max_trading_days: Optional[int] = 5,
) -> Dict[str, object]:
    """增量计算因子（从最新一条因子记录之后开始）。

    Args:
        factors: 需要计算的因子列表。
        ts_codes: 限定计算的证券池。
        skip_existing: 是否跳过已存在数据。
        max_trading_days: 限制本次计算的交易日数量（按交易日计数）。

    Returns:
        包含起止日期、参与交易日及计算结果的字典。
    """

    initialize_database()
    codes_tuple = None
    if ts_codes:
        normalized = [
            code.strip().upper()
            for code in ts_codes
            if isinstance(code, str) and code.strip()
        ]
        codes_tuple = tuple(dict.fromkeys(normalized)) or None

    last_date_str = _latest_factor_trade_date()
    trade_dates = _list_trade_dates_after(last_date_str, codes_tuple, max_trading_days)
    if not trade_dates:
        LOGGER.info("未发现新的交易日需要计算因子（latest=%s）", last_date_str, extra=LOG_EXTRA)
        return {
            "start": None,
            "end": None,
            "trade_dates": [],
            "results": [],
            "count": 0,
        }

    aggregated_results: List[FactorResult] = []
    for trade_date_str in trade_dates:
        trade_day = datetime.strptime(trade_date_str, "%Y%m%d").date()
        aggregated_results.extend(
            compute_factors(
                trade_day,
                factors,
                ts_codes=codes_tuple,
                skip_existing=skip_existing,
            )
        )

    trading_dates = [datetime.strptime(item, "%Y%m%d").date() for item in trade_dates]
    return {
        "start": trading_dates[0],
        "end": trading_dates[-1],
        "trade_dates": trading_dates,
        "results": aggregated_results,
        "count": len(aggregated_results),
    }


def _load_universe(trade_date: str, allowed: Optional[set[str]] = None) -> List[str]:
    query = "SELECT ts_code FROM daily WHERE trade_date = ? ORDER BY ts_code"
    with db_session(read_only=True) as conn:
        rows = conn.execute(query, (trade_date,)).fetchall()
    codes = [row["ts_code"] for row in rows if row["ts_code"]]
    if allowed:
        allowed_upper = {code.upper() for code in allowed}
        return [code for code in codes if code.upper() in allowed_upper]
    return codes


def _existing_factor_codes(trade_date: str) -> set[str]:
    with db_session(read_only=True) as conn:
        rows = conn.execute(
            "SELECT ts_code FROM factors WHERE trade_date = ?",
            (trade_date,),
        ).fetchall()
    return {row["ts_code"] for row in rows if row["ts_code"]}


def _existing_factor_codes_with_factors(trade_date: str, factor_names: List[str]) -> Dict[str, bool]:
    """检查特定日期和因子的数据是否存在
    
    Args:
        trade_date: 交易日期
        factor_names: 因子名称列表
        
    Returns:
        字典，键为股票代码，值为是否存在所有因子
    """
    if not factor_names:
        return {}
        
    valid_names = [
        name
        for name in factor_names
        if isinstance(name, str) and _IDENTIFIER_RE.match(name)
    ]
    if not valid_names:
        return {}

    with db_session(read_only=True) as conn:
        columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(factors)").fetchall()
        }
        selected = [name for name in valid_names if name in columns]
        if not selected:
            return {}

        predicates = " AND ".join(f"{col} IS NOT NULL" for col in selected)
        query = (
            "SELECT ts_code FROM factors "
            "WHERE trade_date = ? AND "
            f"{predicates} "
            "GROUP BY ts_code"
        )
        rows = conn.execute(query, (trade_date,)).fetchall()

    return {row["ts_code"]: True for row in rows if row and row["ts_code"]}


def _list_trade_dates(
    start_date: str,
    end_date: str,
    allowed: Optional[Sequence[str]],
) -> List[str]:
    params: List[str] = [start_date, end_date]
    if allowed:
        placeholders = ", ".join("?" for _ in allowed)
        query = (
            "SELECT DISTINCT trade_date FROM daily "
            "WHERE trade_date BETWEEN ? AND ? "
            f"AND ts_code IN ({placeholders}) "
            "ORDER BY trade_date"
        )
        params.extend(allowed)
    else:
        query = (
            "SELECT DISTINCT trade_date FROM daily "
            "WHERE trade_date BETWEEN ? AND ? "
            "ORDER BY trade_date"
        )
    with db_session(read_only=True) as conn:
        rows = conn.execute(query, params).fetchall()
    return [row["trade_date"] for row in rows if row["trade_date"]]


def _list_trade_dates_after(
    last_trade_date: Optional[str],
    allowed: Optional[Sequence[str]],
    limit: Optional[int],
) -> List[str]:
    params: List[object] = []
    where_clauses: List[str] = []
    if last_trade_date:
        where_clauses.append("trade_date > ?")
        params.append(last_trade_date)
    base_query = "SELECT DISTINCT trade_date FROM daily"
    if allowed:
        placeholders = ", ".join("?" for _ in allowed)
        where_clauses.append(f"ts_code IN ({placeholders})")
        params.extend(allowed)
    if where_clauses:
        base_query += " WHERE " + " AND ".join(where_clauses)
    base_query += " ORDER BY trade_date"
    if limit is not None and limit > 0:
        base_query += f" LIMIT {int(limit)}"
    with db_session(read_only=True) as conn:
        rows = conn.execute(base_query, params).fetchall()
    return [row["trade_date"] for row in rows if row["trade_date"]]


def _latest_factor_trade_date() -> Optional[str]:
    with db_session(read_only=True) as conn:
        try:
            row = conn.execute("SELECT MAX(trade_date) AS max_trade_date FROM factors").fetchone()
        except sqlite3.OperationalError:
            return None
    value = row["max_trade_date"] if row else None
    if not value:
        return None
    return str(value)


def _compute_batch_factors(
    broker: DataBroker,
    ts_codes: List[str],
    trade_date: str,
    specs: Sequence[FactorSpec],
    validation_stats: Dict[str, int],
    batch_index: int = 0,
    total_batches: int = 1,
    processed_securities: int = 0,
    total_securities: int = 0,
    progress: Optional[object] = None,
) -> List[tuple[str, Dict[str, float | None]]]:
    """批量计算多个证券的因子值，提高计算效率"""
    batch_results = []
    
    # 批次化数据可用性检查
    available_codes = _check_batch_data_availability(broker, ts_codes, trade_date, specs)

    snapshot_service = FeatureSnapshotService(broker)
    latest_snapshot = snapshot_service.load_latest(
        trade_date,
        _LATEST_BASE_FIELDS,
        list(available_codes),
        auto_refresh=False,
    )
    
    # 更新UI进度状态 - 开始处理批次
    if progress and total_securities > 0:
        try:
            progress.update_progress(
                current_securities=processed_securities,
                current_batch=batch_index + 1,
                message=f"开始处理批次 {batch_index + 1}/{total_batches}",
            )
        except Exception:  # noqa: BLE001
            LOGGER.debug("Progress handler update_progress 失败", extra=LOG_EXTRA)
            progress = None
    
    for i, ts_code in enumerate(ts_codes):
        try:
            # 检查数据可用性（使用批次化结果）
            if ts_code not in available_codes:
                validation_stats["data_missing"] += 1
                continue
                
            # 计算因子值
            values = _compute_security_factors(
                broker,
                ts_code,
                trade_date,
                specs,
                latest_fields=latest_snapshot.get(ts_code),
            )
            
            if values:
                # 检测并处理异常值
                cleaned_values = detect_outliers(values, ts_code, trade_date)
                if cleaned_values:
                    batch_results.append((ts_code, cleaned_values))
                    validation_stats["success"] += 1
                    
                    # 记录验证统计信息
                    original_count = len(values)
                    cleaned_count = len(cleaned_values)
                    if cleaned_count < original_count:
                        validation_stats["outliers"] += (original_count - cleaned_count)
                        LOGGER.debug(
                            "因子值验证结果 ts_code=%s date=%s original=%d cleaned=%d",
                            ts_code, trade_date, original_count, cleaned_count,
                            extra=LOG_EXTRA
                        )
                else:
                    validation_stats["outliers"] += len(values)
                    LOGGER.warning(
                        "所有因子值均被标记为异常值 ts_code=%s date=%s",
                        ts_code, trade_date,
                        extra=LOG_EXTRA
                    )
            else:
                validation_stats["skipped"] += 1
                
            # 每处理1个证券更新一次进度，确保实时性
            if progress and total_securities > 0:
                current_progress = processed_securities + i + 1
                progress_percentage = (current_progress / total_securities) * 100
                try:
                    progress.update_progress(
                        current_securities=current_progress,
                        current_batch=batch_index + 1,
                        message=f"处理批次 {batch_index + 1}/{total_batches} - 证券 {current_progress}/{total_securities} ({progress_percentage:.1f}%)",
                    )
                except Exception:  # noqa: BLE001
                    LOGGER.debug("Progress handler update_progress 失败", extra=LOG_EXTRA)
                    progress = None
        except Exception as e:
            LOGGER.error(
                "计算因子失败 ts_code=%s err=%s",
                ts_code,
                str(e),
                extra=LOG_EXTRA,
            )
            validation_stats["skipped"] += 1
    
    # 批次处理完成，更新最终进度
    if progress and total_securities > 0:
        final_progress = processed_securities + len(ts_codes)
        progress_percentage = (final_progress / total_securities) * 100
        try:
            progress.update_progress(
                current_securities=final_progress,
                current_batch=batch_index + 1,
                message=f"批次 {batch_index + 1}/{total_batches} 处理完成 - 证券 {final_progress}/{total_securities} ({progress_percentage:.1f}%)",
            )
        except Exception:  # noqa: BLE001
            LOGGER.debug("Progress handler update_progress 失败", extra=LOG_EXTRA)

    return batch_results


def _check_data_availability(
    broker: DataBroker,
    ts_code: str,
    trade_date: str,
    specs: Sequence[FactorSpec],
) -> bool:
    """检查证券数据是否足够计算所有请求的因子"""
    # 检查数据是否满足基本要求
    if not check_data_sufficiency(ts_code, trade_date):
        return False
    
    # 检查快照数据
    latest_fields = broker.fetch_latest(
        ts_code,
        trade_date,
        ["daily.close", "daily_basic.turnover_rate", "daily_basic.pe", "daily_basic.pb"]
    )
    required_fields = {"daily.close", "daily_basic.turnover_rate"}
    for field in required_fields:
        if latest_fields.get(field) is None:
            LOGGER.warning(
                "缺少必需字段 field=%s ts_code=%s date=%s",
                field, ts_code, trade_date,
                extra=LOG_EXTRA
            )
            return False
    
    # 获取收盘价数据并做最终检查
    close_price = latest_fields.get("daily.close")
    if close_price is None or float(close_price) <= 0:
        LOGGER.debug(
            "收盘价数据无效 ts_code=%s date=%s price=%s",
            ts_code, trade_date, close_price,
            extra=LOG_EXTRA
        )
        return False
        
    return True  # 所有检查都通过


def _check_batch_data_availability(
    broker: DataBroker,
    ts_codes: List[str],
    trade_date: str,
    specs: Sequence[FactorSpec],
) -> Set[str]:
    """批次化检查多个证券的数据可用性，使用DataBroker的批次查询方法
    
    Args:
        broker: 数据代理
        ts_codes: 证券代码列表
        trade_date: 交易日期
        specs: 因子规格列表
        
    Returns:
        数据可用的证券代码集合
    """
    if not ts_codes:
        return set()
    
    available_codes = set()
    
    # 使用DataBroker的批次化检查数据充分性
    sufficient_codes = broker.check_batch_data_sufficiency(ts_codes, trade_date)
    
    if not sufficient_codes:
        return available_codes
    
    # 使用DataBroker的批次化获取最新字段数据
    required_fields = ["daily.close", "daily_basic.turnover_rate"]
    batch_fields_data = broker.fetch_batch_latest(list(sufficient_codes), trade_date, required_fields)
    
    # 检查每个证券的必需字段
    for ts_code in sufficient_codes:
        fields_data = batch_fields_data.get(ts_code, {})
        
        # 检查必需字段是否存在
        has_all_required = True
        for field in required_fields:
            if fields_data.get(field) is None:
                LOGGER.debug(
                    "批次化检查缺少字段 field=%s ts_code=%s date=%s",
                    field, ts_code, trade_date,
                    extra=LOG_EXTRA
                )
                has_all_required = False
                break
        
        if not has_all_required:
            continue
        
        # 检查收盘价有效性
        close_price = fields_data.get("daily.close")
        if close_price is None or float(close_price) <= 0:
            LOGGER.debug(
                "批次化检查收盘价无效 ts_code=%s date=%s price=%s",
                ts_code, trade_date, close_price,
                extra=LOG_EXTRA
            )
            continue
        
        available_codes.add(ts_code)
    
    LOGGER.debug(
        "批次化数据可用性检查完成 总证券数=%s 可用证券数=%s",
        len(ts_codes), len(available_codes),
        extra=LOG_EXTRA
    )
    
    return available_codes


def _detect_and_handle_outliers(
    values: Dict[str, float | None],
    ts_code: str,
) -> Dict[str, float | None]:
    """检测并处理因子值中的异常值"""
    result = values.copy()
    outliers_found = False
    
    # 动量因子异常值检测
    for key in [k for k in values if k.startswith("mom_") and values[k] is not None]:
        value = values[key]
        # 异常值检测规则：动量值绝对值大于3视为异常
        if abs(value) > 3.0:
            LOGGER.debug(
                "检测到动量因子异常值 ts_code=%s factor=%s value=%.4f",
                ts_code, key, value,
                extra=LOG_EXTRA,
            )
            # 限制到合理范围
            result[key] = min(3.0, max(-3.0, value))
            outliers_found = True
    
    # 波动率因子异常值检测
    for key in [k for k in values if k.startswith("volat_") and values[k] is not None]:
        value = values[key]
        # 异常值检测规则：波动率大于100%视为异常
        if value > 1.0:
            LOGGER.debug(
                "检测到波动率因子异常值 ts_code=%s factor=%s value=%.4f",
                ts_code, key, value,
                extra=LOG_EXTRA,
            )
            # 限制到合理范围
            result[key] = min(1.0, value)
            outliers_found = True
    
    if outliers_found:
        LOGGER.debug(
            "处理后因子值 ts_code=%s values=%s",
            ts_code, {k: f"{v:.4f}" for k, v in result.items() if v is not None},
            extra=LOG_EXTRA,
        )
        
    return result


def _compute_security_factors(
    broker: DataBroker,
    ts_code: str,
    trade_date: str,
    specs: Sequence[FactorSpec],
    *,
    latest_fields: Optional[Mapping[str, object]] = None,
) -> Dict[str, float | None]:
    """计算单个证券的因子值
    
    包括基础因子、扩展因子和情绪因子的计算。
    """
    # 确定所需的最大窗口大小
    # 包含所有因子（基础因子和扩展因子）的窗口需求
    close_windows = [spec.window for spec in specs]
    turnover_windows = [spec.window for spec in specs if _factor_prefix(spec.name) == "turn"]
    max_close_window = max(close_windows) if close_windows else 0
    max_turn_window = max(turnover_windows) if turnover_windows else 0
    
    # 确保窗口大小至少满足扩展因子的需求
    from app.features.extended_factors import EXTENDED_FACTORS
    extended_windows = [spec.window for spec in EXTENDED_FACTORS]
    max_extended_window = max(extended_windows) if extended_windows else 0
    max_close_window = max(max_close_window, max_extended_window)

    # 获取所需的时间序列数据
    close_series = _fetch_series_values(
        broker,
        "daily",
        "close",
        ts_code,
        trade_date,
        max_close_window,
    )
    
    # 数据有效性检查
    # 检查是否有窗口为0的因子
    has_zero_window = any(spec.window == 0 for spec in specs)
    
    # 如果有窗口为0的因子，使用专门的数据检查函数
    if has_zero_window:
        if not check_data_sufficiency_for_zero_window(ts_code, trade_date):
            LOGGER.debug(
                "数据不满足计算条件(窗口为0) ts_code=%s date=%s",
                ts_code, trade_date,
                extra=LOG_EXTRA
            )
            return {}
    else:
        if not check_data_sufficiency(ts_code, trade_date):
            LOGGER.debug(
                "数据不满足计算条件 ts_code=%s date=%s",
                ts_code, trade_date,
                extra=LOG_EXTRA
            )
            return {}
        
    turnover_series = _fetch_series_values(
        broker,
        "daily_basic",
        "turnover_rate",
        ts_code,
        trade_date,
        max_turn_window,
    )
    
    # 获取成交量数据用于扩展因子计算
    volume_series = _fetch_series_values(
        broker,
        "daily",
        "vol",
        ts_code,
        trade_date,
        max_close_window,  # 使用与价格相同的窗口
    )

    # 获取最新字段值
    if latest_fields is None:
        latest_fields = broker.fetch_latest(
            ts_code,
            trade_date,
            _LATEST_BASE_FIELDS,
        )
    else:
        latest_fields = dict(latest_fields)

    # 计算各个因子值
    results: Dict[str, float | None] = {}
    for spec in specs:
        prefix = _factor_prefix(spec.name)
        if prefix == "mom":
            if len(close_series) >= spec.window:
                results[spec.name] = momentum(close_series, spec.window)
            else:
                results[spec.name] = None
        elif prefix == "volat":
            if len(close_series) >= 2:
                results[spec.name] = volatility(close_series, spec.window)
            else:
                results[spec.name] = None
        elif prefix == "turn":
            if len(turnover_series) >= spec.window:
                results[spec.name] = rolling_mean(turnover_series, spec.window)
            else:
                results[spec.name] = None
        elif spec.name == "val_pe_score":
            pe = latest_fields.get("daily_basic.pe")
            results[spec.name] = _valuation_score(pe, scale=12.0)
        elif spec.name == "val_pb_score":
            pb = latest_fields.get("daily_basic.pb")
            results[spec.name] = _valuation_score(pb, scale=2.5)
        elif spec.name == "volume_ratio_score":
            volume_ratio = latest_fields.get("daily_basic.volume_ratio")
            results[spec.name] = _volume_ratio_score(volume_ratio)
        else:
            # 检查是否为扩展因子
            from app.features.extended_factors import EXTENDED_FACTORS
            extended_factor_names = [spec.name for spec in EXTENDED_FACTORS]
            
            # 检查是否为情绪因子
            sentiment_factor_names = ["sent_momentum", "sent_impact", "sent_market", "sent_divergence"]
            
            if spec.name in extended_factor_names or spec.name in sentiment_factor_names:
                # 扩展因子和情绪因子将在后续统一计算，这里不记录日志
                pass
            else:
                LOGGER.info(
                    "忽略未识别的因子 name=%s ts_code=%s",
                    spec.name,
                    ts_code,
                    extra=LOG_EXTRA,
                )
    
    # 计算扩展因子值
    calculator = ExtendedFactors()
    extended_factors = calculator.compute_all_factors(close_series, volume_series, ts_code, trade_date)
    results.update(extended_factors)
    
    # 计算情感因子
    sentiment_calculator = SentimentFactors()
    sentiment_factors = sentiment_calculator.compute_stock_factors(broker, ts_code, trade_date)
    if sentiment_factors:
        results.update(sentiment_factors)
    
    # 计算风险和估值因子
    value_risk_calculator = ValueRiskFactors()
    
    # 计算val_multiscore
    val_multiscore = value_risk_calculator.compute_val_multiscore(
        pe=latest_fields.get("daily_basic.pe"),
        pb=latest_fields.get("daily_basic.pb"),
        ps=latest_fields.get("daily_basic.ps"),
        dv=latest_fields.get("daily_basic.dv_ratio")
    )
    if val_multiscore is not None:
        results["val_multiscore"] = val_multiscore
        
    # 计算risk_penalty
    volat_20 = results.get("volat_20")
    turnover = latest_fields.get("daily_basic.turnover_rate")
    current_price = latest_fields.get("daily.close")
    avg_price = rolling_mean(close_series, 20) if len(close_series) >= 20 else None
    
    risk_penalty = value_risk_calculator.compute_risk_penalty(
        volatility=volat_20,
        turnover=turnover,
        price=current_price,
        avg_price=avg_price
    )
    if risk_penalty is not None:
        results["risk_penalty"] = risk_penalty
    
    # 确保返回结果不为空
    if not any(v is not None for v in results.values()):
        return {}
    
    return results


def _persist_factor_rows(
    trade_date: str,
    rows: Sequence[tuple[str, Dict[str, float | None]]],
    specs: Sequence[FactorSpec],
) -> None:
    """优化的因子结果持久化函数，支持批量写入"""
    if not rows:
        return
    
    columns = sorted({spec.name for spec in specs})
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # SQL语句准备
    insert_columns = ["ts_code", "trade_date", "updated_at", *columns]
    placeholders = ", ".join(["?"] * len(insert_columns))
    update_clause = ", ".join(
        f"{column}=excluded.{column}" for column in ["updated_at", *columns]
    )
    sql = (
        f"INSERT INTO factors ({', '.join(insert_columns)}) "
        f"VALUES ({placeholders}) "
        f"ON CONFLICT(ts_code, trade_date) DO UPDATE SET {update_clause}"
    )

    # 准备批量写入数据
    batch_size = 500  # 批处理大小
    batch_payloads = []
    
    for ts_code, values in rows:
        # 过滤掉全部为None的行
        if not any(values.get(col) is not None for col in columns):
            continue
            
        payload = [ts_code, trade_date, timestamp]
        payload.extend(values.get(column) for column in columns)
        batch_payloads.append(payload)
    
    if not batch_payloads:
        LOGGER.debug("无可持久化的有效因子数据", extra=LOG_EXTRA)
        return
    
    # 执行批量写入
    total_inserted = 0
    with db_session() as conn:
        # 分批执行以避免SQLite参数限制
        for i in range(0, len(batch_payloads), batch_size):
            batch = batch_payloads[i:i+batch_size]
            try:
                conn.executemany(sql, batch)
                batch_count = len(batch)
                total_inserted += batch_count
                
                if batch_count % (batch_size * 5) == 0:
                    LOGGER.debug(
                        "因子数据持久化进度: %s/%s",
                        min(i + batch_size, len(batch_payloads)),
                        len(batch_payloads),
                        extra=LOG_EXTRA,
                    )
            except sqlite3.Error as e:
                LOGGER.error(
                    "因子数据持久化失败 批次=%s-%s err=%s",
                    i, min(i + batch_size, len(batch_payloads)),
                    str(e),
                    extra=LOG_EXTRA,
                )
    
    LOGGER.info(
        "因子数据持久化完成 写入记录数=%s 总记录数=%s",
        total_inserted,
        len(batch_payloads),
        extra=LOG_EXTRA,
    )


def _ensure_factor_columns(specs: Sequence[FactorSpec]) -> None:
    pending = {spec.name for spec in specs if _IDENTIFIER_RE.match(spec.name)}
    if not pending:
        return
    with db_session() as conn:
        existing_rows = conn.execute("PRAGMA table_info(factors)").fetchall()
        existing = {row["name"] for row in existing_rows}
        for column in sorted(pending - existing):
            conn.execute(f"ALTER TABLE factors ADD COLUMN {column} REAL")


def _fetch_series_values(
    broker: DataBroker,
    table: str,
    column: str,
    ts_code: str,
    trade_date: str,
    window: int,
) -> List[float]:
    if window <= 0:
        return []
    series = broker.fetch_series(table, column, ts_code, trade_date, window)
    values: List[float] = []
    for _dt, raw in series:
        try:
            values.append(float(raw))
        except (TypeError, ValueError):
            continue
    return values


def _factor_prefix(name: str) -> str:
    return name.split("_", 1)[0] if name else ""


def _valuation_score(value: object, *, scale: float) -> float:
    """计算估值指标的标准化分数"""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    
    # 有效性检查
    if numeric <= 0:
        return 0.0
        
    # 异常值处理：限制估值指标的上限
    max_limit = scale * 10  # 设置十倍scale为上限
    if numeric > max_limit:
        numeric = max_limit
    
    # 计算分数
    score = scale / (scale + numeric)
    return max(0.0, min(1.0, score))


def _check_stock_exists(broker: DataBroker, ts_code: str, trade_date: str) -> bool:
    """检查指定日期股票是否存在交易数据"""
    with db_session(read_only=True) as session:
        result = session.execute(
            """
            SELECT 1 FROM daily 
            WHERE ts_code = :ts_code 
            AND trade_date = :trade_date
            LIMIT 1
            """,
            {"ts_code": ts_code, "trade_date": trade_date}
        ).fetchone()
        return bool(result)

def _volume_ratio_score(value: object) -> float:
    """计算量比指标的标准化分数"""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    
    # 有效性检查
    if numeric < 0:
        numeric = 0.0
        
    # 异常值处理：设置量比上限为20
    if numeric > 20:
        numeric = 20
    
    return max(0.0, min(1.0, numeric / 10.0))
