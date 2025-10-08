"""Feature engineering for signals and indicator computation."""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, date, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from app.core.indicators import momentum, rolling_mean, volatility
from app.data.schema import initialize_database
from app.utils.data_access import DataBroker
from app.utils.db import db_session
from app.utils.logging import get_logger
# 导入扩展因子模块
from app.features.extended_factors import ExtendedFactors
from app.features.sentiment_factors import SentimentFactors
from app.features.value_risk_factors import ValueRiskFactors
# 导入因子验证功能
from app.features.validation import check_data_sufficiency, detect_outliers


LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "factor_compute"}
_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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
        existing = _existing_factor_codes(trade_date_str)
        universe = [code for code in universe if code not in existing]
        if not universe:
            LOGGER.debug(
                "目标交易日因子已存在 trade_date=%s universe_size=%s",
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
    
    # 分批处理以优化性能
    for i in range(0, len(universe), batch_size):
        batch = universe[i:i+batch_size]
        batch_results = _compute_batch_factors(broker, batch, trade_date_str, specs, validation_stats)
        
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
        
    LOGGER.info(
        "因子计算完成 总数量:%s 成功:%s 失败:%s",
        len(universe), 
        validation_stats["success"],
        validation_stats["total"] - validation_stats["success"],
        extra=LOG_EXTRA,
    )
    
    return results


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


def _compute_batch_factors(
    broker: DataBroker,
    ts_codes: List[str],
    trade_date: str,
    specs: Sequence[FactorSpec],
    validation_stats: Dict[str, int],
) -> List[tuple[str, Dict[str, float | None]]]:
    """批量计算多个证券的因子值，提高计算效率"""
    batch_results = []
    
    for ts_code in ts_codes:
        try:
            # 先检查数据可用性
            if not _check_data_availability(broker, ts_code, trade_date, specs):
                validation_stats["data_missing"] += 1
                continue
                
            # 计算因子值
            values = _compute_security_factors(broker, ts_code, trade_date, specs)
            
            if values:
                # 检测并处理异常值
                cleaned_values = detect_outliers(values, ts_code, trade_date)
                if cleaned_values:
                    batch_results.append((ts_code, cleaned_values))
                    validation_stats["success"] += 1
                else:
                    validation_stats["outliers"] += 1
            else:
                validation_stats["skipped"] += 1
        except Exception as e:
            LOGGER.error(
                "计算因子失败 ts_code=%s err=%s",
                ts_code,
                str(e),
                extra=LOG_EXTRA,
            )
            validation_stats["skipped"] += 1
    
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
) -> Dict[str, float | None]:
    """计算单个证券的因子值
    
    包括基础因子、扩展因子和情绪因子的计算。
    """
    # 确定所需的最大窗口大小
    close_windows = [spec.window for spec in specs if _factor_prefix(spec.name) in {"mom", "volat"}]
    turnover_windows = [spec.window for spec in specs if _factor_prefix(spec.name) == "turn"]
    max_close_window = max(close_windows) if close_windows else 0
    max_turn_window = max(turnover_windows) if turnover_windows else 0

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
    latest_fields = broker.fetch_latest(
        ts_code,
        trade_date,
        [
            "daily_basic.pe",
            "daily_basic.pb",
            "daily_basic.ps",
            "daily_basic.volume_ratio",
            "daily.amount",
            "daily.vol",
            "daily_basic.dv_ratio",  # 股息率用于扩展因子
        ],
    )

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
            LOGGER.debug(
                "忽略未识别的因子 name=%s ts_code=%s",
                spec.name,
                ts_code,
                extra=LOG_EXTRA,
            )
    
    # 计算扩展因子值
    calculator = ExtendedFactors()
    extended_factors = calculator.compute_all_factors(close_series, volume_series)
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
