"""Factor performance evaluation utilities."""
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from app.features.factors import (
    DEFAULT_FACTORS,
    FactorSpec,
    lookup_factor_spec,
)
from app.utils.db import db_session
from app.utils.logging import get_logger

LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "factor_evaluation"}


class FactorPerformance:
    """因子表现评估结果。"""
    
    def __init__(self, factor_name: str) -> None:
        self.factor_name = factor_name
        self.ic_series: List[float] = []
        self.rank_ic_series: List[float] = []
        self.return_spreads: List[float] = []
        self.sharpe_ratio: Optional[float] = None
        self.turnover_rate: Optional[float] = None
        self.sample_size: int = 0
        
    @property
    def ic_mean(self) -> float:
        """平均IC。"""
        return np.mean(self.ic_series) if self.ic_series else 0.0
        
    @property
    def ic_std(self) -> float:
        """IC标准差。"""
        return np.std(self.ic_series) if self.ic_series else 0.0
        
    @property
    def ic_ir(self) -> float:
        """信息比率。"""
        return self.ic_mean / self.ic_std if self.ic_std > 0 else 0.0
        
    @property
    def rank_ic_mean(self) -> float:
        """平均RankIC。"""
        return np.mean(self.rank_ic_series) if self.rank_ic_series else 0.0
        
    def to_dict(self) -> Dict[str, float]:
        """转换为字典格式。"""
        return {
            "ic_mean": self.ic_mean,
            "ic_std": self.ic_std,
            "ic_ir": self.ic_ir,
            "rank_ic_mean": self.rank_ic_mean,
            "sharpe_ratio": self.sharpe_ratio or 0.0,
            "turnover_rate": self.turnover_rate or 0.0,
            "sample_size": float(self.sample_size),
        }


@dataclass
class FactorPortfolioReport:
    weights: Dict[str, float]
    combined: FactorPerformance
    components: Dict[str, FactorPerformance]

    def to_dict(self) -> Dict[str, object]:
        return {
            "weights": dict(self.weights),
            "combined": self.combined.to_dict(),
            "components": {name: perf.to_dict() for name, perf in self.components.items()},
        }


def evaluate_factor(
    factor_name: str,
    start_date: date,
    end_date: date,
    universe: Optional[List[str]] = None,
) -> FactorPerformance:
    """评估单个因子的预测能力。
    
    Args:
        factor_name: 因子名称
        start_date: 起始日期
        end_date: 结束日期
        universe: 可选的股票池
        
    Returns:
        因子表现评估结果
    """
    performance = FactorPerformance(factor_name)
    spec = lookup_factor_spec(factor_name)
    factor_column = factor_name

    if spec is None:
        LOGGER.warning("未找到因子定义，仍尝试从数据库读取 factor=%s", factor_name, extra=LOG_EXTRA)

    normalized_universe = _normalize_universe(universe)
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    with db_session(read_only=True) as conn:
        if not _has_factor_column(conn, factor_column):
            LOGGER.warning("factors 表缺少列 %s，跳过评估", factor_column, extra=LOG_EXTRA)
            return performance
        trade_dates = _list_factor_dates(conn, start_str, end_str, normalized_universe)

    if not trade_dates:
        LOGGER.info("指定区间内未找到可用因子数据 factor=%s", factor_name, extra=LOG_EXTRA)
        return performance

    usable_trade_dates: List[str] = []

    for trade_date_str in trade_dates:
        with db_session(read_only=True) as conn:
            factor_map = _fetch_factor_cross_section(conn, factor_column, trade_date_str, normalized_universe)
            if not factor_map:
                continue
            next_trade = _next_trade_date(conn, trade_date_str)
            if not next_trade:
                continue
            curr_close = _fetch_close_map(conn, trade_date_str, factor_map.keys())
            next_close = _fetch_close_map(conn, next_trade, factor_map.keys())

        factor_values: List[float] = []
        returns: List[float] = []
        for ts_code, value in factor_map.items():
            curr = curr_close.get(ts_code)
            nxt = next_close.get(ts_code)
            if curr is None or nxt is None or curr <= 0:
                continue
            factor_values.append(value)
            returns.append((nxt - curr) / curr)

        if len(factor_values) < 20:
            continue

        values_array = np.array(factor_values, dtype=float)
        returns_array = np.array(returns, dtype=float)
        if np.ptp(values_array) <= 1e-9 or np.ptp(returns_array) <= 1e-9:
            LOGGER.debug(
                "因子/收益序列波动不足，跳过 date=%s span_factor=%.6f span_return=%.6f",
                trade_date_str,
                float(np.ptp(values_array)),
                float(np.ptp(returns_array)),
                extra=LOG_EXTRA,
            )
            continue

        try:
            ic, _ = stats.pearsonr(values_array, returns_array)
            rank_ic, _ = stats.spearmanr(values_array, returns_array)
        except Exception as exc:  # noqa: BLE001
            LOGGER.debug("IC 计算失败 date=%s err=%s", trade_date_str, exc, extra=LOG_EXTRA)
            continue

        if not (np.isfinite(ic) and np.isfinite(rank_ic)):
            LOGGER.debug(
                "相关系数结果无效 date=%s ic=%s rank_ic=%s",
                trade_date_str,
                ic,
                rank_ic,
                extra=LOG_EXTRA,
            )
            continue

        performance.ic_series.append(ic)
        performance.rank_ic_series.append(rank_ic)
        usable_trade_dates.append(trade_date_str)

        sorted_pairs = sorted(zip(values_array.tolist(), returns_array.tolist()), key=lambda item: item[0])
        quantile = len(sorted_pairs) // 5
        if quantile > 0:
            top_returns = [ret for _, ret in sorted_pairs[-quantile:]]
            bottom_returns = [ret for _, ret in sorted_pairs[:quantile]]
            spread = float(np.mean(top_returns) - np.mean(bottom_returns))
            performance.return_spreads.append(spread)

    if performance.return_spreads:
        returns_mean = float(np.mean(performance.return_spreads))
        returns_std = float(np.std(performance.return_spreads))
        if returns_std > 0:
            performance.sharpe_ratio = returns_mean / returns_std * np.sqrt(252.0)

    performance.sample_size = len(usable_trade_dates)
    performance.turnover_rate = _estimate_turnover_rate(
        factor_column,
        usable_trade_dates,
        normalized_universe,
    )
    return performance


def optimize_factor_weights(
    factor_names: Sequence[str],
    start_date: date,
    end_date: date,
    *,
    universe: Optional[Sequence[str]] = None,
    method: str = "ic_mean",
) -> Tuple[Dict[str, float], Dict[str, FactorPerformance]]:
    """Derive factor weights based on historical performance metrics."""

    if not factor_names:
        raise ValueError("factor_names must not be empty")

    normalized_universe = list(universe) if universe else None
    performances: Dict[str, FactorPerformance] = {}
    scores: Dict[str, float] = {}

    for name in factor_names:
        perf = evaluate_factor(name, start_date, end_date, normalized_universe)
        performances[name] = perf
        if method == "ic_ir":
            metric = perf.ic_ir
        elif method == "rank_ic":
            metric = perf.rank_ic_mean
        else:
            metric = perf.ic_mean
        scores[name] = max(0.0, float(metric))

    weights = _normalize_weight_map(factor_names, scores)
    return weights, performances


def evaluate_factor_portfolio(
    factor_names: Sequence[str],
    start_date: date,
    end_date: date,
    *,
    universe: Optional[Sequence[str]] = None,
    weights: Optional[Dict[str, float]] = None,
    method: str = "ic_mean",
) -> FactorPortfolioReport:
    """Evaluate a weighted combination of factors."""

    if not factor_names:
        raise ValueError("factor_names must not be empty")

    normalized_universe = _normalize_universe(universe)

    if weights is None:
        weights, performances = optimize_factor_weights(
            factor_names,
            start_date,
            end_date,
            universe=universe,
            method=method,
        )
    else:
        weights = _normalize_weight_map(factor_names, weights)
        performances = {
            name: evaluate_factor(name, start_date, end_date, universe)
            for name in factor_names
        }

    weight_vector = [weights[name] for name in factor_names]
    combined = _evaluate_combined_factor(
        factor_names,
        weight_vector,
        start_date,
        end_date,
        normalized_universe,
    )

    return FactorPortfolioReport(weights=weights, combined=combined, components=performances)


def combine_factors(
    factor_names: Sequence[str],
    weights: Optional[Sequence[float]] = None
) -> FactorSpec:
    """组合多个因子。
    
    Args:
        factor_names: 因子名称列表
        weights: 可选的权重列表，默认等权重
        
    Returns:
        组合因子的规格
    """
    if not weights:
        weights = [1.0 / len(factor_names)] * len(factor_names)
    
    name = "combined_" + "_".join(factor_names)
    window = max(
        spec.window
        for spec in DEFAULT_FACTORS
        if spec.name in factor_names
    )
    
    return FactorSpec(name, window)


def _normalize_weight_map(
    factor_names: Sequence[str],
    weights: Dict[str, float],
) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for name in factor_names:
        if name not in weights:
            continue
        try:
            value = float(weights[name])
        except (TypeError, ValueError):
            continue
        if np.isnan(value) or value <= 0.0:
            continue
        normalized[name] = value

    if len(normalized) != len(factor_names):
        weight = 1.0 / len(factor_names)
        return {name: weight for name in factor_names}

    total = sum(normalized.values())
    if total <= 0.0:
        weight = 1.0 / len(factor_names)
        return {name: weight for name in factor_names}

    return {name: value / total for name, value in normalized.items()}


def _normalize_universe(universe: Optional[Sequence[str]]) -> Optional[Tuple[str, ...]]:
    if not universe:
        return None
    unique: Dict[str, None] = {}
    for code in universe:
        value = (code or "").strip().upper()
        if value:
            unique.setdefault(value, None)
    return tuple(unique.keys()) if unique else None


def _has_factor_column(conn, column: str) -> bool:
    rows = conn.execute("PRAGMA table_info(factors)").fetchall()
    available = {row["name"] for row in rows}
    return column in available


def _list_factor_dates(conn, start: str, end: str, universe: Optional[Tuple[str, ...]]) -> List[str]:
    params: List[str] = [start, end]
    query = (
        "SELECT DISTINCT trade_date FROM factors "
        "WHERE trade_date BETWEEN ? AND ?"
    )
    if universe:
        placeholders = ",".join("?" for _ in universe)
        query += f" AND ts_code IN ({placeholders})"
        params.extend(universe)
    query += " ORDER BY trade_date"
    rows = conn.execute(query, params).fetchall()
    return [row["trade_date"] for row in rows if row and row["trade_date"]]


def _fetch_factor_cross_section(
    conn,
    column: str,
    trade_date: str,
    universe: Optional[Tuple[str, ...]],
) -> Dict[str, float]:
    params: List[str] = [trade_date]
    query = f"SELECT ts_code, {column} AS value FROM factors WHERE trade_date = ? AND {column} IS NOT NULL"
    if universe:
        placeholders = ",".join("?" for _ in universe)
        query += f" AND ts_code IN ({placeholders})"
        params.extend(universe)
    rows = conn.execute(query, params).fetchall()
    result: Dict[str, float] = {}
    for row in rows:
        ts_code = row["ts_code"]
        value = row["value"]
        if ts_code is None or value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric):
            continue
        result[ts_code] = numeric
    return result


def _fetch_factor_matrix(
    conn,
    columns: Sequence[str],
    trade_date: str,
    universe: Optional[Tuple[str, ...]],
) -> Dict[str, List[float]]:
    if not columns:
        return {}
    params: List[object] = [trade_date]
    column_clause = ", ".join(columns)
    query = f"SELECT ts_code, {column_clause} FROM factors WHERE trade_date = ?"
    if universe:
        placeholders = ",".join("?" for _ in universe)
        query += f" AND ts_code IN ({placeholders})"
        params.extend(universe)
    rows = conn.execute(query, params).fetchall()
    matrix: Dict[str, List[float]] = {}
    for row in rows:
        ts_code = row["ts_code"]
        if not ts_code:
            continue
        vector: List[float] = []
        valid = True
        for column in columns:
            value = row[column]
            if value is None:
                valid = False
                break
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                valid = False
                break
            if not np.isfinite(numeric):
                valid = False
                break
            vector.append(numeric)
        if not valid:
            continue
        matrix[ts_code] = vector
    return matrix


def _next_trade_date(conn, trade_date: str) -> Optional[str]:
    row = conn.execute(
        "SELECT MIN(trade_date) AS next_date FROM daily WHERE trade_date > ?",
        (trade_date,),
    ).fetchone()
    next_date = row["next_date"] if row else None
    return next_date


def _fetch_close_map(conn, trade_date: str, codes: Sequence[str]) -> Dict[str, float]:
    if not codes:
        return {}
    placeholders = ",".join("?" for _ in codes)
    params = [trade_date, *codes]
    rows = conn.execute(
        f"""
        SELECT ts_code, close
        FROM daily
        WHERE trade_date = ?
          AND ts_code IN ({placeholders})
          AND close IS NOT NULL
        """,
        params,
    ).fetchall()
    result: Dict[str, float] = {}
    for row in rows:
        ts_code = row["ts_code"]
        value = row["close"]
        if ts_code is None or value is None:
            continue
        try:
            result[ts_code] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def _estimate_turnover_from_maps(
    series: Sequence[Tuple[str, Dict[str, float]]],
) -> Optional[float]:
    if len(series) < 2:
        return None
    turnovers: List[float] = []
    for idx in range(1, len(series)):
        _, prev_map = series[idx - 1]
        _, curr_map = series[idx]
        if not prev_map or not curr_map:
            continue
        prev_threshold = np.percentile(list(prev_map.values()), 80)
        curr_threshold = np.percentile(list(curr_map.values()), 80)
        prev_top = {code for code, value in prev_map.items() if value >= prev_threshold}
        curr_top = {code for code, value in curr_map.items() if value >= curr_threshold}
        union = prev_top | curr_top
        if not union:
            continue
        turnover = len(prev_top ^ curr_top) / len(union)
        turnovers.append(turnover)
    if turnovers:
        return float(np.mean(turnovers))
    return None


def _estimate_turnover_rate(
    factor_name: str,
    trade_dates: Sequence[str],
    universe: Optional[Tuple[str, ...]],
) -> Optional[float]:
    if not trade_dates:
        return None
    turnovers: List[float] = []
    for idx in range(1, len(trade_dates)):
        prev_date = trade_dates[idx - 1]
        curr_date = trade_dates[idx]
        with db_session(read_only=True) as conn:
            prev_map = _fetch_factor_cross_section(conn, factor_name, prev_date, universe)
            curr_map = _fetch_factor_cross_section(conn, factor_name, curr_date, universe)

        if not prev_map or not curr_map:
            continue

        prev_threshold = np.percentile(list(prev_map.values()), 80)
        curr_threshold = np.percentile(list(curr_map.values()), 80)
        prev_top = {code for code, value in prev_map.items() if value >= prev_threshold}
        curr_top = {code for code, value in curr_map.items() if value >= curr_threshold}
        if not prev_top and not curr_top:
            continue
        union = prev_top | curr_top
        if not union:
            continue
        turnover = len(prev_top ^ curr_top) / len(union)
        turnovers.append(turnover)

    if turnovers:
        return float(np.mean(turnovers))
    return None


def _evaluate_combined_factor(
    factor_names: Sequence[str],
    weights: Sequence[float],
    start_date: date,
    end_date: date,
    universe: Optional[Tuple[str, ...]],
) -> FactorPerformance:
    performance = FactorPerformance("portfolio")
    if not factor_names or not weights:
        return performance

    weight_array = np.array(weights, dtype=float)
    start_str = start_date.strftime("%Y%m%d")
    end_str = end_date.strftime("%Y%m%d")

    with db_session(read_only=True) as conn:
        trade_dates = _list_factor_dates(conn, start_str, end_str, universe)

    combined_series: List[Tuple[str, Dict[str, float]]] = []

    for trade_date_str in trade_dates:
        with db_session(read_only=True) as conn:
            matrix = _fetch_factor_matrix(conn, factor_names, trade_date_str, universe)
            if not matrix:
                continue
            next_trade = _next_trade_date(conn, trade_date_str)
            if not next_trade:
                continue
            curr_close = _fetch_close_map(conn, trade_date_str, matrix.keys())
            next_close = _fetch_close_map(conn, next_trade, matrix.keys())

        factor_values: List[float] = []
        returns: List[float] = []
        combined_map: Dict[str, float] = {}
        for ts_code, vector in matrix.items():
            curr = curr_close.get(ts_code)
            nxt = next_close.get(ts_code)
            if curr is None or nxt is None or curr <= 0:
                continue
            combined_value = float(np.dot(weight_array, np.array(vector, dtype=float)))
            factor_values.append(combined_value)
            returns.append((nxt - curr) / curr)
            combined_map[ts_code] = combined_value

        if len(factor_values) < 20:
            continue

        values_array = np.array(factor_values, dtype=float)
        returns_array = np.array(returns, dtype=float)
        if np.ptp(values_array) <= 1e-9 or np.ptp(returns_array) <= 1e-9:
            continue

        try:
            ic, _ = stats.pearsonr(values_array, returns_array)
            rank_ic, _ = stats.spearmanr(values_array, returns_array)
        except Exception:  # noqa: BLE001
            continue

        if not (np.isfinite(ic) and np.isfinite(rank_ic)):
            continue

        performance.ic_series.append(float(ic))
        performance.rank_ic_series.append(float(rank_ic))
        combined_series.append((trade_date_str, combined_map))

        sorted_pairs = sorted(zip(values_array.tolist(), returns_array.tolist()), key=lambda item: item[0])
        quantile = len(sorted_pairs) // 5
        if quantile > 0:
            top_returns = [ret for _, ret in sorted_pairs[-quantile:]]
            bottom_returns = [ret for _, ret in sorted_pairs[:quantile]]
            performance.return_spreads.append(float(np.mean(top_returns) - np.mean(bottom_returns)))

    if performance.return_spreads:
        returns_mean = float(np.mean(performance.return_spreads))
        returns_std = float(np.std(performance.return_spreads))
        if returns_std > 0:
            performance.sharpe_ratio = returns_mean / returns_std * np.sqrt(252.0)

    performance.sample_size = len(performance.ic_series)
    performance.turnover_rate = _estimate_turnover_from_maps(combined_series)
    return performance
