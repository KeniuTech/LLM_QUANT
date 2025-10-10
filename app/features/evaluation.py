"""Factor performance evaluation utilities."""
from datetime import date, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from app.features.factors import (
    DEFAULT_FACTORS,
    FactorResult,
    FactorSpec,
    compute_factor_range,
    lookup_factor_spec,
)
from app.utils.data_access import DataBroker
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
            "turnover_rate": self.turnover_rate or 0.0
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
    
    # 导入进度状态模块
    from app.ui.progress_state import factor_progress
    
    # 开始因子计算进度（在异步线程中不直接访问factor_progress）
    # factor_progress.start_calculation(
    #     total_securities=len(universe) if universe else 0,
    #     message=f"开始评估因子 {factor_name}"
    # )
    
    try:
        spec = lookup_factor_spec(factor_name) or FactorSpec(factor_name, 0)

        factor_results = compute_factor_range(
            start_date,
            end_date,
            factors=[spec],
            ts_codes=universe,
            skip_existing=True,
        )
        
        # 因子计算完成（在异步线程中不直接访问factor_progress）
        # factor_progress.complete_calculation(
        #     message=f"因子 {factor_name} 评估完成"
        # )
        
    except Exception as e:
        # 因子计算失败（在异步线程中不直接访问factor_progress）
        # factor_progress.complete_calculation(
        #     message=f"因子 {factor_name} 评估失败: {str(e)}",
        #     success=False
        # )
        raise
    
    # 按日期分组
    date_groups: Dict[date, List[FactorResult]] = {}
    for result in factor_results:
        if result.trade_date not in date_groups:
            date_groups[result.trade_date] = []
        date_groups[result.trade_date].append(result)
    
    # 计算每日IC值和RankIC值
    broker = DataBroker()
    for curr_date, results in sorted(date_groups.items()):
        next_date = curr_date + timedelta(days=1)
        
        # 获取因子值和次日收益率
        factor_values = []
        next_returns = []
        
        for result in results:
            factor_val = result.values.get(factor_name)
            if factor_val is None:
                continue
                
            # 获取次日收益率
            next_close = broker.fetch_latest(
                result.ts_code,
                next_date.strftime("%Y%m%d"),
                ["daily.close"]
            ).get("daily.close")
            
            curr_close = broker.fetch_latest(
                result.ts_code,
                curr_date.strftime("%Y%m%d"),
                ["daily.close"]
            ).get("daily.close")
            
            if next_close and curr_close and curr_close > 0:
                ret = (next_close - curr_close) / curr_close
                factor_values.append(factor_val)
                next_returns.append(ret)
        
        if len(factor_values) >= 20:  # 需要足够多的样本
            # 计算IC
            ic, _ = stats.pearsonr(factor_values, next_returns)
            performance.ic_series.append(ic)
            
            # 计算RankIC
            rank_ic, _ = stats.spearmanr(factor_values, next_returns)
            performance.rank_ic_series.append(rank_ic)
            
            # 计算多空组合收益
            sorted_pairs = sorted(zip(factor_values, next_returns),
                                key=lambda x: x[0])
            n = len(sorted_pairs) // 5  # 五分位
            if n > 0:
                top_returns = [r for _, r in sorted_pairs[-n:]]
                bottom_returns = [r for _, r in sorted_pairs[:n]]
                spread = np.mean(top_returns) - np.mean(bottom_returns)
                performance.return_spreads.append(spread)
    
    # 计算Sharpe比率
    if performance.return_spreads:
        annual_factor = np.sqrt(252)  # 交易日数
        returns_mean = np.mean(performance.return_spreads)
        returns_std = np.std(performance.return_spreads)
        if returns_std > 0:
            performance.sharpe_ratio = returns_mean / returns_std * annual_factor
    
    # 估算换手率
    if factor_results:
        dates = sorted(date_groups.keys())
        turnovers = []
        for i in range(1, len(dates)):
            prev_results = date_groups[dates[i-1]]
            curr_results = date_groups[dates[i]]
            
            # 计算组合变化
            prev_top = {r.ts_code for r in prev_results 
                       if r.values.get(factor_name, float('-inf')) > np.percentile(
                           [res.values.get(factor_name, float('-inf')) 
                            for res in prev_results], 80)}
            curr_top = {r.ts_code for r in curr_results
                       if r.values.get(factor_name, float('-inf')) > np.percentile(
                           [res.values.get(factor_name, float('-inf'))
                            for res in curr_results], 80)}
            
            # 计算换手率
            if prev_top and curr_top:
                turnover = len(prev_top ^ curr_top) / len(prev_top | curr_top)
                turnovers.append(turnover)
                
        if turnovers:
            performance.turnover_rate = np.mean(turnovers)
    
    return performance


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
