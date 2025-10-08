"""LLM cost control and budget management."""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

from .metrics import snapshot

LOGGER = logging.getLogger(__name__)
LOG_EXTRA = {"stage": "cost_control"}


@dataclass
class CostLimits:
    """Cost control limits configuration."""

    hourly_budget: float  # 每小时预算
    daily_budget: float   # 每日预算
    monthly_budget: float  # 每月预算
    model_weights: Dict[str, float] = field(default_factory=dict)  # 模型权重配置

    @classmethod
    def default(cls) -> CostLimits:
        """Create default cost limits."""
        return cls(
            hourly_budget=2.0,   # $2/hour
            daily_budget=20.0,   # $20/day
            monthly_budget=300.0, # $300/month
            model_weights={
                "gpt-4": 0.2,     # 限制GPT-4使用比例
                "gpt-3.5-turbo": 0.6,
                "llama2": 0.2
            }
        )


@dataclass
class ModelCosts:
    """Per-model cost configuration."""

    prompt_cost_per_1k: float
    completion_cost_per_1k: float
    min_tokens: int = 1

    def calculate(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost for token usage."""
        prompt_cost = max(self.min_tokens, prompt_tokens) / 1000 * self.prompt_cost_per_1k
        completion_cost = max(self.min_tokens, completion_tokens) / 1000 * self.completion_cost_per_1k
        return prompt_cost + completion_cost


class CostController:
    """Controls and manages LLM costs."""

    def __init__(self, limits: Optional[CostLimits] = None):
        """Initialize cost controller."""
        self.limits = limits or CostLimits.default()
        # Maintain model-specific pricing in lowercase so lookups remain case-insensitive.
        self._costs: Dict[str, ModelCosts] = {
            "gpt-4": ModelCosts(0.03, 0.06),
            "gpt-4-32k": ModelCosts(0.06, 0.12),
            "gpt-4o": ModelCosts(0.005, 0.015),
            "gpt-4o-mini": ModelCosts(0.0006, 0.0018),
            "gpt-4.1-mini": ModelCosts(0.0008, 0.002),
            "gpt-3.5-turbo": ModelCosts(0.0015, 0.002),
            "gpt-3.5-turbo-16k": ModelCosts(0.003, 0.004),
            "gpt-3.5": ModelCosts(0.0015, 0.002),
            "llama2": ModelCosts(0.0, 0.0),
            "llama3": ModelCosts(0.0, 0.0),
            "phi3": ModelCosts(0.0, 0.0),
            "qwen2": ModelCosts(0.0, 0.0),
            "codellama": ModelCosts(0.0, 0.0)
        }
        # Family-level fallbacks ensure close variants (e.g. gpt-4o-mini-2024) are charged.
        self._cost_prefixes: Dict[str, ModelCosts] = {
            "gpt-4o": self._costs["gpt-4o"],
            "gpt-4.1": self._costs["gpt-4.1-mini"],
            "gpt-4": self._costs["gpt-4"],
            "gpt-3.5": self._costs["gpt-3.5"],
            "llama3": self._costs["llama3"],
            "llama2": self._costs["llama2"],
            "phi3": self._costs["phi3"],
            "qwen2": self._costs["qwen2"],
        }
        self._usage_lock = threading.Lock()
        self._usage: Dict[str, List[Dict[str, Any]]] = {
            "hourly": [],
            "daily": [],
            "monthly": []
        }
        self._last_cleanup = time.time()
        self._cleanup_interval = 3600  # 1小时清理一次历史数据

    def can_use_model(self, model: str, prompt_tokens: int,
                     completion_tokens: int) -> bool:
        """检查是否允许使用指定模型."""
        # 检查成本限制
        if not self._check_budget_limits(model, prompt_tokens, completion_tokens):
            return False

        # 检查模型权重限制
        if not self._check_model_weights(model):
            return False

        return True

    def record_usage(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> bool:
        """记录模型使用情况，并返回是否仍在预算范围内."""

        cost = self._calculate_cost(model, prompt_tokens, completion_tokens)
        within_limits = self._check_budget_limits(model, prompt_tokens, completion_tokens)
        timestamp = time.time()

        usage = {
            "model": model,
            "timestamp": timestamp,
            "cost": cost,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens
        }

        with self._usage_lock:
            self._usage["hourly"].append(usage)
            self._usage["daily"].append(usage)
            self._usage["monthly"].append(usage)

            # 定期清理过期数据
            self._cleanup_old_usage(timestamp)

        if not within_limits:
            LOGGER.warning(
                "Cost limit exceeded after recording usage - model: %s cost=$%.4f",
                model,
                cost,
                extra=LOG_EXTRA,
            )

        return within_limits

    def get_current_costs(self) -> Dict[str, float]:
        """获取当前时段的成本统计."""
        with self._usage_lock:
            now = time.time()
            hour_ago = now - 3600
            day_ago = now - 86400
            month_ago = now - 2592000  # 30天

            hourly = sum(u["cost"] for u in self._usage["hourly"]
                        if u["timestamp"] > hour_ago)
            daily = sum(u["cost"] for u in self._usage["daily"]
                       if u["timestamp"] > day_ago)
            monthly = sum(u["cost"] for u in self._usage["monthly"]
                         if u["timestamp"] > month_ago)

            return {
                "hourly": hourly,
                "daily": daily,
                "monthly": monthly
            }

    def get_model_distribution(self) -> Dict[str, float]:
        """获取模型使用分布."""
        with self._usage_lock:
            now = time.time()
            day_ago = now - 86400
            
            # 统计24小时内的使用情况
            model_calls: Dict[str, int] = {}
            total_calls = 0
            
            for usage in self._usage["daily"]:
                if usage["timestamp"] > day_ago:
                    model = usage["model"]
                    model_calls[model] = model_calls.get(model, 0) + 1
                    total_calls += 1

            if total_calls == 0:
                return {}

            return {
                model: count / total_calls
                for model, count in model_calls.items()
            }

    def is_budget_available(self) -> bool:
        """判断当前预算是否允许继续调用LLM."""

        costs = self.get_current_costs()
        return (
            costs["hourly"] < self.limits.hourly_budget and
            costs["daily"] < self.limits.daily_budget and
            costs["monthly"] < self.limits.monthly_budget
        )

    def _calculate_cost(self, model: str, prompt_tokens: int,
                       completion_tokens: int) -> float:
        """计算使用成本."""
        model_costs = self._resolve_model_cost(model)
        if not model_costs:
            LOGGER.debug(
                "Unknown model cost configuration: %s, using zero cost",
                model,
                extra=LOG_EXTRA,
            )
            return 0.0
        return model_costs.calculate(prompt_tokens, completion_tokens)

    def _resolve_model_cost(self, model: str) -> Optional[ModelCosts]:
        """Resolve the pricing rule for a model, considering family prefixes."""

        if not model:
            return None
        key = model.lower().strip()
        if key in self._costs:
            return self._costs[key]
        for prefix, costs in self._cost_prefixes.items():
            if key.startswith(prefix):
                return costs
        return None

    def _check_budget_limits(self, model: str, prompt_tokens: int,
                           completion_tokens: int) -> bool:
        """检查是否超出预算限制."""
        estimated_cost = self._calculate_cost(model, prompt_tokens, completion_tokens)
        current_costs = self.get_current_costs()

        # 检查各个时间维度的预算限制
        if (current_costs["hourly"] + estimated_cost > self.limits.hourly_budget or
            current_costs["daily"] + estimated_cost > self.limits.daily_budget or
            current_costs["monthly"] + estimated_cost > self.limits.monthly_budget):
            LOGGER.warning(
                "Cost limit exceeded - model: %s, estimated: $%.4f",
                model, estimated_cost, extra=LOG_EXTRA
            )
            return False

        return True

    def _check_model_weights(self, model: str) -> bool:
        """检查是否符合模型权重限制."""
        if model not in self.limits.model_weights:
            return True  # 未配置权重的模型不限制

        distribution = self.get_model_distribution()
        current_weight = distribution.get(model, 0.0)
        max_weight = self.limits.model_weights[model]

        if current_weight >= max_weight:
            LOGGER.warning(
                "Model weight exceeded - model: %s, current: %.1f%%, max: %.1f%%",
                model, current_weight * 100, max_weight * 100, extra=LOG_EXTRA
            )
            return False

        return True

    def _cleanup_old_usage(self, current_time: float) -> None:
        """清理过期的使用记录."""
        if current_time - self._last_cleanup < self._cleanup_interval:
            return

        hour_ago = current_time - 3600
        day_ago = current_time - 86400
        month_ago = current_time - 2592000

        self._usage["hourly"] = [
            u for u in self._usage["hourly"]
            if u["timestamp"] > hour_ago
        ]
        self._usage["daily"] = [
            u for u in self._usage["daily"]
            if u["timestamp"] > day_ago
        ]
        self._usage["monthly"] = [
            u for u in self._usage["monthly"]
            if u["timestamp"] > month_ago
        ]

        self._last_cleanup = current_time


# 全局实例管理
_CONTROLLER_LOCK = threading.Lock()
_GLOBAL_CONTROLLER = CostController()
_LAST_LIMITS: Optional[CostLimits] = None


def get_controller() -> CostController:
    """向后兼容的全局 CostController 访问方法."""

    return _GLOBAL_CONTROLLER


def get_cost_controller() -> CostController:
    """显式返回全局 CostController 实例."""

    return _GLOBAL_CONTROLLER


def set_cost_limits(limits: CostLimits) -> None:
    """设置全局成本限制（兼容旧接口）。"""

    configure_cost_limits(limits)


def configure_cost_limits(limits: CostLimits) -> None:
    """设置全局成本限制，如果变更才更新。"""

    global _LAST_LIMITS
    with _CONTROLLER_LOCK:
        if _LAST_LIMITS is None or limits != _LAST_LIMITS:
            _GLOBAL_CONTROLLER.limits = limits
            _LAST_LIMITS = limits


def budget_available() -> bool:
    """判断全局预算是否仍可用。"""

    with _CONTROLLER_LOCK:
        return _GLOBAL_CONTROLLER.is_budget_available()
