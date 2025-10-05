"""Test cases for LLM cost control system."""
import time

from app.llm.cost import (
    CostLimits,
    ModelCosts,
    CostController,
    configure_cost_limits,
    get_cost_controller,
    budget_available,
)


def test_cost_limits():
    """Test cost limits configuration."""
    limits = CostLimits(
        hourly_budget=10.0,
        daily_budget=100.0,
        monthly_budget=1000.0,
        model_weights={"gpt-4": 0.7, "gpt-3.5-turbo": 0.3}
    )
    
    assert limits.hourly_budget == 10.0
    assert limits.daily_budget == 100.0
    assert limits.monthly_budget == 1000.0
    assert limits.model_weights["gpt-4"] == 0.7


def test_model_costs():
    """Test model cost tracking."""
    costs = ModelCosts(
        prompt_cost_per_1k=0.1,   # $0.1 per 1K tokens
        completion_cost_per_1k=0.2 # $0.2 per 1K tokens
    )
    
    # Test cost calculation
    prompt_tokens = 1000  # 1K tokens
    completion_tokens = 500  # 0.5K tokens
    
    total_cost = costs.calculate(prompt_tokens, completion_tokens)
    expected_cost = (prompt_tokens / 1000 * costs.prompt_cost_per_1k + 
                    completion_tokens / 1000 * costs.completion_cost_per_1k)
    
    assert total_cost == expected_cost  # Should be $0.2


def test_cost_controller():
    """Test cost controller functionality."""
    limits = CostLimits(
        hourly_budget=1.0,
        daily_budget=10.0,
        monthly_budget=100.0,
        model_weights={"gpt-4": 0.7, "gpt-3.5-turbo": 0.3}
    )
    
    controller = CostController(limits=limits)
    
    # First check if we can use model
    assert controller.can_use_model("gpt-4", 1000, 500)
    
    # Then record the usage
    assert controller.record_usage("gpt-4", 1000, 500) is True  # About $0.09
    
    # Record usage for second model to maintain weight balance
    assert controller.can_use_model("gpt-3.5-turbo", 1000, 500)
    assert controller.record_usage("gpt-3.5-turbo", 1000, 500)
    
    # Verify usage tracking
    costs = controller.get_current_costs()
    assert costs["hourly"] > 0
    assert costs["daily"] > 0
    assert costs["monthly"] > 0
    
    # Test model distribution
    distribution = controller.get_model_distribution()
    assert "gpt-4" in distribution and "gpt-3.5-turbo" in distribution
    assert abs(distribution["gpt-4"] - 0.5) < 0.1  # Allow some deviation
    assert abs(distribution["gpt-3.5-turbo"] - 0.5) < 0.1  # Should be roughly balanced


def test_cost_controller_history():
    """Test cost controller usage history."""
    limits = CostLimits(
        hourly_budget=1.0,
        daily_budget=10.0,
        monthly_budget=100.0,
        model_weights={"gpt-4": 0.5, "gpt-3.5-turbo": 0.5}  # Equal weights
    )
    
    controller = CostController(limits=limits)
    
    # Record one usage of each model
    assert controller.can_use_model("gpt-4", 1000, 500)
    assert controller.record_usage("gpt-4", 1000, 500)

    assert controller.can_use_model("gpt-3.5-turbo", 1000, 500)
    assert controller.record_usage("gpt-3.5-turbo", 1000, 500)
    
    # Check usage tracking
    costs = controller.get_current_costs()
    assert costs["hourly"] > 0  # Should have accumulated cost
    
    # Verify the usage distribution is roughly balanced
    distribution = controller.get_model_distribution()
    assert abs(distribution["gpt-4"] - 0.5) < 0.1
    assert abs(distribution["gpt-3.5-turbo"] - 0.5) < 0.1


def test_global_controller_budget_toggle():
    """Ensure global controller respects configured limits and budget flag."""
    limits = CostLimits(hourly_budget=0.05, daily_budget=0.1, monthly_budget=1.0, model_weights={})
    configure_cost_limits(limits)
    controller = get_cost_controller()

    assert budget_available() is True
    within = controller.record_usage("gpt-4", 1000, 1000)
    assert within is False  # deliberately exceed tiny budget
    assert budget_available() is False

    # Reset controller state to avoid cross-test contamination
    with controller._usage_lock:  # type: ignore[attr-defined]
        for bucket in controller._usage.values():  # type: ignore[attr-defined]
            bucket.clear()
        controller._last_cleanup = time.time()  # type: ignore[attr-defined]

    configure_cost_limits(CostLimits.default())
