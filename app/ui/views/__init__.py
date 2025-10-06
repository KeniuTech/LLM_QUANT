"""View modules for Streamlit UI tabs."""

from .today import render_today_plan
from .pool import render_pool_overview
from .backtest import render_backtest_review
from .market import render_market_visualization
from .logs import render_log_viewer
from .settings import render_config_overview, render_llm_settings, render_data_settings
from .tests import render_tests
from .dashboard import render_global_dashboard, update_dashboard_sidebar

__all__ = [
    "render_today_plan",
    "render_pool_overview",
    "render_backtest_review",
    "render_market_visualization",
    "render_log_viewer",
    "render_config_overview",
    "render_llm_settings",
    "render_data_settings",
    "render_tests",
    "render_global_dashboard",
    "update_dashboard_sidebar",
]
