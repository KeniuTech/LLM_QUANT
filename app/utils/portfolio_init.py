"""Initialize portfolio database tables."""
from __future__ import annotations

from typing import Any

from .logging import get_logger
from .config import get_config


LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "portfolio_init"}


def get_portfolio_config() -> dict[str, Any]:
    """获取投资组合配置.
    
    Returns:
        包含以下字段的字典:
        - initial_capital: 初始投资金额
        - currency: 货币类型
        - position_limits: 仓位限制
    """
    config = get_config()
    settings = config.portfolio if hasattr(config, "portfolio") else None
    
    if not settings:
        from .config import PortfolioSettings
        settings = PortfolioSettings()
    
    return {
        "initial_capital": settings.initial_capital,
        "currency": settings.currency,
        "position_limits": {
            "max_position": settings.max_position,
            "min_position": settings.min_position,
            "max_total_positions": settings.max_total_positions,
            "max_sector_exposure": settings.max_sector_exposure
        }
    }

def update_portfolio_config(updates: dict[str, Any]) -> None:
    """更新投资组合配置.
    
    Args:
        updates: 要更新的配置项字典
    """
    from .config import get_config, save_config, PortfolioSettings
    
    # 获取当前配置
    config = get_config()
    
    # 创建新的投资组合设置
    portfolio = PortfolioSettings(
        initial_capital=updates["initial_capital"],
        currency=updates["currency"],
        max_position=updates["position_limits"]["max_position"],
        min_position=updates["position_limits"]["min_position"],
        max_total_positions=updates["position_limits"]["max_total_positions"],
        max_sector_exposure=updates["position_limits"]["max_sector_exposure"]
    )
    
    # 更新配置
    config.portfolio = portfolio
    save_config(config)



SCHEMA_STATEMENTS = [
    # 投资池表
    """
    CREATE TABLE IF NOT EXISTS investment_pool (
        trade_date TEXT,
        ts_code TEXT,
        score REAL,
        status TEXT,
        rationale TEXT,
        tags TEXT,  -- JSON array
        metadata TEXT,  -- JSON object
        PRIMARY KEY (trade_date, ts_code)
    );
    """,
    
    # 数据获取任务表
    """
    CREATE TABLE IF NOT EXISTS fetch_jobs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_type TEXT NOT NULL,
        status TEXT NOT NULL,
        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
        error_msg TEXT,
        metadata TEXT  -- JSON object for additional info
    );
    """,
    
    # 持仓表
    """
    CREATE TABLE IF NOT EXISTS portfolio_positions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts_code TEXT NOT NULL,
        opened_date TEXT NOT NULL,
        closed_date TEXT,
        quantity REAL NOT NULL,
        cost_price REAL NOT NULL,
        market_price REAL,
        market_value REAL,
        realized_pnl REAL DEFAULT 0,
        unrealized_pnl REAL DEFAULT 0,
        target_weight REAL,
        status TEXT NOT NULL DEFAULT 'open',
        notes TEXT,
        metadata TEXT  -- JSON object
    );
    """,
    
    # 投资组合快照表
    """
    CREATE TABLE IF NOT EXISTS portfolio_snapshots (
        trade_date TEXT PRIMARY KEY,
        total_value REAL,
        cash REAL,
        invested_value REAL,
        unrealized_pnl REAL,
        realized_pnl REAL,
        net_flow REAL,
        exposure REAL,
        notes TEXT,
        metadata TEXT  -- JSON object
    );
    """,
]


def initialize_database_schema() -> None:
    """Create database tables if they don't exist."""
    from .db import db_session
    
    with db_session() as conn:
        for statement in SCHEMA_STATEMENTS:
            try:
                conn.execute(statement)
            except Exception:  # noqa: BLE001
                LOGGER.exception(
                    "执行 schema 语句失败",
                    extra={"sql": statement, **LOG_EXTRA}
                )
                raise
