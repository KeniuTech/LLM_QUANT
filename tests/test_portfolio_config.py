"""Test portfolio configuration and initialization."""
import json
from dataclasses import replace
from unittest.mock import patch, MagicMock

import pytest

from app.utils import config as config_module
from app.utils.config import AppConfig, DataPaths, get_config
from app.utils.portfolio_init import update_portfolio_config

from app.utils.portfolio import get_latest_snapshot, list_investment_pool
from app.utils.db import db_session


def test_default_portfolio_config():
    """Test default portfolio configuration."""
    # Mock db_session as a context manager
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=None)
    
    # Mock the database query result
    mock_session.execute.return_value.fetchone.return_value = None
    
    # 使用默认配置
    with patch("app.utils.portfolio.get_portfolio_config") as mock_config, \
         patch("app.utils.portfolio.db_session", return_value=mock_session):
        mock_config.return_value = {
            "initial_capital": 1000000,
            "currency": "CNY"
        }
        
        snapshot = get_latest_snapshot()
        assert snapshot is not None
        assert snapshot.total_value == 1000000
        assert snapshot.cash == 1000000
        assert snapshot.metadata["initial_capital"] == 1000000
        assert snapshot.metadata["currency"] == "CNY"


def test_custom_portfolio_config():
    """Test custom portfolio configuration."""
    # Mock db_session as a context manager
    mock_session = MagicMock()
    mock_session.__enter__ = MagicMock(return_value=mock_session)
    mock_session.__exit__ = MagicMock(return_value=None)
    
    # Mock the database query result
    mock_session.execute.return_value.fetchone.return_value = None
    
    # 使用自定义配置
    with patch("app.utils.portfolio.get_portfolio_config") as mock_config, \
         patch("app.utils.portfolio.db_session", return_value=mock_session):
        mock_config.return_value = {
            "initial_capital": 2000000,
            "currency": "USD"
        }
        
        snapshot = get_latest_snapshot()
        assert snapshot is not None
        assert snapshot.total_value == 2000000
        assert snapshot.cash == 2000000
        assert snapshot.metadata["initial_capital"] == 2000000
        assert snapshot.metadata["currency"] == "USD"


def test_update_portfolio_config_persists(tmp_path):
    cfg = get_config()
    original_paths = cfg.data_paths
    original_portfolio = replace(cfg.portfolio)

    temp_root = tmp_path / "data"
    temp_paths = DataPaths(root=temp_root)
    cfg.data_paths = temp_paths

    updates = {
        "initial_capital": 3_000_000,
        "currency": "USD",
        "position_limits": {
            "max_position": 0.15,
            "min_position": 0.03,
            "max_total_positions": 12,
            "max_sector_exposure": 0.4,
        },
    }

    try:
        update_portfolio_config(updates)

        payload = json.loads(temp_paths.config_file.read_text(encoding="utf-8"))
        assert payload["portfolio"]["initial_capital"] == 3_000_000
        assert payload["portfolio"]["currency"] == "USD"
        limits = payload["portfolio"]["position_limits"]
        assert limits["max_position"] == pytest.approx(0.15)
        assert limits["min_position"] == pytest.approx(0.03)
        assert limits["max_total_positions"] == 12
        assert limits["max_sector_exposure"] == pytest.approx(0.4)

        fresh_cfg = AppConfig()
        fresh_cfg.data_paths = temp_paths
        config_module._load_from_file(fresh_cfg)
        assert fresh_cfg.portfolio.initial_capital == pytest.approx(3_000_000.0)
        assert fresh_cfg.portfolio.currency == "USD"
        assert fresh_cfg.portfolio.max_position == pytest.approx(0.15)
        assert fresh_cfg.portfolio.min_position == pytest.approx(0.03)
        assert fresh_cfg.portfolio.max_total_positions == 12
        assert fresh_cfg.portfolio.max_sector_exposure == pytest.approx(0.4)
    finally:
        cfg.data_paths = original_paths
        cfg.portfolio = original_portfolio
        if temp_paths.config_file.exists():
            temp_paths.config_file.unlink()


def test_list_investment_pool_orders_without_nulls(tmp_path):
    cfg = get_config()
    original_paths = cfg.data_paths

    temp_root = tmp_path / "data"
    temp_paths = DataPaths(root=temp_root)
    cfg.data_paths = temp_paths

    try:
        with db_session() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS investment_pool (
                    trade_date TEXT,
                    ts_code TEXT,
                    score REAL,
                    status TEXT,
                    rationale TEXT,
                    tags TEXT,
                    metadata TEXT,
                    name TEXT,
                    industry TEXT,
                    PRIMARY KEY (trade_date, ts_code)
                )
                """
            )
            conn.executemany(
                """
                INSERT INTO investment_pool (trade_date, ts_code, score, status, rationale, tags, metadata, name, industry)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    ("2024-01-01", "AAA", 0.8, "buy", "", None, None, "Company A", "Technology"),
                    ("2024-01-01", "BBB", None, "hold", "", None, None, "Company B", "Finance"),
                    ("2024-01-01", "CCC", 0.9, "buy", "", None, None, "Company C", "Healthcare"),
                ],
            )

        rows = list_investment_pool(trade_date="2024-01-01")
        assert [row.ts_code for row in rows] == ["CCC", "AAA", "BBB"]
    finally:
        cfg.data_paths = original_paths
