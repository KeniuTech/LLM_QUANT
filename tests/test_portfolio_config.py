"""Test portfolio configuration and initialization."""
from unittest.mock import patch, MagicMock

from app.utils.portfolio import get_latest_snapshot
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
