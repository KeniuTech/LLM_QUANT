from __future__ import annotations

from datetime import date, timedelta

from app.data.schema import initialize_database
from app.utils.db import db_session


def populate_sample_data(ts_code: str, as_of: date, days: int = 60) -> None:
    """Populate ``daily`` 和 ``daily_basic`` 表用于测试。"""
    initialize_database()
    with db_session() as conn:
        for offset in range(days):
            current_day = as_of - timedelta(days=offset)
            trade_date = current_day.strftime("%Y%m%d")
            close = 100 + (days - 1 - offset)
            turnover = 5 + 0.1 * (days - 1 - offset)
            conn.execute(
                """
                INSERT OR REPLACE INTO daily
                (ts_code, trade_date, open, high, low, close, pct_chg, vol, amount)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_code,
                    trade_date,
                    close,
                    close,
                    close,
                    close,
                    0.0,
                    1_000.0,
                    1_000_000.0,
                ),
            )
            pe = 10.0 + (offset % 5)
            pb = 1.5 + (offset % 3) * 0.1
            ps = 2.0 + (offset % 4) * 0.1
            volume_ratio = 0.5 + (offset % 4) * 0.5
            conn.execute(
                """
                INSERT OR REPLACE INTO daily_basic
                (ts_code, trade_date, turnover_rate, turnover_rate_f, volume_ratio, pe, pb, ps)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts_code,
                    trade_date,
                    turnover,
                    turnover,
                    volume_ratio,
                    pe,
                    pb,
                    ps,
                ),
            )
