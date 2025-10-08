"""SQL schema for index membership."""
from app.utils.db import db_session

def initialize_index_membership_tables(conn):
    """Create tables for tracking index membership."""
    conn.execute("""
    CREATE TABLE IF NOT EXISTS index_weight (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        index_code VARCHAR(10) NOT NULL,
        trade_date VARCHAR(8) NOT NULL,
        ts_code VARCHAR(10) NOT NULL,
        weight FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_index_weight_lookup (index_code, trade_date)
    )
    """)

def add_default_indices():
    """Add default index list."""
    indices = [
        ("000300.SH", "沪深300"),
        ("000905.SH", "中证500"),
        ("000852.SH", "中证1000")
    ]
    with db_session() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS indices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            index_code VARCHAR(10) NOT NULL UNIQUE,
            name VARCHAR(50) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        for code, name in indices:
            conn.execute(
                """
                INSERT OR IGNORE INTO indices (index_code, name)
                VALUES (?, ?)
                """,
                (code, name)
            )