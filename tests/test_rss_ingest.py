"""Tests for RSS ingestion utilities."""
from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.ingest import rss
from app.utils import alerts
from app.utils.config import DataPaths, get_config
from app.utils.db import db_session


@pytest.fixture()
def isolated_db(tmp_path):
    """Temporarily redirect database paths for isolated writes."""

    cfg = get_config()
    original_paths = cfg.data_paths
    tmp_root = tmp_path / "data"
    tmp_root.mkdir(parents=True, exist_ok=True)
    cfg.data_paths = DataPaths(root=tmp_root)
    alerts.clear_warnings()
    try:
        yield
    finally:
        cfg.data_paths = original_paths


def test_fetch_rss_feed_parses_entries(monkeypatch):
    published = datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
    sample_feed = (
        f"""
        <rss version=\"2.0\">
          <channel>
            <title>Example</title>
            <item>
              <title>新闻：公司利好公告</title>
              <link>https://example.com/a</link>
              <description><![CDATA[内容包含 000001.SZ ]]></description>
              <pubDate>{published}</pubDate>
              <guid>a</guid>
            </item>
          </channel>
        </rss>
        """
    ).encode("utf-8")

    monkeypatch.setattr(
        rss,
        "_download_feed",
        lambda url, timeout, max_retries, retry_backoff, retry_jitter: sample_feed,
    )

    items = rss.fetch_rss_feed("https://example.com/rss", hours_back=24)

    assert len(items) == 1
    item = items[0]
    assert item.title.startswith("新闻")
    assert item.source == "example.com"


def test_save_news_items_writes_and_deduplicates(isolated_db):
    published = datetime.utcnow() - timedelta(hours=1)
    rss_item = rss.RssItem(
        id="test-id",
        title="利好消息推动股价",
        link="https://example.com/news/test",
        published=published,
        summary="这是一条利好消息。",
        source="测试来源",
        ts_codes=("000001.SZ",),
    )

    inserted = rss.save_news_items([rss_item])
    assert inserted >= 1

    with db_session(read_only=True) as conn:
        row = conn.execute(
            "SELECT ts_code, sentiment, heat FROM news WHERE id = ?",
            ("test-id::000001.SZ",),
        ).fetchone()
        assert row is not None
        assert row["ts_code"] == "000001.SZ"
        assert row["sentiment"] >= 0  # 利好关键词应给出非负情绪
        assert 0 <= row["heat"] <= 1

    # 再次保存同一条新闻应被忽略
    duplicate = rss.save_news_items([rss_item])
    assert duplicate == 0
