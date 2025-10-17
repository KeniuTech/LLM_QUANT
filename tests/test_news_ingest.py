from __future__ import annotations

import json
from datetime import datetime

from app.ingest import entity_recognition
from app.ingest.rss import RssItem, save_news_items
from app.utils.db import db_session


def test_save_news_items_persists_entities_and_heat(isolated_db):
    # Reset mapping state
    entity_recognition._COMPANY_MAPPING_INITIALIZED = False
    mapper = entity_recognition.company_mapper
    mapper.name_to_code.clear()
    mapper.short_names.clear()
    mapper.aliases.clear()

    ts_code = "000001.SZ"
    mapper.add_company(ts_code, "平安银行股份有限公司", "平安银行")

    item = RssItem(
        id="news-1",
        title="平安银行利好消息爆发",
        link="https://example.com/news",
        published=datetime.utcnow(),
        summary="平安银行股份有限公司公布季度业绩，银行板块再迎利好。",
        source="TestWire",
    )

    item.extract_entities()
    assert ts_code in item.ts_codes

    saved = save_news_items([item])
    assert saved == 1

    with db_session(read_only=True) as conn:
        row = conn.execute(
            "SELECT heat, entities FROM news WHERE ts_code = ? ORDER BY pub_time DESC LIMIT 1",
            (ts_code,),
        ).fetchone()

    assert row is not None
    assert 0.0 <= row["heat"] <= 1.0
    assert row["heat"] > 0.6

    entities_payload = json.loads(row["entities"])
    assert ts_code in entities_payload.get("ts_codes", [])
    assert "industries" in entities_payload
    assert "important_keywords" in entities_payload
