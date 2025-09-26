"""RSS ingestion for news and heat scores."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List


@dataclass
class RssItem:
    id: str
    title: str
    link: str
    published: datetime
    summary: str
    source: str


def fetch_rss_feed(url: str) -> List[RssItem]:
    """Download and parse an RSS feed into structured items."""

    raise NotImplementedError


def deduplicate_items(items: Iterable[RssItem]) -> List[RssItem]:
    """Drop duplicate stories by link/id fingerprint."""

    seen = set()
    unique: List[RssItem] = []
    for item in items:
        key = item.id or item.link
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def save_news_items(items: Iterable[RssItem]) -> None:
    """Persist RSS items into the `news` table."""

    raise NotImplementedError
