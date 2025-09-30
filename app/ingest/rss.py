"""RSS ingestion utilities for news sentiment and heat scoring."""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlparse, urljoin
from xml.etree import ElementTree as ET

import requests
from requests import RequestException

import hashlib
import random
import time

try:  # pragma: no cover - optional dependency at runtime
    import feedparser  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - graceful fallback
    feedparser = None  # type: ignore[assignment]

from app.data.schema import initialize_database
from app.utils import alerts
from app.utils.config import get_config
from app.utils.db import db_session
from app.utils.logging import get_logger


LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "rss_ingest"}

DEFAULT_TIMEOUT = 10.0
MAX_SUMMARY_LENGTH = 1500

POSITIVE_KEYWORDS: Tuple[str, ...] = (
    "利好",
    "增长",
    "超预期",
    "创新高",
    "增持",
    "回购",
    "盈利",
    "strong",
    "beat",
    "upgrade",
)
NEGATIVE_KEYWORDS: Tuple[str, ...] = (
    "利空",
    "下跌",
    "亏损",
    "裁员",
    "违约",
    "处罚",
    "暴跌",
    "减持",
    "downgrade",
    "miss",
)

A_SH_CODE_PATTERN = re.compile(r"\b(\d{6})(?:\.(SH|SZ))?\b", re.IGNORECASE)
HK_CODE_PATTERN = re.compile(r"\b(\d{4})\.HK\b", re.IGNORECASE)


@dataclass
class RssFeedConfig:
    """Configuration describing a single RSS source."""

    url: str
    source: str
    ts_codes: Tuple[str, ...] = ()
    keywords: Tuple[str, ...] = ()
    hours_back: int = 48
    max_items: int = 50


@dataclass
class RssItem:
    """Structured representation of an RSS entry."""

    id: str
    title: str
    link: str
    published: datetime
    summary: str
    source: str
    ts_codes: Tuple[str, ...] = ()


DEFAULT_RSS_SOURCES: Tuple[RssFeedConfig, ...] = ()


def fetch_rss_feed(
    url: str,
    *,
    source: Optional[str] = None,
    hours_back: int = 48,
    max_items: int = 50,
    timeout: float = DEFAULT_TIMEOUT,
    max_retries: int = 5,
    retry_backoff: float = 1.5,
    retry_jitter: float = 0.3,
) -> List[RssItem]:
    """Download and parse an RSS feed into structured items."""

    return _fetch_feed_items(
        url,
        source=source,
        hours_back=hours_back,
        max_items=max_items,
        timeout=timeout,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        retry_jitter=retry_jitter,
        allow_html_redirect=True,
    )


def _fetch_feed_items(
    url: str,
    *,
    source: Optional[str],
    hours_back: int,
    max_items: int,
    timeout: float,
    max_retries: int,
    retry_backoff: float,
    retry_jitter: float,
    allow_html_redirect: bool,
) -> List[RssItem]:

    content = _download_feed(
        url,
        timeout,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        retry_jitter=retry_jitter,
    )
    if content is None:
        return []

    if allow_html_redirect:
        feed_links = _extract_html_feed_links(content, url)
        if feed_links:
            LOGGER.info(
                "RSS 页面包含子订阅 %s 个，自动展开",
                len(feed_links),
                extra=LOG_EXTRA,
            )
            aggregated: List[RssItem] = []
            for feed_url in feed_links:
                sub_items = _fetch_feed_items(
                    feed_url,
                    source=source,
                    hours_back=hours_back,
                    max_items=max_items,
                    timeout=timeout,
                    max_retries=max_retries,
                    retry_backoff=retry_backoff,
                    retry_jitter=retry_jitter,
                    allow_html_redirect=False,
                )
                aggregated.extend(sub_items)
                if max_items > 0 and len(aggregated) >= max_items:
                    return aggregated[:max_items]
            if aggregated:
                alerts.clear_warnings(_rss_source_key(url))
            else:
                alerts.add_warning(
                    _rss_source_key(url),
                    "聚合页未返回内容",
                )
            return aggregated

    parsed_entries = _parse_feed_content(content)
    total_entries = len(parsed_entries)
    LOGGER.info(
        "RSS 源获取完成 url=%s raw_entries=%s",
        url,
        total_entries,
        extra=LOG_EXTRA,
    )
    if not parsed_entries:
        LOGGER.warning(
            "RSS 无可解析条目 url=%s snippet=%s",
            url,
            _safe_snippet(content),
            extra=LOG_EXTRA,
        )
        return []

    cutoff = datetime.utcnow() - timedelta(hours=max(1, hours_back))
    source_name = source or _source_from_url(url)
    items: List[RssItem] = []
    seen_ids: set[str] = set()
    for entry in parsed_entries:
        published = entry.get("published") or datetime.utcnow()
        if published < cutoff:
            continue
        title = _clean_text(entry.get("title", ""))
        summary = _clean_text(entry.get("summary", ""))
        link = entry.get("link", "")
        raw_id = entry.get("id") or link
        item_id = _normalise_item_id(raw_id, link, title, published)
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)
        items.append(
            RssItem(
                id=item_id,
                title=title,
                link=link,
                published=published,
                summary=_truncate(summary, MAX_SUMMARY_LENGTH),
                source=source_name,
            )
        )
        if len(items) >= max_items > 0:
            break

    LOGGER.info(
        "RSS 过滤结果 url=%s within_window=%s unique=%s",
        url,
        sum(1 for entry in parsed_entries if (entry.get("published") or datetime.utcnow()) >= cutoff),
        len(items),
        extra=LOG_EXTRA,
    )
    if items:
        alerts.clear_warnings(_rss_source_key(url))

    return items


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


def save_news_items(items: Iterable[RssItem]) -> int:
    """Persist RSS items into the `news` table."""

    initialize_database()
    now = datetime.utcnow()
    rows: List[Tuple[object, ...]] = []

    processed = 0
    for item in items:
        text_payload = f"{item.title}\n{item.summary}"
        sentiment = _estimate_sentiment(text_payload)
        base_codes = tuple(code for code in item.ts_codes if code)
        heat = _estimate_heat(item.published, now, len(base_codes), sentiment)
        entities = json.dumps(
            {
                "ts_codes": list(base_codes),
                "source_url": item.link,
            },
            ensure_ascii=False,
        )
        resolved_codes = base_codes or (None,)
        for ts_code in resolved_codes:
            row_id = item.id if ts_code is None else f"{item.id}::{ts_code}"
            rows.append(
                (
                    row_id,
                    ts_code,
                    item.published.replace(tzinfo=timezone.utc).isoformat(),
                    item.source,
                    item.title,
                    item.summary,
                    item.link,
                    entities,
                    sentiment,
                    heat,
                )
            )
        processed += 1

    if not rows:
        return 0

    inserted = 0
    try:
        with db_session() as conn:
            conn.executemany(
                """
                INSERT OR IGNORE INTO news
                (id, ts_code, pub_time, source, title, summary, url, entities, sentiment, heat)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            inserted = conn.total_changes
    except sqlite3.OperationalError:
        LOGGER.exception("写入新闻数据失败，表结构可能未初始化", extra=LOG_EXTRA)
        return 0
    except Exception:  # pragma: no cover - guard unexpected sqlite errors
        LOGGER.exception("写入新闻数据异常", extra=LOG_EXTRA)
        return 0

    LOGGER.info(
        "RSS 新闻落库完成 processed=%s inserted=%s",
        processed,
        inserted,
        extra=LOG_EXTRA,
    )
    return inserted


def ingest_configured_rss(
    *,
    hours_back: Optional[int] = None,
    max_items_per_feed: Optional[int] = None,
    max_retries: int = 5,
    retry_backoff: float = 2.0,
    retry_jitter: float = 0.5,
) -> int:
    """Ingest all configured RSS feeds into the news store."""

    configs = resolve_rss_sources()
    if not configs:
        LOGGER.info("未配置 RSS 来源，跳过新闻拉取", extra=LOG_EXTRA)
        return 0

    aggregated: List[RssItem] = []
    fetched_count = 0
    for index, cfg in enumerate(configs, start=1):
        window = hours_back or cfg.hours_back
        limit = max_items_per_feed or cfg.max_items
        LOGGER.info(
            "开始拉取 RSS：%s (window=%sh, limit=%s)",
            cfg.url,
            window,
            limit,
            extra=LOG_EXTRA,
        )
        items = fetch_rss_feed(
            cfg.url,
            source=cfg.source,
            hours_back=window,
            max_items=limit,
            max_retries=max_retries,
            retry_backoff=retry_backoff,
            retry_jitter=retry_jitter,
        )
        if not items:
            LOGGER.info("RSS 来源无新内容：%s", cfg.url, extra=LOG_EXTRA)
            continue
        enriched: List[RssItem] = []
        for item in items:
            codes = _assign_ts_codes(item, cfg.ts_codes, cfg.keywords)
            enriched.append(replace(item, ts_codes=tuple(codes)))
        aggregated.extend(enriched)
        fetched_count += len(enriched)
        if fetched_count and index < len(configs):
            time.sleep(2.0)

    if not aggregated:
        LOGGER.info("RSS 来源未产生有效新闻", extra=LOG_EXTRA)
        alerts.add_warning("RSS", "未获取到任何 RSS 新闻")
        return 0

    deduped = deduplicate_items(aggregated)
    LOGGER.info(
        "RSS 聚合完成 total_fetched=%s unique=%s",
        fetched_count,
        len(deduped),
        extra=LOG_EXTRA,
    )
    return save_news_items(deduped)


def resolve_rss_sources() -> List[RssFeedConfig]:
    """Resolve RSS feed configuration from persisted settings."""

    cfg = get_config()
    raw = getattr(cfg, "rss_sources", None) or {}
    feeds: Dict[str, RssFeedConfig] = {}

    def _add_feed(url: str, **kwargs: object) -> None:
        clean_url = url.strip()
        if not clean_url:
            return
        key = clean_url.lower()
        if key in feeds:
            return
        source_name = kwargs.get("source") or _source_from_url(clean_url)
        feeds[key] = RssFeedConfig(
            url=clean_url,
            source=str(source_name),
            ts_codes=tuple(kwargs.get("ts_codes", ()) or ()),
            keywords=tuple(kwargs.get("keywords", ()) or ()),
            hours_back=int(kwargs.get("hours_back", 48) or 48),
            max_items=int(kwargs.get("max_items", 50) or 50),
        )

    if isinstance(raw, dict):
        for key, value in raw.items():
            if isinstance(value, dict):
                if not value.get("enabled", True):
                    continue
                url = str(value.get("url") or key)
                ts_codes = [
                    str(code).strip().upper()
                    for code in value.get("ts_codes", [])
                    if str(code).strip()
                ]
                keywords = [
                    str(token).strip()
                    for token in value.get("keywords", [])
                    if str(token).strip()
                ]
                _add_feed(
                    url,
                    ts_codes=ts_codes,
                    keywords=keywords,
                    hours_back=value.get("hours_back", 48),
                    max_items=value.get("max_items", 50),
                    source=value.get("source") or value.get("label"),
                )
                continue

            if not value:
                continue
            url = key
            ts_codes: List[str] = []
            if "|" in key:
                prefix, url = key.split("|", 1)
                ts_codes = [
                    token.strip().upper()
                    for token in prefix.replace(",", ":").split(":")
                    if token.strip()
                ]
            _add_feed(url, ts_codes=ts_codes)

    if feeds:
        return list(feeds.values())

    return list(DEFAULT_RSS_SOURCES)


def _download_feed(
    url: str,
    timeout: float,
    *,
    max_retries: int,
    retry_backoff: float,
    retry_jitter: float,
) -> Optional[bytes]:
    headers = {
        "User-Agent": "llm-quant/0.1 (+https://github.com/qiang/llm_quant)",
        "Accept": "application/rss+xml, application/atom+xml, application/xml;q=0.9, */*;q=0.8",
    }
    attempt = 0
    delay = max(0.5, retry_backoff)
    while attempt <= max_retries:
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
        except RequestException as exc:
            attempt += 1
            if attempt > max_retries:
                message = f"源请求失败：{url}"
                LOGGER.warning("RSS 请求失败：%s err=%s", url, exc, extra=LOG_EXTRA)
                alerts.add_warning(_rss_source_key(url), message, str(exc))
                return None
            wait = delay + random.uniform(0, retry_jitter)
            LOGGER.info(
                "RSS 请求异常，%.2f 秒后重试 url=%s attempt=%s/%s",
                wait,
                url,
                attempt,
                max_retries,
                extra=LOG_EXTRA,
            )
            time.sleep(max(wait, 0.1))
            delay *= max(1.1, retry_backoff)
            continue

        status = response.status_code
        if 200 <= status < 300:
            return response.content

        if status in {429, 503}:
            attempt += 1
            if attempt > max_retries:
                LOGGER.warning(
                    "RSS 请求失败：%s status=%s 已达到最大重试次数",
                    url,
                    status,
                    extra=LOG_EXTRA,
                )
                alerts.add_warning(
                    _rss_source_key(url),
                    "源限流",
                    f"HTTP {status}",
                )
                return None
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    wait = float(retry_after)
                except ValueError:
                    wait = delay
            else:
                wait = delay
            wait += random.uniform(0, retry_jitter)
            LOGGER.info(
                "RSS 命中限流 status=%s，%.2f 秒后重试 url=%s attempt=%s/%s",
                status,
                wait,
                url,
                attempt,
                max_retries,
                extra=LOG_EXTRA,
            )
            time.sleep(max(wait, 0.1))
            delay *= max(1.1, retry_backoff)
            continue

        LOGGER.warning(
            "RSS 请求失败：%s status=%s",
            url,
            status,
            extra=LOG_EXTRA,
        )
        alerts.add_warning(
            _rss_source_key(url),
            "源响应异常",
            f"HTTP {status}",
        )
        return None

    LOGGER.warning("RSS 请求失败：%s 未获取内容", url, extra=LOG_EXTRA)
    alerts.add_warning(_rss_source_key(url), "未获取内容")
    return None


def _extract_html_feed_links(content: bytes, base_url: str) -> List[str]:
    sample = content[:1024].lower()
    if b"<rss" in sample or b"<feed" in sample:
        return []

    for encoding in ("utf-8", "gb18030", "gb2312"):
        try:
            text = content.decode(encoding)
            break
        except UnicodeDecodeError:
            text = content.decode(encoding, errors="ignore")
            break
    else:
        text = content.decode("utf-8", errors="ignore")

    if "<link" not in text and ".xml" not in text:
        return []

    feed_urls: List[str] = []
    alternates = re.compile(
        r"<link[^>]+rel=[\"']alternate[\"'][^>]+type=[\"']application/(?:rss|atom)\+xml[\"'][^>]*href=[\"']([^\"']+)[\"']",
        re.IGNORECASE,
    )
    for match in alternates.finditer(text):
        href = match.group(1).strip()
        if href:
            feed_urls.append(urljoin(base_url, href))

    if not feed_urls:
        anchors = re.compile(r"href=[\"']([^\"']+\.xml)[\"']", re.IGNORECASE)
        for match in anchors.finditer(text):
            href = match.group(1).strip()
            if href:
                feed_urls.append(urljoin(base_url, href))

    unique_urls: List[str] = []
    seen = set()
    for href in feed_urls:
        if href not in seen and href != base_url:
            seen.add(href)
            unique_urls.append(href)
    return unique_urls


def _safe_snippet(content: bytes, limit: int = 160) -> str:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        try:
            text = content.decode("gb18030", errors="ignore")
        except UnicodeDecodeError:
            text = content.decode("latin-1", errors="ignore")
    cleaned = re.sub(r"\s+", " ", text)
    if len(cleaned) > limit:
        return cleaned[: limit - 3] + "..."
    return cleaned


def _parse_feed_content(content: bytes) -> List[Dict[str, object]]:
    if feedparser is not None:
        parsed = feedparser.parse(content)
        entries = []
        for entry in getattr(parsed, "entries", []) or []:
            entries.append(
                {
                    "id": getattr(entry, "id", None) or getattr(entry, "guid", None),
                    "title": getattr(entry, "title", ""),
                    "link": getattr(entry, "link", ""),
                    "summary": getattr(entry, "summary", "") or getattr(entry, "description", ""),
                    "published": _parse_datetime(
                        getattr(entry, "published", None)
                        or getattr(entry, "updated", None)
                        or getattr(entry, "issued", None)
                    ),
                }
        )
        if entries:
            return entries
    else:  # pragma: no cover - log helpful info when dependency missing
        LOGGER.warning(
            "feedparser 未安装，使用简易 XML 解析器回退处理 RSS",
            extra=LOG_EXTRA,
        )

    return _parse_feed_xml(content)


def _parse_feed_xml(content: bytes) -> List[Dict[str, object]]:
    try:
        xml_text = content.decode("utf-8")
    except UnicodeDecodeError:
        xml_text = content.decode("utf-8", errors="ignore")

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:  # pragma: no cover - depends on remote feed
        LOGGER.warning("RSS XML 解析失败 err=%s", exc, extra=LOG_EXTRA)
        return _lenient_parse_items(xml_text)

    tag = _local_name(root.tag)
    if tag == "rss":
        candidates = root.findall(".//item")
    elif tag == "feed":
        candidates = root.findall(".//{*}entry")
    else:  # fallback
        candidates = root.findall(".//item") or root.findall(".//{*}entry")

    entries: List[Dict[str, object]] = []
    for node in candidates:
        entries.append(
            {
                "id": _child_text(node, {"id", "guid"}),
                "title": _child_text(node, {"title"}) or "",
                "link": _child_text(node, {"link"}) or "",
                "summary": _child_text(node, {"summary", "description"}) or "",
                "published": _parse_datetime(
                    _child_text(node, {"pubDate", "published", "updated"})
                ),
            }
        )
    if not entries and "<item" in xml_text.lower():
        return _lenient_parse_items(xml_text)
    return entries


def _lenient_parse_items(xml_text: str) -> List[Dict[str, object]]:
    """Fallback parser that tolerates malformed RSS by using regular expressions."""

    items: List[Dict[str, object]] = []
    pattern = re.compile(r"<(item|entry)[^>]*>(.+?)</\\1>", re.IGNORECASE | re.DOTALL)
    for match in pattern.finditer(xml_text):
        block = match.group(0)
        title = _extract_tag_text(block, ["title"]) or ""
        link = _extract_link(block)
        summary = _extract_tag_text(block, ["summary", "description"]) or ""
        published_text = _extract_tag_text(block, ["pubDate", "published", "updated"])
        items.append(
            {
                "id": _extract_tag_text(block, ["id", "guid"]) or link,
                "title": title,
                "link": link,
                "summary": summary,
                "published": _parse_datetime(published_text),
            }
        )
    if items:
        LOGGER.info("RSS 采用宽松解析提取 %s 条记录", len(items), extra=LOG_EXTRA)
    return items


def _extract_tag_text(block: str, names: Sequence[str]) -> Optional[str]:
    for name in names:
        pattern = re.compile(rf"<{name}[^>]*>(.*?)</{name}>", re.IGNORECASE | re.DOTALL)
        match = pattern.search(block)
        if match:
            text = re.sub(r"<[^>]+>", " ", match.group(1))
            return _clean_text(text)
    return None


def _extract_link(block: str) -> str:
    href_pattern = re.compile(r"<link[^>]*href=\"([^\"]+)\"[^>]*>", re.IGNORECASE)
    match = href_pattern.search(block)
    if match:
        return match.group(1).strip()
    inline_pattern = re.compile(r"<link[^>]*>(.*?)</link>", re.IGNORECASE | re.DOTALL)
    match = inline_pattern.search(block)
    if match:
        return match.group(1).strip()
    return ""


def _assign_ts_codes(
    item: RssItem,
    base_codes: Sequence[str],
    keywords: Sequence[str],
) -> List[str]:
    matches: set[str] = set()
    text = f"{item.title} {item.summary}".lower()
    if keywords:
        for keyword in keywords:
            token = keyword.lower().strip()
            if token and token in text:
                matches.update(code.strip().upper() for code in base_codes if code)
                break
    else:
        matches.update(code.strip().upper() for code in base_codes if code)

    detected = _detect_ts_codes(text)
    matches.update(detected)
    return [code for code in matches if code]


def _detect_ts_codes(text: str) -> List[str]:
    codes: set[str] = set()
    for match in A_SH_CODE_PATTERN.finditer(text):
        digits, suffix = match.groups()
        if suffix:
            codes.add(f"{digits}.{suffix.upper()}")
        else:
            exchange = "SH" if digits.startswith(tuple("569")) else "SZ"
            codes.add(f"{digits}.{exchange}")
    for match in HK_CODE_PATTERN.finditer(text):
        digits = match.group(1)
        codes.add(f"{digits.zfill(4)}.HK")
    return sorted(codes)


def _estimate_sentiment(text: str) -> float:
    normalized = text.lower()
    score = 0
    for keyword in POSITIVE_KEYWORDS:
        if keyword.lower() in normalized:
            score += 1
    for keyword in NEGATIVE_KEYWORDS:
        if keyword.lower() in normalized:
            score -= 1
    if score == 0:
        return 0.0
    return max(-1.0, min(1.0, score / 3.0))


def _estimate_heat(
    published: datetime,
    now: datetime,
    code_count: int,
    sentiment: float,
) -> float:
    delta_hours = max(0.0, (now - published).total_seconds() / 3600.0)
    recency = max(0.0, 1.0 - min(delta_hours, 72.0) / 72.0)
    coverage_bonus = min(code_count, 3) * 0.05
    sentiment_bonus = min(abs(sentiment) * 0.1, 0.2)
    heat = recency + coverage_bonus + sentiment_bonus
    return max(0.0, min(1.0, round(heat, 4)))


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is not None:
            dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt
    except (TypeError, ValueError):
        pass

    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value[:19], fmt)
        except ValueError:
            continue
    return None


def _clean_text(value: Optional[str]) -> str:
    if not value:
        return ""
    text = re.sub(r"<[^>]+>", " ", value)
    return re.sub(r"\s+", " ", text).strip()


def _truncate(value: str, length: int) -> str:
    if len(value) <= length:
        return value
    return value[: length - 3].rstrip() + "..."


def _normalise_item_id(
    raw_id: Optional[str], link: str, title: str, published: datetime
) -> str:
    candidate = (raw_id or link or title).strip()
    if candidate:
        return candidate
    fingerprint = f"{title}|{published.isoformat()}"
    return hashlib.blake2s(fingerprint.encode("utf-8"), digest_size=16).hexdigest()


def _source_from_url(url: str) -> str:
    try:
        parsed = urlparse(url)
    except ValueError:
        return url
    host = parsed.netloc or url
    return host.lower()


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _child_text(node: ET.Element, candidates: set[str]) -> Optional[str]:
    for child in node:
        name = _local_name(child.tag)
        if name in candidates and child.text:
            return child.text.strip()
        if name == "link":
            href = child.attrib.get("href")
            if href:
                return href.strip()
    return None


__all__ = [
    "RssFeedConfig",
    "RssItem",
    "fetch_rss_feed",
    "deduplicate_items",
    "save_news_items",
    "ingest_configured_rss",
    "resolve_rss_sources",
]
def _rss_source_key(url: str) -> str:
    return f"RSS|{url}".strip()
