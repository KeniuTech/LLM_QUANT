"""GDELT Doc API ingestion utilities built on top of gdeltdoc."""
from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Union

try:  # pragma: no cover - optional dependency
    from gdeltdoc import GdeltDoc, Filters  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency
    GdeltDoc = None  # type: ignore[assignment]
    Filters = None  # type: ignore[assignment]

from app.utils.config import get_config
from app.utils.db import db_session
from app.utils.logging import get_logger

from . import rss as rss_ingest

LOGGER = get_logger(__name__)
LOG_EXTRA = {"stage": "gdelt_ingest"}
DateLike = Union[date, datetime]

_LANGUAGE_CANONICAL: Dict[str, str] = {
    "en": "en",
    "eng": "en",
    "english": "en",
    "zh": "zh",
    "zho": "zh",
    "zh-cn": "zh",
    "zh-hans": "zh",
    "zh-hant": "zh",
    "zh_tw": "zh",
    "chinese": "zh",
}

_LAST_INGEST_STATS: Dict[str, int] = {"fetched": 0, "deduped": 0, "inserted": 0}


@dataclass
class GdeltSourceConfig:
    """Configuration describing a single GDELT filter set."""

    key: str
    label: str
    filters: Dict[str, object] = field(default_factory=dict)
    ts_codes: Sequence[str] = field(default_factory=tuple)
    keywords: Sequence[str] = field(default_factory=tuple)
    num_records: int = 50


def resolve_gdelt_sources() -> List[GdeltSourceConfig]:
    """Resolve configured GDELT filter groups."""

    cfg = get_config()
    raw = getattr(cfg, "gdelt_sources", None) or {}

    sources: List[GdeltSourceConfig] = []
    if isinstance(raw, dict):
        for key, data in raw.items():
            if not isinstance(data, dict):
                continue
            if not data.get("enabled", True):
                continue
            label = str(data.get("label") or key)
            filters = data.get("filters") if isinstance(data.get("filters"), dict) else {}
            ts_codes = [
                str(code).strip().upper()
                for code in data.get("ts_codes", [])
                if isinstance(code, str) and code.strip()
            ]
            keywords = [
                str(token).strip()
                for token in data.get("keywords", [])
                if isinstance(token, str) and token.strip()
            ]
            num_records = data.get("num_records")
            if not isinstance(num_records, int) or num_records <= 0:
                num_records = 50
            sources.append(
                GdeltSourceConfig(
                    key=str(key),
                    label=label,
                    filters=dict(filters),
                    ts_codes=tuple(ts_codes),
                    keywords=tuple(keywords),
                    num_records=num_records,
                )
            )
    return sources


def _ensure_datetime(value: DateLike, *, start_of_day: bool = True) -> datetime:
    if isinstance(value, datetime):
        return _normalize_timestamp(value)
    if start_of_day:
        return datetime.combine(value, datetime.min.time())
    return datetime.combine(value, datetime.max.time())


def _normalize_timestamp(value: datetime) -> datetime:
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def _load_last_published(source_key: str) -> Optional[datetime]:
    try:
        with db_session(read_only=True) as conn:
            row = conn.execute(
                "SELECT last_published FROM ingest_state WHERE source = ?",
                (source_key,),
            ).fetchone()
    except sqlite3.OperationalError:
        return None
    if not row:
        return None
    raw = row["last_published"]
    if not raw:
        return None
    try:
        return _normalize_timestamp(datetime.fromisoformat(raw))
    except ValueError:
        LOGGER.debug("无法解析 GDELT 状态时间 source=%s value=%s", source_key, raw, extra=LOG_EXTRA)
        return None


def _save_last_published(source_key: str, published: datetime) -> None:
    timestamp = _normalize_timestamp(published).isoformat()
    try:
        with db_session() as conn:
            conn.execute(
                """
                INSERT INTO ingest_state (source, last_published)
                VALUES (?, ?)
                ON CONFLICT(source) DO UPDATE SET last_published = excluded.last_published
                """,
                (source_key, timestamp),
            )
    except sqlite3.OperationalError:
        LOGGER.debug("写入 ingest_state 失败，表可能不存在", extra=LOG_EXTRA)


def _parse_gdelt_datetime(raw: object) -> datetime:
    if isinstance(raw, datetime):
        return _normalize_timestamp(raw)
    if raw is None:
        return _normalize_timestamp(datetime.utcnow())
    text = str(raw).strip()
    if not text:
        return _normalize_timestamp(datetime.utcnow())
    # Common GDELT formats: YYYYMMDDHHMMSS or ISO8601
    try:
        if text.isdigit() and len(text) == 14:
            return _normalize_timestamp(datetime.strptime(text, "%Y%m%d%H%M%S"))
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        return _normalize_timestamp(datetime.fromisoformat(text))
    except ValueError:
        pass
    try:
        return _normalize_timestamp(datetime.strptime(text, "%Y-%m-%d %H:%M:%S"))
    except ValueError:
        LOGGER.debug("无法解析 GDELT 日期：%s", text, extra=LOG_EXTRA)
        return _normalize_timestamp(datetime.utcnow())


def _build_rss_item(record: Dict[str, object], config: GdeltSourceConfig) -> Optional[rss_ingest.RssItem]:
    url = record.get("url") or record.get("url_mobile")
    if not isinstance(url, str) or not url.strip():
        return None
    url = url.strip()

    title = record.get("title") or record.get("seendate")
    if not isinstance(title, str) or not title.strip():
        title = url
    title = title.strip()

    published_raw = (
        record.get("seendate")
        or record.get("publishDate")
        or record.get("date")
        or record.get("firstseendate")
    )
    published = _parse_gdelt_datetime(published_raw)

    summary_candidates: Iterable[object] = (
        record.get("summary"),
        record.get("snippet"),
        record.get("excerpt"),
        record.get("altText"),
        record.get("domain"),
    )
    summary = ""
    for candidate in summary_candidates:
        if isinstance(candidate, str) and candidate.strip():
            summary = candidate.strip()
            break
    if not summary:
        source_country = record.get("sourcecountry")
        language = record.get("language")
        details = [
            str(value).strip()
            for value in (source_country, language)
            if isinstance(value, str) and value.strip()
        ]
        summary = " / ".join(details) if details else title

    source = record.get("sourcecommonname") or record.get("domain")
    if not isinstance(source, str) or not source.strip():
        source = config.label or "GDELT"
    source = source.strip()

    fingerprint = f"{url}|{published.isoformat()}|{config.key}"
    article_id = hashlib.blake2s(fingerprint.encode("utf-8"), digest_size=16).hexdigest()

    return rss_ingest.RssItem(
        id=article_id,
        title=title,
        link=url,
        published=published,
        summary=summary,
        source=source,
        metadata={
            "source_key": config.key,
            "source_label": config.label,
            "source_type": "gdelt",
        },
    )


def fetch_gdelt_articles(
    config: GdeltSourceConfig,
    *,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> List[rss_ingest.RssItem]:
    """Fetch article list from GDELT based on the supplied configuration."""

    if GdeltDoc is None or Filters is None:
        LOGGER.warning("未安装 gdeltdoc，跳过 GDELT 拉取", extra=LOG_EXTRA)
        return []

    base_filters = dict(config.filters)
    base_filters.setdefault("num_records", config.num_records)
    original_timespan = base_filters.get("timespan")
    filters_kwargs = dict(base_filters)

    def _strip_quotes(token: str) -> str:
        stripped = token.strip()
        if (stripped.startswith('"') and stripped.endswith('"')) or (stripped.startswith("'") and stripped.endswith("'")):
            return stripped[1:-1].strip()
        return stripped

    def _normalize_keywords(value: object) -> object:
        if isinstance(value, str):
            parts = [part.strip() for part in re.split(r"\s+OR\s+", value) if part.strip()]
            if len(parts) <= 1:
                return _strip_quotes(value)
            normalized = [_strip_quotes(part) for part in parts]
            return normalized
        if isinstance(value, (list, tuple, set)):
            normalized = [_strip_quotes(str(item)) for item in value if str(item).strip()]
            return normalized
        return value

    def _sanitize(filters: Dict[str, object]) -> Dict[str, object]:
        cleaned = dict(filters)
        def _normalise_sequence_field(field: str, mapping: Optional[Dict[str, str]] = None) -> None:
            value = cleaned.get(field)
            if isinstance(value, (list, tuple, set)):
                items: List[str] = []
                for token in value:
                    if not token:
                        continue
                    token_str = str(token).strip()
                    if not token_str:
                        continue
                    mapped = mapping.get(token_str.lower(), token_str) if mapping else token_str
                    if mapped not in items:
                        items.append(mapped)
                if not items:
                    cleaned.pop(field, None)
                elif len(items) == 1:
                    cleaned[field] = items[0]
                else:
                    cleaned[field] = items
            elif isinstance(value, str):
                stripped = value.strip()
                if not stripped:
                    cleaned.pop(field, None)
                elif mapping:
                    cleaned[field] = mapping.get(stripped.lower(), stripped)
                else:
                    cleaned[field] = stripped
            elif value is None:
                cleaned.pop(field, None)

        _normalise_sequence_field("language", _LANGUAGE_CANONICAL)
        _normalise_sequence_field("country")
        _normalise_sequence_field("domain")
        _normalise_sequence_field("domain_exact")

        keyword_value = cleaned.get("keyword")
        if keyword_value is not None:
            normalized_keyword = _normalize_keywords(keyword_value)
            if isinstance(normalized_keyword, list):
                if not normalized_keyword:
                    cleaned.pop("keyword", None)
                elif len(normalized_keyword) == 1:
                    cleaned["keyword"] = normalized_keyword[0]
                else:
                    cleaned["keyword"] = normalized_keyword
            elif isinstance(normalized_keyword, str):
                cleaned["keyword"] = normalized_keyword
            else:
                cleaned.pop("keyword", None)

        return cleaned

    if start or end:
        filters_kwargs.pop("timespan", None)
    if start:
        filters_kwargs["start_date"] = start
    if end:
        filters_kwargs["end_date"] = end

    filters_kwargs = _sanitize(filters_kwargs)

    client = GdeltDoc()

    def _run_query(kwargs: Dict[str, object]) -> Optional[pd.DataFrame]:
        try:
            filter_obj = Filters(**kwargs)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("GDELT 过滤器解析失败 key=%s err=%s", config.key, exc, extra=LOG_EXTRA)
            return None
        try:
            return client.article_search(filter_obj)
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            if "Invalid/Unsupported Language" in message and kwargs.get("language"):
                LOGGER.warning(
                    "GDELT 语言过滤不被支持，移除后重试 key=%s languages=%s",
                    config.key,
                    kwargs.get("language"),
                    extra=LOG_EXTRA,
                )
                retry_kwargs = dict(kwargs)
                retry_kwargs.pop("language", None)
                return _run_query(retry_kwargs)
            LOGGER.warning("GDELT 请求失败 key=%s err=%s", config.key, exc, extra=LOG_EXTRA)
            return None

    df = _run_query(filters_kwargs)
    if df is None or df.empty:
        if (start or end) and original_timespan:
            fallback_kwargs = dict(base_filters)
            fallback_kwargs["timespan"] = original_timespan
            fallback_kwargs.pop("start_date", None)
            fallback_kwargs.pop("end_date", None)
            fallback_kwargs = _sanitize(fallback_kwargs)
            LOGGER.info(
                "GDELT 无匹配结果，尝试使用 timespan 回退 key=%s timespan=%s",
                config.key,
                original_timespan,
                extra=LOG_EXTRA,
            )
            df = _run_query(fallback_kwargs)
    if df is None or df.empty:
        LOGGER.info("GDELT 无匹配结果 key=%s", config.key, extra=LOG_EXTRA)
        return []

    items: List[rss_ingest.RssItem] = []
    for record in df.to_dict(orient="records"):
        item = _build_rss_item(record, config)
        if not item:
            continue
        assigned_codes = rss_ingest._assign_ts_codes(item, config.ts_codes, config.keywords)  # type: ignore[attr-defined]
        items.append(replace(item, ts_codes=tuple(assigned_codes)))
    return items


def _update_last_published_state(items: Sequence[rss_ingest.RssItem]) -> None:
    latest_by_source: Dict[str, datetime] = {}
    for item in items:
        metadata = item.metadata or {}
        source_key = str(metadata.get("source_key", ""))
        if not source_key:
            continue
        current = latest_by_source.get(source_key)
        published = item.published
        if current is None or published > current:
            latest_by_source[source_key] = published
    for source_key, timestamp in latest_by_source.items():
        _save_last_published(source_key, timestamp)


def ingest_configured_gdelt(
    start: Optional[DateLike] = None,
    end: Optional[DateLike] = None,
    *,
    incremental: bool = True,
) -> int:
    """Ingest all configured GDELT sources into the news store."""

    sources = resolve_gdelt_sources()
    if not sources:
        LOGGER.info("未配置 GDELT 来源，跳过新闻拉取", extra=LOG_EXTRA)
        return 0

    start_dt = _ensure_datetime(start) if start else None
    end_dt = _ensure_datetime(end, start_of_day=False) if end else None

    aggregated: List[rss_ingest.RssItem] = []
    fetched = 0
    for config in sources:
        source_start = start_dt
        effective_incremental = incremental
        if start_dt is not None or end_dt is not None:
            effective_incremental = False
        elif incremental:
            last_seen = _load_last_published(config.key)
            if last_seen:
                candidate = last_seen + timedelta(seconds=1)
                if source_start is None or candidate > source_start:
                    source_start = candidate
        LOGGER.info(
            "开始拉取 GDELT：%s start=%s end=%s incremental=%s",
            config.label,
            source_start.isoformat() if source_start else None,
            end_dt.isoformat() if end_dt else None,
            effective_incremental,
            extra=LOG_EXTRA,
        )

        items: List[rss_ingest.RssItem] = []
        if source_start and end_dt and source_start <= end_dt:
            chunk_start = source_start
            while chunk_start <= end_dt:
                chunk_end = min(chunk_start + timedelta(days=1) - timedelta(seconds=1), end_dt)
                chunk_items = fetch_gdelt_articles(config, start=chunk_start, end=chunk_end)
                if chunk_items:
                    items.extend(chunk_items)
                chunk_start = chunk_end + timedelta(seconds=1)
        else:
            items = fetch_gdelt_articles(config, start=source_start, end=end_dt)
        if not items:
            continue
        aggregated.extend(items)
        fetched += len(items)
        LOGGER.info("GDELT 来源 %s 返回 %s 条记录", config.label, len(items), extra=LOG_EXTRA)

    if not aggregated:
        _LAST_INGEST_STATS.update({"fetched": 0, "deduped": 0, "inserted": 0})
        return 0

    deduped = rss_ingest.deduplicate_items(aggregated)
    if not deduped:
        LOGGER.info("GDELT 数据全部为重复项，跳过落库", extra=LOG_EXTRA)
        _update_last_published_state(aggregated)
        _LAST_INGEST_STATS.update({"fetched": fetched, "deduped": 0, "inserted": 0})
        return 0

    inserted = rss_ingest.save_news_items(deduped)
    if inserted:
        _update_last_published_state(deduped)
    else:
        _update_last_published_state(aggregated)
    _LAST_INGEST_STATS.update({"fetched": fetched, "deduped": len(deduped), "inserted": inserted})
    LOGGER.info(
        "GDELT 新闻落库完成 fetched=%s deduped=%s inserted=%s",
        fetched,
        len(deduped),
        inserted,
        extra=LOG_EXTRA,
    )
    return inserted


def get_last_ingest_stats() -> Dict[str, int]:
    """Return a copy of the most recent ingestion stats."""

    return dict(_LAST_INGEST_STATS)


__all__ = [
    "GdeltSourceConfig",
    "resolve_gdelt_sources",
    "fetch_gdelt_articles",
    "ingest_configured_gdelt",
    "get_last_ingest_stats",
]
