"""RSS ingestion utilities for news sentiment and heat scoring."""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple
from urllib.parse import urlparse, urljoin
from xml.etree import ElementTree as ET

import requests
from requests import RequestException

import hashlib
import random
import time

from app.ingest.entity_recognition import company_mapper, initialize_company_mapping

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
    # 中文积极关键词
    "利好", "增长", "超预期", "创新高", "增持", "回购", "盈利", 
    "高增长", "业绩好", "优秀", "强劲", "突破", "新高", "上升",
    "上涨", "反弹", "复苏", "景气", "扩张", "加速", "改善",
    "提升", "增加", "优化", "利好消息", "超预期", "超出预期",
    "盈利超预期", "利润增长", "收入增长", "订单增长", "销量增长",
    "高景气", "量价齐升", "拐点", "反转", "政策利好", "政策支持",
    # 英文积极关键词
    "strong", "beat", "upgrade", "growth", "positive", "better",
    "exceed", "surpass", "outperform", "rally", "bullish", "upbeat",
    "improve", "increase", "rise", "gain", "profit", "earnings",
    "recovery", "expansion", "boom", "upside", "promising",
)
NEGATIVE_KEYWORDS: Tuple[str, ...] = (
    # 中文消极关键词
    "利空", "下跌", "亏损", "裁员", "违约", "处罚", "暴跌", "减持",
    "业绩差", "下滑", "下降", "恶化", "亏损", "不及预期", "低于预期",
    "业绩下滑", "利润下降", "收入下降", "订单减少", "销量减少",
    "利空消息", "不及预期", "低于预期", "亏损超预期", "利润下滑",
    "需求萎缩", "量价齐跌", "拐点向下", "政策利空", "政策收紧",
    "监管收紧", "处罚", "调查", "违规", "风险", "警示", "预警",
    "降级", "抛售", "减持", "暴跌", "大跌", "下挫", "阴跌",
    # 英文消极关键词
    "downgrade", "miss", "weak", "decline", "negative", "worse",
    "drop", "fall", "loss", "losses", "slowdown", "contract",
    "bearish", "pessimistic", "worsen", "decrease", "reduce",
    "slide", "plunge", "crash", "deteriorate", "risk", "warning",
    "regulatory", "penalty", "investigation",
)

A_SH_CODE_PATTERN = re.compile(r"\b(\d{6})(?:\.(SH|SZ))?\b", re.IGNORECASE)
HK_CODE_PATTERN = re.compile(r"\b(\d{4})\.HK\b", re.IGNORECASE)

# 行业关键词映射表
INDUSTRY_KEYWORDS: Dict[str, List[str]] = {
    "半导体": ["半导体", "芯片", "集成电路", "IC", "晶圆", "封装", "设计", "制造", "光刻"],
    "新能源": ["新能源", "光伏", "太阳能", "风电", "风电设备", "锂电池", "储能", "氢能"],
    "医药": ["医药", "生物制药", "创新药", "医疗器械", "疫苗", "CXO", "CDMO", "CRO"],
    "消费": ["消费", "食品", "饮料", "白酒", "啤酒", "乳制品", "零食", "零售", "家电"],
    "科技": ["科技", "人工智能", "AI", "云计算", "大数据", "互联网", "软件", "SaaS"],
    "金融": ["银行", "保险", "券商", "证券", "金融", "资管", "基金", "投资"],
    "地产": ["房地产", "地产", "物业", "建筑", "建材", "家居"],
    "汽车": ["汽车", "新能源汽车", "智能汽车", "自动驾驶", "零部件", "锂电"],
}


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
class StockMention:
    """A mention of a stock in text."""
    matched_text: str
    ts_code: str
    match_type: str  # 'code', 'full_name', 'short_name', 'alias'
    context: str     # 相关的上下文片段
    confidence: float  # 匹配的置信度

@dataclass
class RssItem:
    """Structured representation of an RSS entry."""

    id: str
    title: str
    link: str
    published: datetime
    summary: str
    source: str
    ts_codes: List[str] = field(default_factory=list)
    stock_mentions: List[StockMention] = field(default_factory=list)
    industries: List[str] = field(default_factory=list)
    important_keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, object] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize company mapper if not already initialized."""
        # 测试环境下跳过数据库初始化
        if not hasattr(self, '_skip_db_init'):  # 仅在非测试环境下初始化
            from app.utils.db import db_session
            
            # 如果company_mapper还没有数据，初始化它
            if not company_mapper.name_to_code:
                with db_session() as conn:
                    initialize_company_mapping(conn)
    
    def extract_entities(self) -> None:
        """Extract and validate entity mentions from title and summary."""
        # 分别处理标题和摘要
        title_matches = company_mapper.find_codes(self.title)
        summary_matches = company_mapper.find_codes(self.summary)
        
        # 按优先级合并去重后的匹配
        code_best_matches = {}  # ts_code -> (matched_text, match_type, is_title, context)
        
        # 优先级顺序: 代码 > 全称 > 简称 > 别名
        priority = {'code': 0, 'full_name': 1, 'short_name': 2, 'alias': 3}
        
        for matches, text, is_title in [(title_matches, self.title, True), 
                                      (summary_matches, self.summary, False)]:
            for matched_text, ts_code, match_type in matches:
                # 提取上下文
                context = self._extract_context(text, matched_text)
                
                # 如果是新代码或优先级更高的匹配
                if (ts_code not in code_best_matches or 
                    priority[match_type] < priority[code_best_matches[ts_code][1]]):
                    code_best_matches[ts_code] = (matched_text, match_type, is_title, context)
        
        # 创建股票提及列表
        for ts_code, (matched_text, match_type, is_title, context) in code_best_matches.items():
            confidence = self._calculate_confidence(match_type, matched_text, context, is_title)
            
            mention = StockMention(
                matched_text=matched_text,
                ts_code=ts_code,
                match_type=match_type,
                context=context,
                confidence=confidence
            )
            self.stock_mentions.append(mention)
        
        # 更新ts_codes列表，只包含高置信度的匹配
        self.ts_codes = list(set(
            mention.ts_code
            for mention in self.stock_mentions
            if mention.confidence > 0.7  # 只保留高置信度的匹配
        ))
        
        # 提取行业关键词
        self.extract_industries()
        
        # 提取重要关键词
        self.extract_important_keywords()
        
    def _extract_context(self, text: str, matched_text: str) -> str:
        """提取匹配文本的上下文，尽量提取完整的句子."""
        # 找到匹配文本的位置
        start_pos = text.find(matched_text)
        if start_pos == -1:
            return ""
            
        # 向前找到句子开始（句号、问号、感叹号或换行符之后）
        sent_start = start_pos
        while sent_start > 0:
            if text[sent_start-1] in '。？！\n':
                break
            sent_start -= 1
            
        # 向后找到句子结束
        sent_end = start_pos + len(matched_text)
        while sent_end < len(text):
            if text[sent_end] in '。？！\n':
                sent_end += 1
                break
            sent_end += 1
            
        # 如果上下文太长，则截取固定长度
        context = text[sent_start:sent_end].strip()
        if len(context) > 100:  # 最大上下文长度
            start = max(0, start_pos - 30)
            end = min(len(text), start_pos + len(matched_text) + 30)
            context = text[start:end].strip()
            
        return context
        
    def extract_industries(self) -> None:
        """从新闻标题和摘要中提取行业关键词."""
        content = f"{self.title} {self.summary}".lower()
        found_industries = set()
        
        # 对每个行业检查其关键词
        for industry, keywords in INDUSTRY_KEYWORDS.items():
            # 如果找到任意关键词，认为属于该行业
            if any(keyword.lower() in content for keyword in keywords):
                found_industries.add(industry)
                
        self.industries = list(found_industries)
        
    def extract_important_keywords(self) -> None:
        """提取重要关键词，包括积极/消极情感词和特定事件."""
        content = f"{self.title} {self.summary}".lower()
        found_keywords = set()
        
        # 1. 检查积极关键词
        for keyword in POSITIVE_KEYWORDS:
            if keyword.lower() in content:
                found_keywords.add(f"+{keyword}")  # 加前缀表示积极
                
        # 2. 检查消极关键词
        for keyword in NEGATIVE_KEYWORDS:
            if keyword.lower() in content:
                found_keywords.add(f"-{keyword}")  # 加前缀表示消极
                
        # 3. 检查特定事件关键词
        event_keywords = {
            # 公司行为
            "收购": "M&A",
            "并购": "M&A",
            "重组": "重组",
            "分拆": "分拆",
            "上市": "IPO",
            # 财务事件
            "业绩": "业绩",
            "亏损": "业绩预警",
            "盈利": "业绩预增",
            "分红": "分红",
            "回购": "回购",
            # 监管事件
            "立案": "监管",
            "调查": "监管",
            "问询": "监管",
            "处罚": "处罚",
            # 重大项目
            "中标": "中标",
            "签约": "签约",
            "战略合作": "合作",
        }
        
        for trigger, event in event_keywords.items():
            if trigger in content:
                found_keywords.add(f"#{event}")  # 加前缀表示事件
                
        self.important_keywords = list(found_keywords)
        
    def _calculate_confidence(self, match_type: str, matched_text: str, context: str, is_title: bool = False) -> float:
        """计算实体匹配的置信度.
        
        考虑以下因素：
        1. 匹配类型的基础置信度
        2. 实体在文本中的位置（标题/开头更重要）
        3. 上下文关键词
        4. 股票相关动词
        5. 实体的完整性
        """
        # 基础置信度
        base_confidence = {
            'code': 0.9,      # 直接的股票代码匹配
            'full_name': 0.85,# 完整公司名称匹配
            'short_name': 0.7,# 公司简称匹配
            'alias': 0.6      # 别名匹配
        }.get(match_type, 0.5)
        
        confidence = base_confidence
        context_lower = context.lower()
        
        # 1. 位置加权
        if is_title:
            confidence += 0.1
        if context.startswith(matched_text):
            confidence += 0.05
            
        # 2. 实体完整性检查
        if match_type == 'code' and '.' in matched_text:  # 完整股票代码（带市场后缀）
            confidence += 0.05
        elif match_type == 'full_name' and any(suffix in matched_text for suffix in ["股份有限公司", "有限公司"]):
            confidence += 0.05
            
        # 3. 上下文关键词
        context_bonus = 0.0
        corporate_terms = ["公司", "集团", "企业", "上市", "控股", "总部"]
        if any(term in context_lower for term in corporate_terms):
            context_bonus += 0.1
            
        # 4. 股票相关动词
        stock_verbs = ["发布", "公告", "披露", "表示", "报告", "投资", "回购", "增持", "减持"]
        if any(verb in context_lower for verb in stock_verbs):
            context_bonus += 0.05
            
        # 5. 财务/业务相关词汇
        business_terms = ["业绩", "营收", "利润", "股价", "市值", "经营", "产品", "服务", "战略"]
        if any(term in context_lower for term in business_terms):
            context_bonus += 0.05
            
        # 限制上下文加成的最大值
        confidence += min(context_bonus, 0.2)
            
        # 确保置信度在0-1之间
        return min(1.0, max(0.0, confidence))


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


def _canonical_link(item: RssItem) -> str:
    link = (item.link or "").strip().lower()
    if link:
        return link
    if item.id:
        return item.id
    fingerprint = f"{item.title}|{item.published.isoformat() if item.published else ''}"
    return hashlib.sha1(fingerprint.encode("utf-8")).hexdigest()


def _is_gdelt_item(item: RssItem) -> bool:
    metadata = item.metadata or {}
    return metadata.get("source_type") == "gdelt" or bool(metadata.get("source_key"))


def deduplicate_items(items: Iterable[RssItem]) -> List[RssItem]:
    """Drop duplicate stories by canonical link while preferring GDELT sources."""

    selected: Dict[str, RssItem] = {}
    order: List[str] = []

    for item in items:
        preassigned_codes = list(item.ts_codes or [])
        # 提取实体和相关信息
        item.extract_entities()

        keep = False
        if _is_gdelt_item(item):
            keep = True
        elif item.stock_mentions:
            keep = True
        elif preassigned_codes:
            if not item.ts_codes:
                item.ts_codes = preassigned_codes
            keep = True

        if not keep:
            continue

        key = _canonical_link(item)
        existing = selected.get(key)
        if existing is None:
            selected[key] = item
            order.append(key)
            continue

        if _is_gdelt_item(item) and not _is_gdelt_item(existing):
            selected[key] = item
        elif _is_gdelt_item(item) == _is_gdelt_item(existing):
            if item.published and existing.published:
                if item.published > existing.published:
                    selected[key] = item
            else:
                selected[key] = item

    return [selected[key] for key in order if key in selected]


def save_news_items(items: Iterable[RssItem]) -> int:
    """Persist RSS items into the `news` table."""

    initialize_database()
    now = datetime.utcnow()
    rows: List[Tuple[object, ...]] = []

    processed = 0
    gdelt_urls: Set[str] = set()
    for item in items:
        text_payload = f"{item.title}\n{item.summary}"
        sentiment = _estimate_sentiment(text_payload)
        base_codes = tuple(code for code in item.ts_codes if code)
        # 更新调用，添加新增的参数
        heat = _estimate_heat(
            item.published, 
            now, 
            len(base_codes), 
            sentiment, 
            text_length=len(text_payload),
            industry_count=len(item.industries)
        )
        # 构建包含更多信息的entities对象
        entity_payload = {
            "ts_codes": list(base_codes),
            "source_url": item.link,
            "industries": item.industries,  # 添加行业信息
            "important_keywords": item.important_keywords,  # 添加重要关键词
            "text_length": len(text_payload),  # 添加文本长度信息
        }
        if item.metadata:
            entity_payload["metadata"] = dict(item.metadata)
            if _is_gdelt_item(item) and item.link:
                gdelt_urls.add(item.link.strip())
        entities = json.dumps(entity_payload, ensure_ascii=False)
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
            if gdelt_urls:
                conn.executemany(
                    """
                    DELETE FROM news
                    WHERE url = ?
                      AND (json_extract(entities, '$.metadata.source_type') IS NULL
                           OR json_extract(entities, '$.metadata.source_type') != 'gdelt')
                    """,
                    [(url,) for url in gdelt_urls],
                )
            conn.executemany(
                """
                INSERT OR REPLACE INTO news
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
    """为新闻条目分配股票代码，并同时提取行业信息和重要关键词"""
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
    
    # 检测相关行业
    item.industries = _detect_industries(text)
    
    # 提取重要关键词
    item.important_keywords = _extract_important_keywords(text)
    
    return [code for code in matches if code]

def _detect_industries(text: str) -> List[str]:
    """根据文本内容检测相关行业"""
    detected_industries = []
    text_lower = text.lower()
    
    for industry, keywords in INDUSTRY_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                if industry not in detected_industries:
                    detected_industries.append(industry)
                # 一个行业匹配一个关键词即可
                break
    
    return detected_industries

def _extract_important_keywords(text: str) -> List[str]:
    """从文本中提取重要关键词，包括情感词和行业词"""
    important_keywords = []
    text_lower = text.lower()
    
    # 提取情感关键词
    for keyword in POSITIVE_KEYWORDS + NEGATIVE_KEYWORDS:
        if keyword.lower() in text_lower and keyword not in important_keywords:
            important_keywords.append(keyword)
    
    # 提取行业关键词
    for keywords in INDUSTRY_KEYWORDS.values():
        for keyword in keywords:
            if keyword.lower() in text_lower and keyword not in important_keywords:
                important_keywords.append(keyword)
    
    # 限制关键词数量
    return important_keywords[:10]  # 最多返回10个关键词


def _detect_ts_codes(text: str) -> List[str]:
    """增强的股票代码检测函数，改进代码识别的准确性"""
    codes: set[str] = set()
    
    # 检测A股和港股代码
    for match in A_SH_CODE_PATTERN.finditer(text):
        digits, suffix = match.groups()
        # 确保是有效的股票代码（避免误识别其他6位数字）
        if _is_valid_stock_code(digits, suffix):
            if suffix:
                codes.add(f"{digits}.{suffix.upper()}")
            else:
                # 根据数字范围推断交易所
                exchange = "SH" if digits.startswith(tuple("569")) else "SZ"
                codes.add(f"{digits}.{exchange}")
    
    # 检测港股代码
    for match in HK_CODE_PATTERN.finditer(text):
        digits = match.group(1)
        # 补全为4位数字
        codes.add(f"{digits.zfill(4)}.HK")
    
    # 检测可能的股票简称和代码关联
    codes.update(_detect_codes_by_company_name(text))
    
    return sorted(codes)


def _is_valid_stock_code(digits: str, suffix: Optional[str]) -> bool:
    """验证是否为有效的股票代码"""
    # 排除明显不是股票代码的数字组合
    if len(digits) != 6:
        return False
    
    # 上海证券交易所股票代码范围：600000-609999 (A股), 688000-688999 (科创板), 500000-599999 (基金)
    # 深圳证券交易所股票代码范围：000001-009999 (主板), 300000-309999 (创业板), 002000-002999 (中小板)
    # 这里做简单的范围验证，避免误识别
    if suffix and suffix.upper() in ("SH", "SZ"):
        return True
    
    # 没有后缀时，通过数字范围判断
    code_int = int(digits)
    return (
            (600000 <= code_int <= 609999) or  # 上交所A股
            (688000 <= code_int <= 688999) or  # 科创板
            (1 <= code_int <= 9999) or         # 深交所主板 (去掉前导零)
            (300000 <= code_int <= 309999) or  # 创业板
            (2000 <= code_int <= 2999)         # 中小板 (去掉前导零)
        )


def _detect_codes_by_company_name(text: str) -> List[str]:
    """通过公司名称识别可能的股票代码
    注意：这是一个简化版本，实际应用中可能需要更复杂的映射表
    """
    # 这里仅作为示例，实际应用中应该使用更完善的公司名称-代码映射
    # 这里我们返回空列表，但保留函数结构以便未来扩展
    return []


def _estimate_sentiment(text: str) -> float:
    """增强的情感分析函数，提高情绪识别准确率"""
    normalized = text.lower()
    score = 0.0
    positive_matches = 0
    negative_matches = 0
    
    # 计算关键词匹配次数
    for keyword in POSITIVE_KEYWORDS:
        if keyword.lower() in normalized:
            # 情感词权重：根据重要性调整权重
            weight = _get_sentiment_keyword_weight(keyword, positive=True)
            score += weight
            positive_matches += 1
    
    for keyword in NEGATIVE_KEYWORDS:
        if keyword.lower() in normalized:
            # 情感词权重：根据重要性调整权重
            weight = _get_sentiment_keyword_weight(keyword, positive=False)
            score -= weight
            negative_matches += 1
    
    # 处理无匹配的情况
    if positive_matches == 0 and negative_matches == 0:
        # 尝试通过否定词和转折词分析
        return _analyze_neutral_text(normalized)
    
    # 归一化情感得分
    max_score = max(3.0, positive_matches + negative_matches)  # 确保分母不为零且有合理缩放
    normalized_score = score / max_score
    
    # 限制在[-1.0, 1.0]范围内
    return max(-1.0, min(1.0, normalized_score))


def _get_sentiment_keyword_weight(keyword: str, positive: bool) -> float:
    """根据关键词的重要性返回不同的权重"""
    # 基础权重
    base_weight = 1.0
    
    # 强情感词增加权重
    strong_positive = ["超预期", "超出预期", "盈利超预期", "利好", "upgrade", "beat"]
    strong_negative = ["不及预期", "低于预期", "亏损超预期", "利空", "downgrade", "miss"]
    
    if positive:
        if keyword in strong_positive:
            return base_weight * 1.5
    else:
        if keyword in strong_negative:
            return base_weight * 1.5
    
    # 弱情感词降低权重
    weak_positive = ["增长", "改善", "增加", "rise", "increase", "improve"]
    weak_negative = ["下降", "减少", "恶化", "drop", "decrease", "decline"]
    
    if positive:
        if keyword in weak_positive:
            return base_weight * 0.8
    else:
        if keyword in weak_negative:
            return base_weight * 0.8
    
    return base_weight


def _analyze_neutral_text(text: str) -> float:
    """分析无明显情感词的文本"""
    # 检查是否包含否定词和情感词的组合
    negation_words = ["不", "非", "无", "未", "没有", "不是", "不会"]
    
    # 简单的否定模式识别（实际应用中可能需要更复杂的NLP处理）
    for neg_word in negation_words:
        neg_pos = text.find(neg_word)
        if neg_pos != -1:
            # 检查否定词后面是否有积极或消极关键词
            window = text[neg_pos:neg_pos + 30]  # 检查否定词后30个字符
            for pos_word in POSITIVE_KEYWORDS:
                if pos_word.lower() in window:
                    return -0.3  # 否定积极词，轻微消极
            for neg_word2 in NEGATIVE_KEYWORDS:
                if neg_word2.lower() in window:
                    return 0.3  # 否定消极词，轻微积极
    
    # 检查是否包含中性偏积极或偏消极的表达
    neutral_positive = ["稳定", "平稳", "正常", "符合预期", "stable", "steady", "normal"]
    neutral_negative = ["波动", "不确定", "风险", "挑战", "fluctuate", "uncertain", "risk"]
    
    for word in neutral_positive:
        if word.lower() in text:
            return 0.1
    for word in neutral_negative:
        if word.lower() in text:
            return -0.1
    
    return 0.0


def _estimate_heat(
    published: datetime,
    now: datetime,
    code_count: int,
    sentiment: float,
    text_length: int = 0,
    source_quality: float = 1.0,
    industry_count: int = 0,
) -> float:
    """增强的热度评分函数，考虑更多影响热度的因素"""
    # 时效性得分（基础权重0.5）
    delta_hours = max(0.0, (now - published).total_seconds() / 3600.0)
    # 根据时间衰减曲线调整时效性得分
    if delta_hours < 1:
        recency = 1.0  # 1小时内的新闻时效性最高
    elif delta_hours < 6:
        recency = 0.8  # 1-6小时
    elif delta_hours < 24:
        recency = 0.6  # 6-24小时
    elif delta_hours < 48:
        recency = 0.3  # 24-48小时
    else:
        recency = 0.1  # 超过48小时
    
    # 覆盖度得分（基础权重0.2）- 涉及的股票数量
    coverage_score = min(code_count / 5, 1.0) * 0.2
    
    # 情感强度得分（基础权重0.15）
    sentiment_score = min(abs(sentiment), 1.0) * 0.15
    
    # 内容丰富度得分（基础权重0.1）
    content_score = min(text_length / 1000, 1.0) * 0.1  # 基于文本长度评估
    
    # 行业覆盖度得分（基础权重0.05）
    industry_score = min(industry_count / 3, 1.0) * 0.05  # 涉及多个行业可能更具影响力
    
    # 来源质量调整因子（0.5-1.5）
    source_adjustment = source_quality
    
    # 计算综合热度得分
    heat = (recency + coverage_score + sentiment_score + content_score + industry_score) * source_adjustment
    
    # 限制在[0.0, 1.0]范围内并保留4位小数
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
