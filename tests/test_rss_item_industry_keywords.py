"""Test industry and keyword extraction in RSS processing."""
from datetime import datetime, timezone

from app.ingest.rss import RssItem

def test_industry_extraction():
    """Test industry keyword extraction."""
    # 创建测试新闻并跳过数据库初始化
    class TestRssItem(RssItem):
        _skip_db_init = True
        
    item = TestRssItem(
        id="test_news",
        title="某半导体公司推出新一代芯片",
        link="http://example.com",
        published=datetime.now(timezone.utc),
        summary="该公司在集成电路领域取得重大突破，新产品将用于5G通信",
        source="test"
    )
    
    # 提取行业关键词
    item.extract_industries()
    
    # 验证结果
    assert "半导体" in item.industries
    assert len(item.industries) >= 1

def test_important_keyword_extraction():
    """Test important keyword extraction."""
    # 创建测试新闻（包含积极、消极和事件关键词）并跳过数据库初始化
    class TestRssItem(RssItem):
        _skip_db_init = True
        
    item = TestRssItem(
        id="test_news",
        title="某公司业绩超预期，同时宣布重大收购计划",
        link="http://example.com",
        published=datetime.now(timezone.utc),
        summary="营收增长显著，但部分业务亏损，将通过并购扩张",
        source="test"
    )
    
    # 提取重要关键词
    item.extract_important_keywords()
    
    # 验证结果：应该包含积极、消极和事件关键词
    keywords = set(item.important_keywords)
    
    # 检查是否包含至少一个积极关键词（前缀为+）
    assert any(k.startswith('+') for k in keywords)
    
    # 检查是否包含至少一个消极关键词（前缀为-）
    assert any(k.startswith('-') for k in keywords)
    
    # 检查是否包含至少一个事件关键词（前缀为#）
    assert any(k.startswith('#') for k in keywords)

def test_rss_item_full_extraction():
    """Test full entity, industry and keyword extraction."""
    # 创建一个包含多种信息的测试新闻并跳过数据库初始化
    class TestRssItem(RssItem):
        _skip_db_init = True
        
    item = TestRssItem(
        id="test_news",
        title="半导体行业利好：某公司重大突破",
        link="http://example.com",
        published=datetime.now(timezone.utc),
        summary="集成电路领域取得重大进展，业绩超预期，宣布增持计划",
        source="test"
    )
    
    # 提取所有信息
    item.extract_entities()  # 这将自动调用行业和关键词提取
    
    # 验证结果的完整性
    assert item.industries  # 应该至少识别出半导体行业
    assert item.important_keywords  # 应该找到关键词
    
    # 验证关键词类型的完整性
    keywords = set(item.important_keywords)
    assert any(k.startswith('+') for k in keywords)  # 积极关键词
    assert any(k.startswith('#') for k in keywords)  # 事件关键词
