"""Test improved entity recognition in RSS processing."""
from datetime import datetime, timezone
import pytest

from app.ingest.entity_recognition import CompanyNameMapper, company_mapper
from app.ingest.rss import RssItem, StockMention

def test_company_name_mapper():
    mapper = CompanyNameMapper()
    
    # 添加测试公司
    mapper.add_company(
        ts_code="000001.SZ",
        full_name="平安银行股份有限公司",
        short_name="平安银行",
        aliases=["平安", "PAB"]
    )
    
    # 测试名称匹配，会找到所有可能的匹配
    matches = mapper.find_codes("平安银行股份有限公司公布2025年业绩")
    assert len([m for m in matches if m[1] == "000001.SZ"]) >= 1
    # 确保有一个全称匹配
    assert any(m[2] == "full_name" and m[1] == "000001.SZ" for m in matches)
    
    # 测试简称匹配
    matches = mapper.find_codes("平安银行发布公告")
    # 应该找到简称匹配，可能还有其他匹配
    assert any(m[1] == "000001.SZ" and m[2] == "short_name" for m in matches)
    
    # 测试别名匹配
    matches = mapper.find_codes("PAB发布新产品")
    # 应该至少找到别名匹配
    assert any(m[1] == "000001.SZ" and m[2] == "alias" for m in matches)
    
    # 测试股票代码直接匹配
    matches = mapper.find_codes("000001.SZ开盘上涨")
    assert len(matches) == 1
    assert matches[0][1] == "000001.SZ"
    assert matches[0][2] == "code"

def test_rss_item_entity_extraction():
    # 使用全局company_mapper
    company_mapper.name_to_code.clear()  # 清除之前的数据
    company_mapper.add_company(
        ts_code="000001.SZ",
        full_name="平安银行股份有限公司",
        short_name="平安银行",
        aliases=["平安", "PAB"]
    )
    
    # 创建测试新闻并跳过数据库初始化
    class TestRssItem(RssItem):
        _skip_db_init = True
        
    item = TestRssItem(
        id="test_news",
        title="平安银行发布2025年业绩预告",
        link="http://example.com",
        published=datetime.now(timezone.utc),
        summary="平安银行股份有限公司（000001.SZ）今日发布2025年业绩预告",
        source="test"
    )
    
    # 提取实体
    item.extract_entities()
    
    # 验证结果：由于优先级机制，只会保留最优的匹配
    matched_types = set(m.match_type for m in item.stock_mentions)
    assert "code" in matched_types or "full_name" in matched_types  # 应该至少找到代码或全称匹配
    
    # 验证唯一股票代码
    assert len(item.ts_codes) == 1  # 只有一个唯一的股票代码
    assert item.ts_codes[0] == "000001.SZ"
    
    # 验证置信度计算
    high_confidence = [m for m in item.stock_mentions if m.confidence > 0.7]
    assert len(high_confidence) >= 1  # 至少应该有一个高置信度的匹配

def test_rss_item_context_extraction():
    # 使用全局company_mapper
    company_mapper.name_to_code.clear()  # 清除之前的数据
    company_mapper.add_company(
        ts_code="000001.SZ",
        full_name="平安银行股份有限公司",
        short_name="平安银行"
    )
    
    # 创建带有上下文的测试新闻并跳过数据库初始化
    class TestRssItem(RssItem):
        _skip_db_init = True
        
    item = TestRssItem(
        id="test_news",
        title="多家银行业绩报告",
        link="http://example.com",
        published=datetime.now(timezone.utc),
        summary="在银行业整体向好的背景下，平安银行表现突出，营收增长明显",
        source="test"
    )
    
    # 提取实体
    item.extract_entities()
    
    # 验证上下文提取
    assert len(item.stock_mentions) > 0
    mention = item.stock_mentions[0]
    assert len(mention.context) <= 70  # 上下文长度限制（30字符前后）
    assert "平安银行" in mention.context
    assert "银行业" in mention.context  # 应包含前文
    assert "营收增长" in mention.context  # 应包含后文
