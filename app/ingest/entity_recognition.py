"""Stock code mapping and entity recognition utilities."""
from __future__ import annotations

import re
import sqlite3
from typing import Dict, List, Optional, Set, Tuple

from app.utils.logging import get_logger

LOGGER = get_logger(__name__)

_COMPANY_MAPPING_INITIALIZED = False

# 股票代码正则表达式
A_SH_CODE_PATTERN = re.compile(r"\b(\d{6})(\.(?:SH|SZ))?\b", re.IGNORECASE)
HK_CODE_PATTERN = re.compile(r"\b(\d{4})\.HK\b", re.IGNORECASE)

def normalize_stock_code(code: str, explicit_market: str = None) -> str:
    """规范化股票代码格式.
    
    Args:
        code: 原始股票代码
        explicit_market: 显式指定的市场，如 'SH' 或 'SZ'
        
    Returns:
        标准格式的股票代码，如 '000001.SZ'
    """
    if '.' in code:
        return code.upper()
        
    if explicit_market:
        return f"{code}.{explicit_market.upper()}"
        
    # 根据代码规则判断市场
    if code.startswith('6'):
        return f"{code}.SH"
    elif code.startswith(('0', '3')):
        return f"{code}.SZ"
    else:
        return f"{code}.SH"  # 默认使用上交所

# 公司名称变体模式
COMPANY_SUFFIXES = ["股份", "科技", "公司", "集团", "股份有限公司", "有限公司"]

class CompanyNameMapper:
    """Map company names to stock codes with fuzzy matching."""
    
    def __init__(self):
        self.name_to_code: Dict[str, str] = {}  # 完整名称到代码映射
        self.short_names: Dict[str, str] = {}   # 简称到代码映射
        self.aliases: Dict[str, str] = {}       # 别名到代码映射
        
    def add_company(self, ts_code: str, full_name: str, short_name: str, aliases: List[str] = None):
        """Add a company to the mapping.
        
        Args:
            ts_code: Stock code in format like '000001.SZ'
            full_name: Full registered company name
            short_name: Official short name
            aliases: List of alternative names
        """
        # 存储完整名称映射
        self.name_to_code[full_name] = ts_code
        
        # 存储简称映射
        self.short_names[short_name] = ts_code
        
        # 生成和存储名称变体
        name_variants = self._generate_name_variants(full_name)
        for variant in name_variants:
            if variant not in self.aliases:
                self.aliases[variant] = ts_code
                
        # 存储额外的别名
        if aliases:
            for alias in aliases:
                if alias not in self.aliases:
                    self.aliases[alias] = ts_code
                    
    def _generate_name_variants(self, full_name: str) -> Set[str]:
        """Generate possible variants of a company name."""
        variants = set()
        
        # 仅移除整个公司类型后缀
        for suffix in COMPANY_SUFFIXES:
            if full_name.endswith(suffix):
                variant = full_name[:-len(suffix)].strip()
                if len(variant) > 2:  # 避免太短的变体
                    variants.add(variant)
                break
        
        return variants
        
    def find_codes(self, text: str) -> List[Tuple[str, str, str]]:
        """Find company mentions and corresponding stock codes in text.
        
        Returns:
            List of tuples (matched_text, stock_code, match_type)
            where match_type is one of 'code', 'full_name', 'short_name', 'alias'
        """
        matches = []
        
        # 1. 查找直接的股票代码
        for match in A_SH_CODE_PATTERN.finditer(text):
            code = match.group(1)
            explicit_market = match.group(2)[1:] if match.group(2) else None
            ts_code = normalize_stock_code(code, explicit_market)
            matches.append((match.group(), ts_code, 'code'))
            
        for match in HK_CODE_PATTERN.finditer(text):
            ts_code = match.group()
            matches.append((match.group(), ts_code, 'code'))
            
        # 2. 按优先级顺序查找公司名称
        # 完整名称优先级最高
        for name, code in self.name_to_code.items():
            if name in text:
                matches.append((name, code, 'full_name'))
                
        # 其次是简称
        for name, code in self.short_names.items():
            if name in text:
                matches.append((name, code, 'short_name'))
                
        # 最后是别名
        for alias, code in self.aliases.items():
            if alias in text:
                matches.append((alias, code, 'alias'))
                
        return matches

# 创建全局单例实例
company_mapper = CompanyNameMapper()

def initialize_company_mapping(db_connection) -> None:
    """从数据库加载公司名称映射.
    
    Args:
        db_connection: SQLite数据库连接
    """
    global _COMPANY_MAPPING_INITIALIZED
    if _COMPANY_MAPPING_INITIALIZED:
        return
    _COMPANY_MAPPING_INITIALIZED = True

    cursor = db_connection.cursor()
    try:
        cursor.execute(
            """
            SELECT ts_code, name, short_name
            FROM stock_company
            WHERE name IS NOT NULL
            """
        )
    except sqlite3.OperationalError as exc:  # pragma: no cover - defensive
        LOGGER.debug(
            "stock_company 表不存在，跳过公司映射初始化 err=%s",
            exc,
            extra={"stage": "entity_recognition"},
        )
        cursor.close()
        return

    for ts_code, name, short_name in cursor.fetchall():
        if name and short_name:
            company_mapper.add_company(ts_code, name, short_name)

    cursor.close()
