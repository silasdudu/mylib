"""
搜索引擎包装器模块，提供各种搜索引擎的统一接口实现
"""
from typing import List, Optional

from base.search.engine import SearchEngine, SearchResult
from module.search.basic import BasicSearchEngine, BasicSearchConfig
from module.search.bing import BingSearch, BingSearchConfig
from module.search.google import GoogleSearch, GoogleSearchConfig
from module.search.brave import BraveSearch, BraveSearchConfig
from module.search.tavily import TavilySearch, TavilySearchConfig


class BingSearchWrapper(SearchEngine):
    """必应搜索引擎包装器"""
    
    def __init__(self, config: Optional[BingSearchConfig] = None):
        self.search_engine = BingSearch(config)
    
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        """搜索相关信息"""
        return await self.search_engine.search(query, **kwargs)
    
    async def get_content(self, url: str) -> str:
        """获取网页内容"""
        return await self.search_engine.get_content(url)


class GoogleSearchWrapper(SearchEngine):
    """谷歌搜索引擎包装器"""
    
    def __init__(self, config: Optional[GoogleSearchConfig] = None):
        self.search_engine = GoogleSearch(config)
    
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        """搜索相关信息"""
        return await self.search_engine.search(query, **kwargs)
    
    async def get_content(self, url: str) -> str:
        """获取网页内容"""
        return await self.search_engine.get_content(url)


class BraveSearchWrapper(SearchEngine):
    """Brave搜索引擎包装器"""
    
    def __init__(self, config: Optional[BraveSearchConfig] = None):
        self.search_engine = BraveSearch(config)
    
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        """搜索相关信息"""
        return await self.search_engine.search(query, **kwargs)
    
    async def get_content(self, url: str) -> str:
        """获取网页内容"""
        return await self.search_engine.get_content(url)


class TavilySearchWrapper(SearchEngine):
    """Tavily搜索引擎包装器"""
    
    def __init__(self, config: Optional[TavilySearchConfig] = None):
        self.search_engine = TavilySearch(config)
    
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        """搜索相关信息"""
        return await self.search_engine.search(query, **kwargs)
    
    async def get_content(self, url: str) -> str:
        """获取网页内容"""
        return await self.search_engine.get_content(url)


class BasicSearchWrapper(SearchEngine):
    """基础搜索引擎包装器"""
    
    def __init__(self, config: Optional[BasicSearchConfig] = None):
        self.search_engine = BasicSearchEngine(config)
    
    async def search(self, query: str, **kwargs) -> List[SearchResult]:
        """搜索相关信息"""
        return await self.search_engine.search(query, **kwargs)
    
    async def get_content(self, url: str) -> str:
        """获取网页内容"""
        return await self.search_engine.get_content(url) 