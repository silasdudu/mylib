"""
在线搜索引擎模块，提供统一的搜索接口
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


@dataclass
class SearchResult:
    """搜索结果"""
    title: str
    url: str
    snippet: str
    source: str
    timestamp: str
    metadata: Dict[str, Any] = None


class SearchConfig(BaseModel):
    """搜索配置"""
    max_results: int = 10
    language: str = "zh-CN"
    region: str = "CN"
    safe_search: bool = True
    time_range: Optional[str] = None  # 如"1d", "1w", "1m"
    extra_params: Dict[str, Any] = {}


class SearchEngine(ABC):
    """搜索引擎抽象基类"""
    
    def __init__(self, config: SearchConfig):
        self.config = config
    
    @abstractmethod
    async def search(
        self,
        query: str,
        **kwargs
    ) -> List[SearchResult]:
        """执行搜索"""
        pass
    
    @abstractmethod
    async def get_content(
        self,
        url: str
    ) -> str:
        """获取网页内容"""
        pass


class SearchAggregator:
    """搜索聚合器，组合多个搜索引擎的结果"""
    
    def __init__(self, engines: Dict[str, SearchEngine]):
        self.engines = engines
    
    async def search(
        self,
        query: str,
        weights: Optional[Dict[str, float]] = None
    ) -> List[SearchResult]:
        """聚合多个搜索引擎的结果"""
        all_results = []
        weights = weights or {name: 1.0 for name in self.engines}
        
        for name, engine in self.engines.items():
            if name in weights and weights[name] > 0:
                results = await engine.search(query)
                for result in results:
                    result.metadata = result.metadata or {}
                    result.metadata["engine"] = name
                    result.metadata["weight"] = weights[name]
                all_results.extend(results)
        
        # 按权重排序
        all_results.sort(
            key=lambda x: x.metadata["weight"],
            reverse=True
        )
        
        return all_results


class ContentProcessor(ABC):
    """内容处理器抽象基类"""
    
    @abstractmethod
    async def process(
        self,
        content: str,
        url: str
    ) -> str:
        """处理网页内容"""
        pass
    
    @abstractmethod
    async def extract_main_content(
        self,
        html: str
    ) -> str:
        """提取主要内容"""
        pass
    
    @abstractmethod
    async def clean_content(
        self,
        content: str
    ) -> str:
        """清理内容"""
        pass 