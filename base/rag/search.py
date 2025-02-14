"""
搜索引擎模块，提供高级搜索和匹配优化功能
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from .retriever import Retriever, SearchResult


class SearchConfig(BaseModel):
    """搜索配置"""
    min_score: float = 0.0
    max_results: int = 10
    use_reranking: bool = False
    extra_params: Dict[str, Any] = {}


@dataclass
class SearchQuery:
    """搜索查询"""
    text: str
    filters: Dict[str, Any] = None
    metadata: Dict[str, Any] = None


class SearchEngine(ABC):
    """搜索引擎抽象基类"""
    
    def __init__(
        self,
        config: SearchConfig,
        retriever: Retriever
    ):
        self.config = config
        self.retriever = retriever
    
    @abstractmethod
    async def search(
        self,
        query: Union[str, SearchQuery],
        **kwargs
    ) -> List[SearchResult]:
        """执行搜索"""
        pass
    
    @abstractmethod
    async def filter_results(
        self,
        results: List[SearchResult],
        filters: Dict[str, Any]
    ) -> List[SearchResult]:
        """过滤搜索结果"""
        pass
    
    @abstractmethod
    async def rerank_results(
        self,
        query: Union[str, SearchQuery],
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """重新排序结果"""
        pass


class BasicSearchEngine(SearchEngine):
    """基础搜索引擎"""
    
    async def search(
        self,
        query: Union[str, SearchQuery],
        **kwargs
    ) -> List[SearchResult]:
        """执行基础搜索"""
        # 处理查询
        if isinstance(query, str):
            query = SearchQuery(text=query)
            
        # 获取检索结果
        results = await self.retriever.search(
            query.text,
            top_k=self.config.max_results
        )
        
        # 应用过滤器
        if query.filters:
            results = await self.filter_results(results, query.filters)
            
        # 重新排序
        if self.config.use_reranking:
            results = await self.rerank_results(query, results)
            
        # 应用分数阈值
        results = [
            r for r in results
            if r.score >= self.config.min_score
        ]
        
        return results[:self.config.max_results]
    
    async def filter_results(
        self,
        results: List[SearchResult],
        filters: Dict[str, Any]
    ) -> List[SearchResult]:
        """基于元数据过滤结果"""
        filtered = []
        
        for result in results:
            match = True
            for key, value in filters.items():
                if key in result.metadata:
                    if isinstance(value, (list, tuple, set)):
                        if result.metadata[key] not in value:
                            match = False
                            break
                    elif result.metadata[key] != value:
                        match = False
                        break
                else:
                    match = False
                    break
                    
            if match:
                filtered.append(result)
                
        return filtered
    
    async def rerank_results(
        self,
        query: Union[str, SearchQuery],
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """简单的重新排序"""
        # 这里可以实现更复杂的重排序逻辑
        return sorted(
            results,
            key=lambda x: x.score,
            reverse=True
        )


class SearchEngineRegistry:
    """搜索引擎注册表"""
    
    def __init__(self):
        self._engines: Dict[str, SearchEngine] = {}
        
    def register(
        self,
        name: str,
        engine: SearchEngine
    ) -> None:
        """注册搜索引擎"""
        self._engines[name] = engine
        
    def get_engine(
        self,
        name: str
    ) -> Optional[SearchEngine]:
        """获取搜索引擎"""
        return self._engines.get(name)
        
    def list_engines(self) -> List[str]:
        """列出所有搜索引擎"""
        return list(self._engines.keys()) 