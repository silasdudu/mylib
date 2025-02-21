"""
搜索引擎模块，提供统一的搜索接口
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union, Optional
from pydantic import BaseModel

from .retriever import Retriever, SearchResult
from .reranker import Reranker


class SearchConfig(BaseModel):
    """搜索引擎配置基类"""
    extra_params: Dict[str, Any] = {}  # 额外的特定实现参数


class SearchQuery(BaseModel):
    """搜索查询"""
    text: str
    filters: Optional[Dict[str, Any]] = None
    extra_params: Dict[str, Any] = {}


class SearchEngine(ABC):
    """搜索引擎抽象基类"""
    
    def __init__(
        self,
        config: SearchConfig,
        retriever: Retriever,
        reranker: Optional[Reranker] = None
    ):
        self.config = config
        self.retriever = retriever
        self.reranker = reranker
    
    @abstractmethod
    async def search(
        self,
        query: Union[str, SearchQuery],
        **kwargs
    ) -> List[SearchResult]:
        """执行搜索
        
        Args:
            query: 查询文本或查询对象
            **kwargs: 额外的搜索参数
            
        Returns:
            搜索结果列表
        """
        pass
    
    async def rerank_results(
        self,
        query: SearchQuery,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """重新排序结果
        
        Args:
            query: 查询对象
            results: 检索结果列表
            
        Returns:
            重新排序后的结果列表
        """
        if not self.reranker:
            return results
        return await self.reranker.rerank(query.text, results)
    
    @abstractmethod
    async def filter_results(
        self,
        results: List[SearchResult],
        filters: Dict[str, Any]
    ) -> List[SearchResult]:
        """过滤结果
        
        Args:
            results: 搜索结果列表
            filters: 过滤条件
            
        Returns:
            过滤后的结果列表
        """
        pass


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