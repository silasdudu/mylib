"""
混合RAG系统模块，支持文档检索和SQL查询的组合
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from .generator import GeneratorInput, GeneratorOutput
from .retriever import SearchResult
from .sql import SQLResult


@dataclass
class HybridContext:
    """混合上下文，包含检索结果和SQL查询结果"""
    search_results: List[SearchResult]
    sql_results: List[SQLResult]
    metadata: Dict[str, Any] = None


class HybridGeneratorInput(GeneratorInput):
    """扩展的生成器输入，支持SQL结果"""
    sql_results: List[SQLResult] = None


class HybridGenerator(ABC):
    """混合生成器抽象基类"""
    
    @abstractmethod
    async def generate(
        self,
        query: str,
        context: HybridContext
    ) -> GeneratorOutput:
        """根据混合上下文生成回答"""
        pass
    
    @abstractmethod
    async def should_use_sql(
        self,
        query: str
    ) -> bool:
        """判断是否需要使用SQL查询"""
        pass
    
    @abstractmethod
    async def format_context(
        self,
        context: HybridContext
    ) -> str:
        """格式化混合上下文"""
        pass


class QueryRouter(ABC):
    """查询路由器抽象基类"""
    
    @abstractmethod
    async def route_query(
        self,
        query: str
    ) -> Dict[str, float]:
        """为查询分配不同组件的权重"""
        pass
    
    @abstractmethod
    async def merge_results(
        self,
        search_results: List[SearchResult],
        sql_results: List[SQLResult]
    ) -> HybridContext:
        """合并检索结果和SQL结果"""
        pass


class HybridRAGConfig(BaseModel):
    """混合RAG系统配置"""
    use_sql: bool = True
    use_search: bool = True
    sql_threshold: float = 0.5  # SQL查询的触发阈值
    max_sql_queries: int = 3  # 单次回答最大SQL查询次数
    extra_params: Dict[str, Any] = {} 