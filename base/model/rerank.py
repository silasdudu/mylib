"""
重排序模型模块，用于对检索结果进行重新排序
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel

from ..rag.retriever import SearchResult


class RerankConfig(BaseModel):
    """重排序模型配置"""
    model_name: str
    batch_size: int = 32
    max_length: int = 512
    score_type: str = "normalized"  # normalized, raw
    extra_params: Dict[str, Any] = {}


class RerankOutput(BaseModel):
    """重排序输出"""
    score: float  # 相似度分数
    metadata: Dict[str, Any] = {}  # 额外元数据


class Reranker(ABC):
    """重排序模型抽象基类"""
    
    def __init__(self, config: RerankConfig):
        self.config = config
    
    @abstractmethod
    async def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None
    ) -> List[RerankOutput]:
        """对文档列表进行重新排序
        
        Args:
            query: 查询文本
            documents: 待排序的文档列表
            top_k: 返回的结果数量
            
        Returns:
            重排序后的结果列表
        """
        pass
    
    @abstractmethod
    async def rerank_search_results(
        self,
        query: str,
        results: List[SearchResult],
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """对搜索结果进行重新排序
        
        Args:
            query: 查询文本
            results: 搜索结果列表
            top_k: 返回的结果数量
            
        Returns:
            重排序后的搜索结果列表
        """
        pass
    
    @abstractmethod
    async def compute_similarity(
        self,
        text1: Union[str, List[str]],
        text2: Union[str, List[str]]
    ) -> Union[float, List[float]]:
        """计算文本相似度
        
        Args:
            text1: 第一个文本或文本列表
            text2: 第二个文本或文本列表
            
        Returns:
            相似度分数或分数列表
        """
        pass
    
    def normalize_score(self, score: float) -> float:
        """归一化分数到 [0, 1] 区间
        
        Args:
            score: 原始分数
            
        Returns:
            归一化后的分数
        """
        if self.config.score_type == "normalized":
            # 对于不同的模型可能需要不同的归一化方法
            # 这里使用简单的 sigmoid 函数
            import math
            return 1 / (1 + math.exp(-score))
        return score 