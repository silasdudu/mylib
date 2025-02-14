"""
重排序模块，对检索结果进行二次排序
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from .retriever import SearchResult


class RerankerConfig(BaseModel):
    """重排序配置"""
    score_threshold: float = 0.0
    max_results: int = 10
    extra_params: Dict[str, Any] = {}


class Reranker(ABC):
    """重排序器抽象基类"""
    
    def __init__(self, config: RerankerConfig):
        self.config = config
    
    @abstractmethod
    async def rerank(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """重新排序结果"""
        pass
    
    @abstractmethod
    async def compute_score(
        self,
        query: str,
        result: SearchResult
    ) -> float:
        """计算相关性分数"""
        pass


class CrossEncoderReranker(Reranker):
    """基于交叉编码器的重排序器"""
    
    async def rerank(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """使用交叉编码器重新排序"""
        # 为每个结果计算新的分数
        scored_results = []
        for result in results:
            score = await self.compute_score(query, result)
            if score >= self.config.score_threshold:
                # 创建新的SearchResult，更新分数
                scored_result = SearchResult(
                    chunk=result.chunk,
                    score=score,
                    metadata={
                        **(result.metadata or {}),
                        "original_score": result.score
                    }
                )
                scored_results.append(scored_result)
                
        # 按新分数排序
        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results[:self.config.max_results]
    
    async def compute_score(
        self,
        query: str,
        result: SearchResult
    ) -> float:
        """计算查询和文档的相关性分数"""
        # 实现具体的分数计算逻辑
        raise NotImplementedError()


class EnsembleReranker(Reranker):
    """集成重排序器"""
    
    def __init__(
        self,
        config: RerankerConfig,
        rerankers: List[Reranker],
        weights: Optional[List[float]] = None
    ):
        super().__init__(config)
        self.rerankers = rerankers
        self.weights = weights or [1.0] * len(rerankers)
        
        if len(self.weights) != len(self.rerankers):
            raise ValueError("权重数量必须与重排序器数量相同")
    
    async def rerank(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """组合多个重排序器的结果"""
        # 获取每个重排序器的结果
        all_results = []
        for reranker, weight in zip(self.rerankers, self.weights):
            reranked = await reranker.rerank(query, results)
            for result in reranked:
                # 更新分数，加权组合
                result.score *= weight
            all_results.extend(reranked)
            
        # 对于相同的chunk，合并并累加分数
        merged_results = {}
        for result in all_results:
            chunk_id = result.chunk.metadata.chunk_id
            if chunk_id in merged_results:
                merged_results[chunk_id].score += result.score
            else:
                merged_results[chunk_id] = result
                
        # 转换回列表并排序
        final_results = list(merged_results.values())
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        return final_results[:self.config.max_results]
    
    async def compute_score(
        self,
        query: str,
        result: SearchResult
    ) -> float:
        """组合多个重排序器的分数"""
        scores = []
        for reranker, weight in zip(self.rerankers, self.weights):
            score = await reranker.compute_score(query, result)
            scores.append(score * weight)
        return sum(scores)


class RerankerRegistry:
    """重排序器注册表"""
    
    def __init__(self):
        self._rerankers: Dict[str, Reranker] = {}
        
    def register(
        self,
        name: str,
        reranker: Reranker
    ) -> None:
        """注册重排序器"""
        self._rerankers[name] = reranker
        
    def get_reranker(
        self,
        name: str
    ) -> Optional[Reranker]:
        """获取重排序器"""
        return self._rerankers.get(name)
        
    def list_rerankers(self) -> List[str]:
        """列出所有重排序器"""
        return list(self._rerankers.keys()) 