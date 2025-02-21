"""
交叉编码器重排序器实现
"""
from typing import List, Dict, Any
from pydantic import Field
from base.rag.reranker import Reranker, RerankerConfig, SearchResult


class CrossEncoderConfig(RerankerConfig):
    """交叉编码器重排序器配置"""
    score_threshold: float = Field(default=0.0, description="分数阈值")
    max_results: int = Field(default=10, description="最大返回结果数")
    model_name: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", description="使用的交叉编码器模型名称")
    batch_size: int = Field(default=32, description="批处理大小")
    normalize_scores: bool = Field(default=True, description="是否对分数进行归一化")


class CrossEncoderReranker(Reranker):
    """基于交叉编码器的重排序器"""
    
    def __init__(self, config: CrossEncoderConfig):
        super().__init__(config)

    async def rerank(
        self,
        query: str,
        results: List[SearchResult]
    ) -> List[SearchResult]:
        """使用交叉编码器重新排序
        
        Args:
            query: 查询文本
            results: 检索结果列表
            
        Returns:
            重新排序后的结果列表
        """
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
        """计算查询和文档的相关性分数
        
        Args:
            query: 查询文本
            result: 检索结果
            
        Returns:
            相关性分数
            
        Raises:
            NotImplementedError: 需要在具体实现中重写此方法
        """
        raise NotImplementedError("需要在具体实现中实现compute_score方法") 