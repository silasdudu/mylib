"""
集成重排序器实现
"""
from typing import List, Optional
from base.rag.reranker import Reranker, RerankConfig, SearchResult


class EnsembleReranker(Reranker):
    """集成重排序器"""
    
    def __init__(
        self,
        config: RerankConfig,
        rerankers: List[Reranker],
        weights: Optional[List[float]] = None
    ):
        """初始化集成重排序器
        
        Args:
            config: 重排序器配置
            rerankers: 重排序器列表
            weights: 可选的权重列表，如果不提供则使用均匀权重
        """
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
        """组合多个重排序器的结果
        
        Args:
            query: 查询文本
            results: 检索结果列表
            
        Returns:
            重新排序后的结果列表
        """
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