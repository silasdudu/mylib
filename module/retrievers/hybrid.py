"""
混合检索器实现
"""
from typing import List, Optional, Dict, Any, Union, Tuple
import numpy as np
from base.rag.retriever import Retriever, RetrieverConfig, SearchResult
from base.rag.chunking import Chunk
from base.model.embedding import EmbeddingOutput


class HybridRetrieverConfig(RetrieverConfig):
    """混合检索器配置"""
    retrievers: List[Tuple[Retriever, float]] = []  # 检索器列表及其权重
    fusion_method: str = "weighted_sum"  # 融合方法：weighted_sum, rrf, max_score
    rrf_k: int = 60  # RRF方法的k参数
    normalize_scores: bool = True  # 是否对分数进行归一化
    dedup_results: bool = True  # 是否去重结果


class HybridRetriever(Retriever):
    """混合检索器实现，集成多个检索器"""
    
    def __init__(self, config: HybridRetrieverConfig, **kwargs):
        """初始化混合检索器
        
        Args:
            config: 检索器配置
            **kwargs: 额外的参数，传递给父类
        """
        super().__init__(config, **kwargs)
        self.config: HybridRetrieverConfig = config
        
        if not self.config.retrievers:
            raise ValueError("必须提供至少一个检索器")
            
        # 验证权重和是否为1
        total_weight = sum(weight for _, weight in self.config.retrievers)
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"检索器权重之和必须为1，当前为: {total_weight}")
    
    def _normalize_scores(self, results: List[SearchResult]) -> List[SearchResult]:
        """归一化检索结果的分数
        
        Args:
            results: 检索结果列表
            
        Returns:
            归一化后的检索结果列表
        """
        if not results:
            return results
            
        # 提取分数
        scores = np.array([r.score for r in results])
        
        # Min-Max归一化
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score > min_score:
            normalized_scores = (scores - min_score) / (max_score - min_score)
            
            # 更新结果分数
            for result, score in zip(results, normalized_scores):
                result.score = float(score)
                
        return results
    
    def _weighted_sum_fusion(
        self,
        all_results: List[List[SearchResult]],
        weights: List[float]
    ) -> List[SearchResult]:
        """加权求和融合方法
        
        Args:
            all_results: 所有检索器的结果列表
            weights: 对应的权重列表
            
        Returns:
            融合后的结果列表
        """
        # 构建chunk_id到分数的映射
        chunk_scores: Dict[str, float] = {}
        chunk_map: Dict[str, SearchResult] = {}
        
        # 对每个检索器的结果进行加权
        for results, weight in zip(all_results, weights):
            for result in results:
                chunk_id = result.chunk.metadata.chunk_id
                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = 0
                    chunk_map[chunk_id] = result
                chunk_scores[chunk_id] += result.score * weight
        
        # 构建最终结果
        final_results = []
        for chunk_id, score in chunk_scores.items():
            result = chunk_map[chunk_id]
            result.score = score
            final_results.append(result)
        
        # 按分数排序
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        return final_results
    
    def _reciprocal_rank_fusion(
        self,
        all_results: List[List[SearchResult]]
    ) -> List[SearchResult]:
        """倒数排名融合方法
        
        Args:
            all_results: 所有检索器的结果列表
            
        Returns:
            融合后的结果列表
        """
        # 构建chunk_id到RRF分数的映射
        chunk_scores: Dict[str, float] = {}
        chunk_map: Dict[str, SearchResult] = {}
        
        # 计算每个结果的RRF分数
        k = self.config.rrf_k
        for results in all_results:
            for rank, result in enumerate(results, 1):
                chunk_id = result.chunk.metadata.chunk_id
                if chunk_id not in chunk_scores:
                    chunk_scores[chunk_id] = 0
                    chunk_map[chunk_id] = result
                # RRF公式: 1 / (k + r)，其中r是排名
                chunk_scores[chunk_id] += 1 / (k + rank)
        
        # 构建最终结果
        final_results = []
        for chunk_id, score in chunk_scores.items():
            result = chunk_map[chunk_id]
            result.score = score
            final_results.append(result)
        
        # 按分数排序
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        return final_results
    
    def _max_score_fusion(
        self,
        all_results: List[List[SearchResult]]
    ) -> List[SearchResult]:
        """最大分数融合方法
        
        Args:
            all_results: 所有检索器的结果列表
            
        Returns:
            融合后的结果列表
        """
        # 构建chunk_id到最大分数的映射
        chunk_scores: Dict[str, float] = {}
        chunk_map: Dict[str, SearchResult] = {}
        
        # 找出每个chunk的最高分数
        for results in all_results:
            for result in results:
                chunk_id = result.chunk.metadata.chunk_id
                if chunk_id not in chunk_scores or result.score > chunk_scores[chunk_id]:
                    chunk_scores[chunk_id] = result.score
                    chunk_map[chunk_id] = result
        
        # 构建最终结果
        final_results = list(chunk_map.values())
        
        # 按分数排序
        final_results.sort(key=lambda x: x.score, reverse=True)
        
        return final_results
    
    async def index(
        self,
        chunks: List[Chunk],
        embeddings: Optional[List[EmbeddingOutput]] = None
    ) -> None:
        """索引文档分块
        
        Args:
            chunks: 文档分块列表
            embeddings: 可选的向量表示列表
        """
        # 为每个检索器建立索引
        for retriever, _ in self.config.retrievers:
            await retriever.index(chunks, embeddings)
    
    async def search(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """搜索相关内容
        
        Args:
            query: 查询文本
            top_k: 返回结果数量
            
        Returns:
            检索结果列表
        """
        # 获取所有检索器的结果
        all_results = []
        weights = []
        
        for retriever, weight in self.config.retrievers:
            results = await retriever.search(query, top_k)
            
            # 如果配置要求，对每个检索器的结果进行归一化
            if self.config.normalize_scores:
                results = self._normalize_scores(results)
                
            all_results.append(results)
            weights.append(weight)
        
        # 根据选择的融合方法合并结果
        if self.config.fusion_method == "weighted_sum":
            final_results = self._weighted_sum_fusion(all_results, weights)
        elif self.config.fusion_method == "rrf":
            final_results = self._reciprocal_rank_fusion(all_results)
        elif self.config.fusion_method == "max_score":
            final_results = self._max_score_fusion(all_results)
        else:
            raise ValueError(f"不支持的融合方法: {self.config.fusion_method}")
        
        # 如果需要去重
        if self.config.dedup_results:
            seen_chunks = set()
            deduped_results = []
            for result in final_results:
                chunk_id = result.chunk.metadata.chunk_id
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    deduped_results.append(result)
            final_results = deduped_results
        
        # 限制返回数量
        k = top_k or self.config.top_k
        return final_results[:k]
    
    async def delete(
        self,
        chunk_ids: List[str]
    ) -> None:
        """删除索引
        
        Args:
            chunk_ids: 要删除的chunk_id列表
        """
        # 在所有检索器中删除索引
        for retriever, _ in self.config.retrievers:
            await retriever.delete(chunk_ids) 