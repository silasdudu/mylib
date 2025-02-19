"""
向量检索器实现，基于向量相似度搜索
"""
from typing import List, Optional, Dict, Any
import numpy as np
from base.rag.retriever import Retriever, RetrieverConfig, SearchResult
from base.rag.chunking import Chunk
from base.model.embedding import EmbeddingOutput


class VectorRetrieverConfig(RetrieverConfig):
    """向量检索器配置"""
    distance_metric: str = "cosine"  # 距离度量方式：cosine, l2, dot_product
    normalize_vectors: bool = True  # 是否对向量进行归一化
    cache_vectors: bool = True  # 是否缓存向量


class VectorRetriever(Retriever):
    """基于向量相似度的检索器实现"""
    
    def __init__(self, config: VectorRetrieverConfig, **kwargs):
        """初始化向量检索器
        
        Args:
            config: 检索器配置
            **kwargs: 额外的参数，传递给父类
        """
        super().__init__(config, **kwargs)
        self.config: VectorRetrieverConfig = config
        
        # 初始化存储
        self._vectors = None  # 向量数组
        self._chunks = {}  # 文档块字典
        self._chunk_ids = []  # chunk_id列表
    
    def _validate_vectors(self, vectors: List[List[float]]) -> np.ndarray:
        """验证并标准化向量格式
        
        Args:
            vectors: 输入向量列表
            
        Returns:
            标准化后的numpy数组
        """
        vectors_array = np.array(vectors)
        if len(vectors_array.shape) != 2:
            raise ValueError(f"向量必须是二维数组，当前维度: {vectors_array.shape}")
            
        if self.config.normalize_vectors:
            # L2归一化
            norms = np.linalg.norm(vectors_array, axis=1, keepdims=True)
            vectors_array = vectors_array / np.maximum(norms, 1e-12)
            
        return vectors_array
    
    def _compute_similarity(
        self,
        query_vector: np.ndarray,
        index_vectors: np.ndarray
    ) -> np.ndarray:
        """计算查询向量与索引向量的相似度
        
        Args:
            query_vector: 查询向量
            index_vectors: 索引向量数组
            
        Returns:
            相似度分数数组
        """
        if self.config.distance_metric == "cosine":
            # 计算余弦相似度
            if self.config.normalize_vectors:
                # 如果向量已经归一化，直接计算点积
                return np.dot(index_vectors, query_vector)
            else:
                # 否则需要计算余弦相似度
                norm_query = np.linalg.norm(query_vector)
                norm_index = np.linalg.norm(index_vectors, axis=1)
                return np.dot(index_vectors, query_vector) / (norm_query * norm_index)
        elif self.config.distance_metric == "l2":
            # 计算欧氏距离（转换为相似度）
            distances = np.linalg.norm(index_vectors - query_vector, axis=1)
            return 1 / (1 + distances)
        elif self.config.distance_metric == "dot_product":
            # 计算点积
            return np.dot(index_vectors, query_vector)
        else:
            raise ValueError(f"不支持的距离度量方式: {self.config.distance_metric}")
    
    async def index(
        self,
        chunks: List[Chunk],
        embeddings: Optional[List[EmbeddingOutput]] = None
    ) -> None:
        """索引文档分块
        
        Args:
            chunks: 文档分块列表
            embeddings: 可选的向量表示列表，如果不提供则使用嵌入模型生成
        """
        if not embeddings and self.embedding_model:
            embeddings = await self.embedding_model.embed_chunks(chunks)
        elif not embeddings:
            raise ValueError("必须提供embeddings或embedding_model")
            
        # 提取向量
        vectors = [e.vector for e in embeddings]
        
        # 验证并标准化向量
        self._vectors = self._validate_vectors(vectors)
        
        # 存储文档块
        self._chunks = {chunk.metadata.chunk_id: chunk for chunk in chunks}
        self._chunk_ids = [chunk.metadata.chunk_id for chunk in chunks]
    
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
        if not self._vectors is not None:
            raise RuntimeError("尚未建立索引")
            
        if not self.embedding_model:
            raise ValueError("未设置embedding_model")
            
        # 生成查询向量
        query_embedding = await self.embedding_model.embed(query)
        query_vector = self._validate_vectors([query_embedding.vector])[0]
        
        # 计算相似度
        scores = self._compute_similarity(query_vector, self._vectors)
        
        # 获取top_k结果
        k = top_k or self.config.top_k
        top_indices = np.argsort(scores)[-k:][::-1]
        
        # 构建结果
        results = []
        for idx in top_indices:
            chunk_id = self._chunk_ids[idx]
            chunk = self._chunks[chunk_id]
            results.append(SearchResult(
                chunk=chunk,
                score=float(scores[idx])
            ))
            
        return results
    
    async def delete(
        self,
        chunk_ids: List[str]
    ) -> None:
        """删除索引
        
        Args:
            chunk_ids: 要删除的chunk_id列表
        """
        if self._vectors is None:
            return
            
        # 找出要删除的索引
        indices_to_delete = []
        for i, chunk_id in enumerate(self._chunk_ids):
            if chunk_id in chunk_ids:
                indices_to_delete.append(i)
                self._chunks.pop(chunk_id, None)
                
        # 更新向量数组和chunk_id列表
        if indices_to_delete:
            mask = np.ones(len(self._chunk_ids), dtype=bool)
            mask[indices_to_delete] = False
            self._vectors = self._vectors[mask]
            self._chunk_ids = [
                chunk_id for i, chunk_id in enumerate(self._chunk_ids)
                if i not in indices_to_delete
            ] 