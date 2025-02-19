"""
KNN 检索器实现，基于 K 近邻算法的文本检索
"""
from typing import List, Optional, Dict, Any, Union
import numpy as np
from sklearn.neighbors import NearestNeighbors
from base.rag.retriever import Retriever, RetrieverConfig, SearchResult
from base.rag.chunking import Chunk
from base.model.embedding import EmbeddingOutput


class KNNRetrieverConfig(RetrieverConfig):
    """KNN 检索器配置"""
    n_neighbors: int = 5  # 近邻数量
    algorithm: str = "auto"  # KNN算法：auto, ball_tree, kd_tree, brute
    metric: str = "cosine"  # 距离度量：cosine, euclidean, manhattan
    leaf_size: int = 30  # 叶子节点大小，用于 ball_tree 和 kd_tree
    normalize_vectors: bool = True  # 是否对向量进行归一化
    cache_vectors: bool = True  # 是否缓存向量


class KNNRetriever(Retriever):
    """基于 K 近邻算法的检索器实现"""
    
    def __init__(self, config: KNNRetrieverConfig, **kwargs):
        """初始化 KNN 检索器
        
        Args:
            config: 检索器配置
            **kwargs: 额外的参数，传递给父类
        """
        super().__init__(config, **kwargs)
        self.config: KNNRetrieverConfig = config
        
        # 初始化 KNN 模型
        self._knn = NearestNeighbors(
            n_neighbors=config.n_neighbors,
            algorithm=config.algorithm,
            metric=config.metric,
            leaf_size=config.leaf_size,
            n_jobs=-1  # 使用所有CPU核心
        )
        
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
    
    def _convert_distance_to_score(self, distances: np.ndarray) -> np.ndarray:
        """将距离转换为相似度分数
        
        Args:
            distances: 距离数组
            
        Returns:
            相似度分数数组
        """
        if self.config.metric == "cosine":
            # 余弦距离已经是1-相似度，直接转换
            return 1 - distances
        else:
            # 对于其他距离度量，使用指数衰减转换为相似度
            return np.exp(-distances)
    
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
        
        # 训练 KNN 模型
        self._knn.fit(self._vectors)
        
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
        if self._vectors is None:
            raise RuntimeError("尚未建立索引")
            
        if not self.embedding_model:
            raise ValueError("未设置embedding_model")
            
        # 生成查询向量
        query_embedding = await self.embedding_model.embed(query)
        query_vector = self._validate_vectors([query_embedding.vector])[0]
        
        # 查找最近邻
        k = min(top_k or self.config.top_k, len(self._chunk_ids))
        distances, indices = self._knn.kneighbors(
            query_vector.reshape(1, -1),
            n_neighbors=k
        )
        
        # 将距离转换为分数
        scores = self._convert_distance_to_score(distances[0])
        
        # 构建结果
        results = []
        for idx, score in zip(indices[0], scores):
            chunk_id = self._chunk_ids[idx]
            chunk = self._chunks[chunk_id]
            results.append(SearchResult(
                chunk=chunk,
                score=float(score)
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
            
            # 重新训练 KNN 模型
            if len(self._chunk_ids) > 0:
                self._knn.fit(self._vectors) 