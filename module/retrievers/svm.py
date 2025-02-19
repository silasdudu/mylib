"""
SVM 检索器实现，基于支持向量机的文本检索
"""
from typing import List, Optional, Dict, Any, Union
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from sklearn.exceptions import NotFittedError
from base.rag.retriever import Retriever, RetrieverConfig, SearchResult
from base.rag.chunking import Chunk
from base.model.embedding import EmbeddingOutput


class SVMRetrieverConfig(RetrieverConfig):
    """SVM 检索器配置"""
    c: float = 1.0  # 正则化参数
    max_iter: int = 1000  # 最大迭代次数
    class_weight: str = "balanced"  # 类别权重：None, balanced
    dual: bool = True  # 是否使用对偶优化问题
    normalize_vectors: bool = True  # 是否对向量进行归一化
    cache_vectors: bool = True  # 是否缓存向量
    threshold: float = 0.0  # 相似度阈值，低于此值的结果将被过滤
    positive_threshold: float = 0.5  # 正样本阈值，用于生成伪标签


class SVMRetriever(Retriever):
    """基于支持向量机的检索器实现"""
    
    def __init__(self, config: SVMRetrieverConfig, **kwargs):
        """初始化 SVM 检索器
        
        Args:
            config: 检索器配置
            **kwargs: 额外的参数，传递给父类
        """
        super().__init__(config, **kwargs)
        self.config: SVMRetrieverConfig = config
        
        # 初始化 SVM 模型
        self._svm = LinearSVC(
            C=config.c,
            max_iter=config.max_iter,
            class_weight=config.class_weight,
            dual=config.dual
        )
        
        # 初始化存储
        self._vectors = None  # 向量数组
        self._chunks = {}  # 文档块字典
        self._chunk_ids = []  # chunk_id列表
        self._is_fitted = False  # 模型是否已训练
    
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
            vectors_array = normalize(vectors_array, norm='l2', axis=1)
            
        return vectors_array
    
    def _generate_pseudo_labels(
        self,
        query_vector: np.ndarray,
        index_vectors: np.ndarray
    ) -> np.ndarray:
        """生成伪标签用于训练 SVM
        
        Args:
            query_vector: 查询向量
            index_vectors: 索引向量数组
            
        Returns:
            伪标签数组
        """
        # 计算余弦相似度
        similarities = np.dot(index_vectors, query_vector)
        if self.config.normalize_vectors:
            similarities = similarities / (
                np.linalg.norm(query_vector) *
                np.linalg.norm(index_vectors, axis=1)
            )
        
        # 根据相似度阈值生成伪标签
        labels = np.zeros(len(index_vectors), dtype=int)
        labels[similarities > self.config.positive_threshold] = 1
        
        return labels
    
    def _convert_decision_to_score(self, decisions: np.ndarray) -> np.ndarray:
        """将 SVM 决策值转换为相似度分数
        
        Args:
            decisions: SVM 决策值数组
            
        Returns:
            相似度分数数组
        """
        # 使用 sigmoid 函数将决策值映射到 [0, 1] 区间
        return 1 / (1 + np.exp(-decisions))
    
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
        
        # 重置模型状态
        self._is_fitted = False
    
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
        
        # 生成伪标签并训练 SVM
        labels = self._generate_pseudo_labels(query_vector, self._vectors)
        
        # 如果没有正样本，返回空结果
        if sum(labels) == 0:
            return []
            
        # 训练 SVM
        try:
            self._svm.fit(self._vectors, labels)
            self._is_fitted = True
        except Exception as e:
            raise RuntimeError(f"SVM训练失败: {str(e)}")
        
        # 获取决策值
        try:
            decisions = self._svm.decision_function(self._vectors)
        except NotFittedError:
            raise RuntimeError("SVM模型尚未训练")
            
        # 将决策值转换为分数
        scores = self._convert_decision_to_score(decisions)
        
        # 过滤低于阈值的结果
        mask = scores > self.config.threshold
        if not np.any(mask):
            return []
            
        # 获取top_k结果
        k = min(top_k or self.config.top_k, np.sum(mask))
        top_indices = np.argsort(scores[mask])[-k:][::-1]
        
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
            
            # 重置模型状态
            self._is_fitted = False 