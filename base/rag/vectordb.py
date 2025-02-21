"""
向量数据库基类，提供统一的接口定义和基础功能
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
import numpy as np
from pydantic import BaseModel

from .chunking import Chunk
from ..core.logging import AsyncLogger, LogLevel
from ..model.embedding import EmbeddingOutput


class VectorDBConfig(BaseModel):
    """向量数据库配置"""
    dimension: int  # 向量维度
    top_k: int = 5  # 默认返回的最相似结果数量
    distance_metric: str = "cosine"  # 距离度量方式：cosine, euclidean, dot_product
    batch_size: int = 32  # 批处理大小
    extra_params: Dict[str, Any] = {}  # 额外的特定实现参数


class SearchResult(BaseModel):
    """搜索结果"""
    chunk: Chunk  # 文档块
    score: float  # 相似度分数
    metadata: Dict[str, Any] = {}  # 额外元数据


class VectorDB(ABC):
    """向量数据库抽象基类"""
    
    def __init__(
        self,
        config: VectorDBConfig,
        logger: Optional[AsyncLogger] = None
    ):
        """初始化向量数据库
        
        Args:
            config: 数据库配置
            logger: 可选的日志记录器
        """
        self.config = config
        self.logger = logger
        self._index_ready = False
    
    async def _log(self, level: LogLevel, message: str):
        """内部日志记录方法"""
        if self.logger:
            await self.logger.log(level, message)
    
    def _validate_vectors(self, vectors: Union[List[List[float]], np.ndarray]) -> np.ndarray:
        """验证并标准化向量格式
        
        Args:
            vectors: 输入向量，可以是列表或numpy数组
            
        Returns:
            标准化后的numpy数组
            
        Raises:
            ValueError: 当向量维度不匹配时抛出
        """
        if isinstance(vectors, list):
            vectors = np.array(vectors)
        
        if len(vectors.shape) != 2:
            raise ValueError(f"向量必须是二维数组，当前维度: {vectors.shape}")
            
        if vectors.shape[1] != self.config.dimension:
            raise ValueError(
                f"向量维度不匹配，期望维度: {self.config.dimension}，"
                f"实际维度: {vectors.shape[1]}"
            )
            
        return vectors
    
    def _compute_similarity(
        self,
        query_vector: np.ndarray,
        index_vectors: np.ndarray
    ) -> np.ndarray:
        """计算查询向量与索引向量的相似度
        
        Args:
            query_vector: 查询向量
            index_vectors: 索引向量
            
        Returns:
            相似度分数数组
        """
        if self.config.distance_metric == "cosine":
            # 计算余弦相似度
            norm_query = np.linalg.norm(query_vector)
            norm_index = np.linalg.norm(index_vectors, axis=1)
            return np.dot(index_vectors, query_vector) / (norm_query * norm_index)
        elif self.config.distance_metric == "euclidean":
            # 计算欧氏距离（转换为相似度）
            distances = np.linalg.norm(index_vectors - query_vector, axis=1)
            return 1 / (1 + distances)
        elif self.config.distance_metric == "dot_product":
            # 计算点积
            return np.dot(index_vectors, query_vector)
        else:
            raise ValueError(f"不支持的距离度量方式: {self.config.distance_metric}")
    
    @abstractmethod
    async def create_index(self, vectors: List[List[float]], chunks: List[Chunk]) -> None:
        """创建向量索引
        
        Args:
            vectors: 向量列表
            chunks: 对应的文档块列表
        """
        pass
    
    @abstractmethod
    async def add_vectors(self, vectors: List[List[float]], chunks: List[Chunk]) -> None:
        """添加向量到索引
        
        Args:
            vectors: 向量列表
            chunks: 对应的文档块列表
        """
        pass
    
    @abstractmethod
    async def delete_vectors(self, chunk_ids: List[str]) -> None:
        """从索引中删除向量
        
        Args:
            chunk_ids: 要删除的文档块ID列表
        """
        pass
    
    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        top_k: Optional[int] = None,
        filter_params: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """搜索最相似的向量
        
        Args:
            query_vector: 查询向量
            top_k: 返回结果数量，如果不指定则使用配置中的值
            filter_params: 可选的过滤参数
            
        Returns:
            搜索结果列表
        """
        pass
    
    @abstractmethod
    async def batch_search(
        self,
        query_vectors: List[List[float]],
        top_k: Optional[int] = None,
        filter_params: Optional[Dict[str, Any]] = None
    ) -> List[List[SearchResult]]:
        """批量搜索最相似的向量
        
        Args:
            query_vectors: 查询向量列表
            top_k: 每个查询返回的结果数量
            filter_params: 可选的过滤参数
            
        Returns:
            每个查询对应的搜索结果列表
        """
        pass
    
    @abstractmethod
    async def save(self, path: str) -> None:
        """保存索引到文件
        
        Args:
            path: 保存路径
        """
        pass
    
    @abstractmethod
    async def load(self, path: str) -> None:
        """从文件加载索引
        
        Args:
            path: 索引文件路径
        """
        pass
    
    @property
    def is_ready(self) -> bool:
        """索引是否已准备就绪"""
        return self._index_ready