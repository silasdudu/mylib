"""
检索器模块，支持从向量库中检索相关内容
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from .chunking import Chunk
from ..model.embedding import EmbeddingModel, EmbeddingOutput


class RetrieverConfig(BaseModel):
    """检索器配置基类"""
    extra_params: Dict[str, Any] = {}  # 额外的特定实现参数


@dataclass
class SearchResult:
    """检索结果"""
    chunk: Chunk
    score: float
    metadata: Dict[str, Any] = None


class Retriever(ABC):
    """检索器抽象基类"""
    
    def __init__(
        self,
        config: RetrieverConfig,
        embedding_model: Optional[EmbeddingModel] = None
    ):
        self.config = config
        self.embedding_model = embedding_model
    
    @abstractmethod
    async def index(
        self,
        chunks: List[Chunk],
        embeddings: Optional[List[EmbeddingOutput]] = None
    ) -> None:
        """索引文档分块"""
        pass
    
    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """搜索相关内容"""
        pass
    
    @abstractmethod
    async def delete(
        self,
        chunk_ids: List[str]
    ) -> None:
        """删除索引"""
        pass


class RetrieverRegistry:
    """检索器注册表"""
    
    def __init__(self):
        self._retrievers: Dict[str, Retriever] = {}
        
    def register(
        self,
        name: str,
        retriever: Retriever
    ) -> None:
        """注册检索器"""
        self._retrievers[name] = retriever
        
    def get_retriever(
        self,
        name: str
    ) -> Optional[Retriever]:
        """获取检索器"""
        return self._retrievers.get(name)
        
    def list_retrievers(self) -> List[str]:
        """列出所有检索器"""
        return list(self._retrievers.keys()) 