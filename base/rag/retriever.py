"""
检索器模块，支持从向量库中检索相关内容
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from .chunking import Chunk
from .embedding import EmbeddingModel, EmbeddingOutput


class RetrieverConfig(BaseModel):
    """检索器配置"""
    top_k: int = 5
    score_threshold: float = 0.0
    extra_params: Dict[str, Any] = {}


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


class VectorRetriever(Retriever):
    """向量检索器"""
    
    async def index(
        self,
        chunks: List[Chunk],
        embeddings: Optional[List[EmbeddingOutput]] = None
    ) -> None:
        """索引向量"""
        if not embeddings and self.embedding_model:
            embeddings = await self.embedding_model.embed_chunks(chunks)
        elif not embeddings:
            raise ValueError("必须提供embeddings或embedding_model")
            
        # 实现向量索引逻辑
        raise NotImplementedError()
    
    async def search(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """向量检索"""
        if not self.embedding_model:
            raise ValueError("未设置embedding_model")
            
        # 生成查询向量
        query_embedding = await self.embedding_model.embed(query)
        
        # 实现向量检索逻辑
        raise NotImplementedError()
    
    async def delete(
        self,
        chunk_ids: List[str]
    ) -> None:
        """删除向量索引"""
        # 实现删除逻辑
        raise NotImplementedError()


class HybridRetriever(Retriever):
    """混合检索器"""
    
    def __init__(
        self,
        config: RetrieverConfig,
        embedding_model: Optional[EmbeddingModel] = None,
        retrievers: List[Retriever] = None
    ):
        super().__init__(config, embedding_model)
        self.retrievers = retrievers or []
    
    async def index(
        self,
        chunks: List[Chunk],
        embeddings: Optional[List[EmbeddingOutput]] = None
    ) -> None:
        """索引到所有检索器"""
        for retriever in self.retrievers:
            await retriever.index(chunks, embeddings)
    
    async def search(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[SearchResult]:
        """混合检索"""
        all_results = []
        
        # 从所有检索器获取结果
        for retriever in self.retrievers:
            results = await retriever.search(query, top_k)
            all_results.extend(results)
            
        # 合并和排序结果
        sorted_results = sorted(
            all_results,
            key=lambda x: x.score,
            reverse=True
        )
        
        # 返回top_k个结果
        k = top_k or self.config.top_k
        return sorted_results[:k]
    
    async def delete(
        self,
        chunk_ids: List[str]
    ) -> None:
        """从所有检索器删除"""
        for retriever in self.retrievers:
            await retriever.delete(chunk_ids)


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