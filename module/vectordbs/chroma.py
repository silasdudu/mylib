"""
基于 Chroma 的向量数据库实现
"""
from typing import List, Optional, Dict, Any
import os
import json
import numpy as np
import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from chromadb.errors import InvalidCollectionException

from base.rag.vectordb import VectorDB, VectorDBConfig, SearchResult
from base.rag.chunking import Chunk, ChunkMetadata
from base.core.logging import LogLevel, AsyncLogger


class ChromaVectorDBConfig(VectorDBConfig):
    """Chroma 向量数据库配置"""
    persist_directory: str = "data/chroma"  # 持久化目录
    collection_name: str = "document_chunks"  # 集合名称
    distance_metric: str = "cosine"  # 距离度量方式：cosine, l2, ip
    metadata_fields: List[str] = []  # 要索引的元数据字段


class ChromaVectorDB(VectorDB):
    """基于 Chroma 的向量数据库实现"""
    
    def __init__(self, config: ChromaVectorDBConfig, logger: Optional[AsyncLogger] = None):
        """初始化 Chroma 向量数据库
        
        Args:
            config: 数据库配置
            logger: 可选的日志记录器
        """
        super().__init__(config, logger)
        self.config: ChromaVectorDBConfig = config
        
        # 初始化 ChromaDB 客户端
        self._client = chromadb.Client(Settings(
            persist_directory=config.persist_directory,
            anonymized_telemetry=False
        ))
        
        # 获取或创建集合
        self._collection = self._get_or_create_collection()
        
    def _get_or_create_collection(self) -> Collection:
        """获取或创建集合"""
        try:
            return self._client.get_collection(
                name=self.config.collection_name,
                embedding_function=None  # 我们自己处理向量生成
            )
        except InvalidCollectionException:
            # 如果collection不存在，就创建一个新的
            return self._client.create_collection(
                name=self.config.collection_name,
                embedding_function=None,
                metadata={"dimension": self.config.dimension}
            )
            
    def _metadata_to_chunk(self, chunk_id: str, text: str, metadata: Dict[str, Any]) -> Chunk:
        """从元数据构建 Chunk 对象
        
        Args:
            chunk_id: 块ID
            text: 文本内容
            metadata: 元数据
            
        Returns:
            Chunk 对象
        """
        chunk_metadata = ChunkMetadata(
            chunk_id=chunk_id,
            doc_id=metadata.get("doc_id", ""),
            start_char=metadata.get("start_char", 0),
            end_char=metadata.get("end_char", len(text)),
            text_len=len(text),
            extra=metadata
        )
        return Chunk(text=text, metadata=chunk_metadata)
        
    def _prepare_metadata(self, chunk: Chunk) -> Dict[str, Any]:
        """准备要存储的元数据
        
        Args:
            chunk: 文档块
            
        Returns:
            元数据字典
        """
        metadata = {
            "doc_id": chunk.metadata.doc_id,
            "start_char": chunk.metadata.start_char,
            "end_char": chunk.metadata.end_char,
            "text_len": chunk.metadata.text_len
        }
        
        # 添加额外的元数据字段
        if chunk.metadata.extra:
            for field in self.config.metadata_fields:
                if field in chunk.metadata.extra:
                    metadata[field] = chunk.metadata.extra[field]
                    
        return metadata
    
    async def create_index(self, vectors: List[List[float]], chunks: List[Chunk]) -> None:
        """创建向量索引
        
        Args:
            vectors: 向量列表
            chunks: 对应的文档块列表
        """
        await self._log(LogLevel.INFO, f"创建索引，向量数量: {len(vectors)}")
        
        # 验证向量格式
        vectors_array = self._validate_vectors(vectors)
        
        # 准备数据
        chunk_ids = [chunk.metadata.chunk_id for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        metadatas = [self._prepare_metadata(chunk) for chunk in chunks]
        
        # 添加到集合
        self._collection.add(
            ids=chunk_ids,
            embeddings=vectors_array.tolist(),
            documents=texts,
            metadatas=metadatas
        )
        
        self._index_ready = True
        await self._log(LogLevel.INFO, "索引创建完成")
        
    async def add_vectors(self, vectors: List[List[float]], chunks: List[Chunk]) -> None:
        """添加向量到索引
        
        Args:
            vectors: 向量列表
            chunks: 对应的文档块列表
        """
        await self._log(LogLevel.INFO, f"添加向量，数量: {len(vectors)}")
        
        # 验证向量格式
        vectors_array = self._validate_vectors(vectors)
        
        # 准备数据
        chunk_ids = [chunk.metadata.chunk_id for chunk in chunks]
        texts = [chunk.text for chunk in chunks]
        metadatas = [self._prepare_metadata(chunk) for chunk in chunks]
        
        # 添加到集合
        self._collection.add(
            ids=chunk_ids,
            embeddings=vectors_array.tolist(),
            documents=texts,
            metadatas=metadatas
        )
        
        await self._log(LogLevel.INFO, "向量添加完成")
        
    async def delete_vectors(self, chunk_ids: List[str]) -> None:
        """从索引中删除向量
        
        Args:
            chunk_ids: 要删除的文档块ID列表
        """
        await self._log(LogLevel.INFO, f"删除向量，数量: {len(chunk_ids)}")
        self._collection.delete(ids=chunk_ids)
        await self._log(LogLevel.INFO, "向量删除完成")
        
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
        if not self._index_ready:
            await self._log(LogLevel.WARNING, "索引尚未创建")
            return []
            
        # 验证查询向量
        query_array = self._validate_vectors([query_vector])
        
        # 执行搜索
        k = top_k or self.config.top_k
        await self._log(LogLevel.INFO, f"执行搜索，top_k={k}")
        
        results = self._collection.query(
            query_embeddings=query_array.tolist(),
            n_results=k,
            where=filter_params
        )
        
        await self._log(LogLevel.INFO, f"搜索结果: {results}")
        
        # 构建结果
        search_results = []
        if results["ids"]:
            for i, (chunk_id, distance) in enumerate(zip(results["ids"][0], results["distances"][0])):
                # 从ChromaDB结果直接构建Chunk对象
                chunk = self._metadata_to_chunk(
                    chunk_id=chunk_id,
                    text=results["documents"][0][i],
                    metadata=results["metadatas"][0][i]
                )
                
                # 计算相似度分数
                if self.config.distance_metric == "cosine":
                    score = 1 - distance
                else:  # l2 或 ip
                    score = 1 / (1 + distance)
                
                search_results.append(SearchResult(
                    chunk=chunk,
                    score=float(score)
                ))
                    
        await self._log(LogLevel.INFO, f"构建了 {len(search_results)} 个搜索结果")
        return search_results
        
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
        if not self._index_ready:
            await self._log(LogLevel.WARNING, "索引尚未创建")
            return []
            
        # 验证查询向量
        query_array = self._validate_vectors(query_vectors)
        
        # 执行批量搜索
        k = top_k or self.config.top_k
        results = self._collection.query(
            query_embeddings=query_array.tolist(),
            n_results=k,
            where=filter_params
        )
        
        # 构建结果
        batch_results = []
        if results["ids"]:
            for batch_idx, (batch_ids, batch_distances) in enumerate(zip(results["ids"], results["distances"])):
                query_results = []
                for i, (chunk_id, distance) in enumerate(zip(batch_ids, batch_distances)):
                    # 从ChromaDB结果直接构建Chunk对象
                    chunk = self._metadata_to_chunk(
                        chunk_id=chunk_id,
                        text=results["documents"][batch_idx][i],
                        metadata=results["metadatas"][batch_idx][i]
                    )
                    
                    # 计算相似度分数
                    if self.config.distance_metric == "cosine":
                        score = 1 - distance
                    else:  # l2 或 ip
                        score = 1 / (1 + distance)
                    
                    query_results.append(SearchResult(
                        chunk=chunk,
                        score=float(score)
                    ))
                batch_results.append(query_results)
                
        return batch_results
        
    async def save(self, path: str) -> None:
        """保存索引到文件
        
        Args:
            path: 保存路径
        """
        # ChromaDB 已经自动持久化，不需要额外保存
        pass
        
    async def load(self, path: str) -> None:
        """从文件加载索引
        
        Args:
            path: 索引文件路径
        """
        # ChromaDB 已经自动持久化，不需要额外加载
        self._index_ready = True 